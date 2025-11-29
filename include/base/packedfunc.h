#pragma once
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <tuple>
#include "object.h" // 确保引用的是最新的 object.h
#include <iostream> // 用于调试信息

namespace kxc {

/*
@brief 声明在前
*/
class Args;
class RetValue;
/*
@brief 声明枚举类型：TypeCode，用于表示参数类型，这里使用枚举是为了方便类型检查和编译期计算
*/
enum TypeCode : int {
    kInt = 0,
    kFloat = 1,
    kString = 2,
    kObjectRef = 3,
    kNull = 4,
    // KDeviceRef 现在不需要了，所有 Object 都通过 kObjectRef 传递，
    // 具体类型在 C++ 端通过 Object::GetTypeId() 或 As<T>() 判断
};
/*
@brief 声明联合体：Value，在同一时间每个实例只存在一个成员，省内存不出错，类型切换方便
*/
union Value {
    int64_t v_int;
    double v_float;
    const char* v_str;
    const Object* v_object; // 指向 const Object*
    // 注意：联合体不能包含非平凡类型（如 std::string, ObjectRef），
    // 只能包含 POD 类型或具有平凡构造/析构/拷贝的类型。
    // 所以 v_str 和 v_object 都是裸指针，生命周期由外部管理。
};
/*
@brief 声明类：Args，用于表示函数参数，包含参数值、参数类型和参数数量，
    提供访问参数值和参数类型的方法，以及参数数量的查询方法
*/
class Args {
public:
    // 注意：这里的 values 和 type_codes 只是指向外部数据的指针，
    // Args 不拥有这些数据，所以不需要析构函数来释放它们。
    // 引用计数由 PackedFunc 的 __call__ 中的 `IncRef` 和 `RetValue` 的析构函数负责。
    const Value* values;
    const TypeCode* type_codes;
    int num_args;

    Args(const Value* values, const TypeCode* type_codes, int num_args)
        : values(values), type_codes(type_codes), num_args(num_args) {}

    int size() const { return num_args; }

    const Value& operator[](int index) const {
        if (index >= num_args) throw std::out_of_range("index out of range");
        return values[index];
    }
    TypeCode type_code(int index) const {
        if (index >= num_args) throw std::out_of_range("index out of range");
        return type_codes[index];
    }
};

/*
@brief 声明类型别名：PackedFunc，参数为Args，返回值为RetValue*
*/
// 注意：PackedFunc 自身也需要能够作为 ObjectRef 传入和传出
// 为了简化，我们暂时让 PackedFunc 接受非 ObjectRef 的参数，
// 但实际中 PackedFunc 也应该是一个 Object 的子类
class PackedFunc_Internal {
public:
    // 包装 std::function
    PackedFunc_Internal(std::function<void(Args, RetValue*)> f) : func_(f) {}

    // 重载操作符以便像函数一样调用
    void operator()(Args args, RetValue* rv) const {
        if (func_) {
            func_(args, rv);
        } else {
            throw std::runtime_error("PackedFunc is not set.");
        }
    }

    explicit operator bool() const {
        return func_ != nullptr;
    }

private:
    std::function<void(Args, RetValue*)> func_;
};

// 实际暴露给外部的 PackedFunc 是 ObjectRef，包装了 PackedFunc_Internal
// 但为了保持你的原始 PackedFunc 类型定义，我们这里稍微调整下
// 如果 PackedFunc 要作为 Object 传递，它本身需要是一个 Object 子类
// 假设 PackedFunc 作为一个 Object 的子类
class PackedFunc : public Object {
public:
    PackedFunc(std::function<void(Args, RetValue*)> f = nullptr) : internal_func_(f) {}

    void operator()(Args args, RetValue* rv) const {
        internal_func_(args, rv);
    }
    
    explicit operator bool() const {
        return (bool)internal_func_;
    }

    const TypeIndex GetTypeId() const override {
        // 为 PackedFunc 分配一个唯一的 TypeIndex
        return kKXC_OBJECT_TYPE + 2; // 假设 Device 是 +1
    }

private:
    PackedFunc_Internal internal_func_;
};

/*
@brief 声明类：RetValue，用于表示函数返回值，包含返回值和返回值类型，
    提供赋值运算符和类型查询方法
*/
class RetValue {
private:
    Value value_;
    TypeCode type_code_;
    std::string str_holder_; // 用于延长字符串生命周期
    ObjectRef obj_holder_;   // 用于延长 ObjectRef 的生命周期

public:
    TypeCode type_code() const { return type_code_; }

    RetValue() : type_code_(kNull) {}
    
    // 析构函数：确保正确减少 Object 的引用计数
    ~RetValue() {
        if (type_code_ == kObjectRef) {
            // obj_holder_ 的析构会自动 DecRef
            // 如果 obj_holder_ 持有的是 value_.v_object, 那么不需要额外操作
            // 如果 obj_holder_ 是通过 ObjectRef(value_.v_object) 构造的，它会管理
        }
    }

    // 拷贝构造和赋值运算符：确保 ObjectRef 的引用计数正确
    RetValue(const RetValue& other) : 
        value_(other.value_), 
        type_code_(other.type_code_),
        str_holder_(other.str_holder_)
    {
        if (type_code_ == kObjectRef) {
            obj_holder_ = other.obj_holder_; // 拷贝 ObjectRef，IncRef
            value_.v_object = obj_holder_.get(); // 指向 obj_holder_ 持有的对象
        }
    }

    RetValue& operator=(const RetValue& other) {
        if (this == &other) return *this;
        
        // 先清理当前持有的 Object
        if (type_code_ == kObjectRef && obj_holder_.get()) {
            // obj_holder_ 赋值前会自动 DecRef
        }

        value_ = other.value_;
        type_code_ = other.type_code_;
        str_holder_ = other.str_holder_;
        
        if (type_code_ == kObjectRef) {
            obj_holder_ = other.obj_holder_; // 拷贝 ObjectRef，IncRef
            value_.v_object = obj_holder_.get();
        }
        return *this;
    }


    RetValue& operator=(int64_t v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        value_.v_int = v;
        type_code_ = kInt;
        return *this;
    }
    RetValue& operator=(int v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        value_.v_int = v;
        type_code_ = kInt;
        return *this;
    }
    RetValue& operator=(double v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        value_.v_float = v;
        type_code_ = kFloat;
        return *this;
    }
    RetValue& operator=(const ObjectRef& v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        obj_holder_ = v; // 让 obj_holder_ 管理引用计数
        value_.v_object = obj_holder_.get(); // value_.v_object 指向 obj_holder_ 持有的对象
        type_code_ = kObjectRef;
        return *this;
    }
    RetValue& operator=(const std::string& v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        str_holder_ = v; // 复制字符串到成员变量
        value_.v_str = str_holder_.c_str(); // 指向内部存储
        type_code_ = kString;
        return *this;
    }
    RetValue& operator=(const char* v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        str_holder_ = v; // 复制字符串到成员变量
        value_.v_str = str_holder_.c_str(); // 指向内部存储
        type_code_ = kString;
        return *this;
    }

    // 类型转换操作符 (As<T> 更好，但为了兼容你的现有结构)
    operator int64_t() const {
        if (type_code_ == kInt) return value_.v_int;
        if (type_code_ == kFloat) return static_cast<int64_t>(value_.v_float);
        throw std::runtime_error("Type mismatch: expected int");
    }
    operator double() const {
        if (type_code_ == kFloat) return value_.v_float;
        if (type_code_ == kInt) return static_cast<double>(value_.v_int);
        throw std::runtime_error("Type mismatch: expected double");
    }
    operator std::string() const {
        if (type_code_ == kString) return value_.v_str;
        throw std::runtime_error("Type mismatch: expected string");
    }
    operator ObjectRef() const {
        if (type_code_ == kObjectRef) return obj_holder_; // 直接返回内部 ObjectRef 副本
        if (type_code_ == kNull) return ObjectRef(nullptr);
        throw std::runtime_error("Type mismatch: expected ObjectRef");
    }
    
    // As<T> 辅助函数
    template<typename T>
    T As() const;

    template<>
    int64_t As<int64_t>() const {
        if (type_code_ == kInt) return value_.v_int;
        if (type_code_ == kFloat) return static_cast<int64_t>(value_.v_float);
        throw std::runtime_error("Type mismatch: expected int");
    }
    template<>
    double As<double>() const {
        if (type_code_ == kFloat) return value_.v_float;
        if (type_code_ == kInt) return static_cast<double>(value_.v_int);
        throw std::runtime_error("Type mismatch: expected double");
    }
    template<>
    std::string As<std::string>() const {
        if (type_code_ == kString) return value_.v_str;
        throw std::runtime_error("Type mismatch: expected string");
    }
    template<>
    const char* As<const char*>() const {
        if (type_code_ == kString) return value_.v_str;
        throw std::runtime_error("Type mismatch: expected string");
    }
    template<>
    ObjectRef As<ObjectRef>() const {
        if (type_code_ == kObjectRef) return obj_holder_;
        if (type_code_ == kNull) return ObjectRef(nullptr);
        throw std::runtime_error("Type mismatch: expected ObjectRef");
    }
};

/*
@brief 声明结构体：ArgConverter，用于将Args中的参数值转换为指定类型，
    提供静态方法From，参数为参数值和参数类型，返回值为转换后的参数值，
    抛出运行时错误如果类型不匹配
*/
template <typename T>
struct ArgConverter;

template <typename T>
struct ArgConverter<const T&> {
    static T From(const Value& v, TypeCode t) {
        return ArgConverter<T>::From(v, t);
    }
};

template <typename T>
struct ArgConverter<T&> {
    static T From(const Value& v, TypeCode t) {
        return ArgConverter<T>::From(v, t);
    }
};

template<> struct ArgConverter<int> {
    static int From(const Value& v, TypeCode t) {
        if (t != kInt) throw std::runtime_error("Type mismatch, expected int");
        return static_cast<int>(v.v_int);
    }
};
template<> struct ArgConverter<int64_t> {
    static int64_t From(const Value& v, TypeCode t) {
        if (t != kInt) throw std::runtime_error("Type mismatch, expected int64_t");
        return v.v_int;
    }
};
template<> struct ArgConverter<double> {
    static double From(const Value& v, TypeCode t) {
        if (t != kFloat) throw std::runtime_error("Type mismatch, expected double");
        return v.v_float;
    }
};
template<> struct ArgConverter<std::string> {
    static std::string From(const Value& v, TypeCode t) {
        if (t != kString) throw std::runtime_error("Type mismatch, expected string");
        // 注意：v.v_str 的生命周期由外部控制，这里复制到 std::string 是安全的
        return std::string(v.v_str);
    }
};
template<> struct ArgConverter<ObjectRef> {
    static ObjectRef From(const Value& v, TypeCode t) {
        if (t != kObjectRef) throw std::runtime_error("Type mismatch, expected ObjectRef");
        // 从 Value 构造 ObjectRef，ObjectRef 会增加引用计数
        return ObjectRef(v.v_object);
    }
};
template<> struct ArgConverter<const char*> { // 支持直接传入 const char*
    static const char* From(const Value& v, TypeCode t) {
        if (t != kString) throw std::runtime_error("Type mismatch, expected const char*");
        return v.v_str;
    }
};


namespace detail {
template <typename T>
struct TFunctionTraits;
/*
真正做事的版本
@brief 声明结构体：TFunctionTraits，用于获取函数的参数数量、返回值类型和参数类型，
    提供静态常量成员Arity、ReturnType和ArgsTuple，以及模板方法ArgType用于获取第I个参数类型，
    这里使用了模板元编程技术，在编译期计算参数数量和参数类型，避免了运行时的类型检查和转换
*/
template<typename Ret, typename... Args>
struct TFunctionTraits<Ret(Args ...)> {
    using type = TFunctionTraits;
    static constexpr size_t Arity = sizeof...(Args);
    using ReturnType = Ret;
    using ArgsTuple = std::tuple<Args...>;
    template <size_t I>
    using ArgType = typename std::tuple_element<I, ArgsTuple>::type;
};
/*忽略掉了类型中的 C::* 和 const 部分，只把 Ret(Args...) 这个纯函数签名传递给基础版本去分析。*/
template <typename Ret, typename... Args>
struct TFunctionTraits<Ret(*)(Args ...)> : TFunctionTraits<Ret(Args...)> {};
template<typename C, typename Ret, typename... Args>
struct TFunctionTraits<Ret(C::*)(Args...) const> : TFunctionTraits<Ret(Args...)> {};
template<typename C, typename Ret, typename... Args>
struct TFunctionTraits<Ret(C::*)(Args...)> : TFunctionTraits<Ret(Args...)> {};
/*
The Entry Point
传入的是functor，然后获取其中的成员函数指针，然后传给他自己的其他特化版本。
*/
template<typename Functor>
struct TFunctionTraits {
private:
    template<typename F> static std::true_type check(decltype(&F::operator()));
    template<typename F> static std::false_type check(...);
    // 允许传入 lambda，所以这里不再强求是 functor，如果不是 lambda 且没有 operator()，会是编译错误
    // static_assert(std::is_same_v<decltype(check<Functor>()), std::true_type>, "Not a functor");
public:
    // 如果是 lambda 或函数对象，获取其 operator() 的签名
    // 如果是普通函数指针，则 TFunctionTraits<Ret(*)(Args ...)> 会匹配
    using type = std::conditional_t<
        std::is_function<std::remove_pointer_t<Functor>>::value || std::is_member_function_pointer<Functor>::value,
        TFunctionTraits<Functor>,
        TFunctionTraits<decltype(&Functor::operator())>
    >;
};

// 重载 unpack_call 以便 WrappedFunc 可以返回 ObjectRef
template<typename Traits, typename F, size_t... I>
    void unpack_call(F&& f, Args args, RetValue* rv, std::index_sequence<I...>) {
        if (args.num_args != Traits::Arity) {
             throw std::runtime_error("Function arity mismatch: expected " + std::to_string(Traits::Arity) +
                                         ", got " + std::to_string(args.num_args));
        }

        // 解包参数并调用函数
        if constexpr (std::is_void_v<typename Traits::ReturnType>) {
            std::invoke(std::forward<F>(f),
                ArgConverter<typename Traits::template ArgType<I>>::From(args[I], args.type_code(I))...
            );
            *rv = kNull; // void 返回值设置为 Null
        } else {
            *rv = std::invoke(std::forward<F>(f),
                ArgConverter<typename Traits::template ArgType<I>>::From(args[I], args.type_code(I))...
            );
        }
    }

    template<typename F>
    PackedFunc Wrap(F&& f) {
        /*@brief TFunctionTraits结构体的别名*/
        using Traits = typename TFunctionTraits<std::decay_t<F>>::type;
        return PackedFunc([f_moved = std::forward<F>(f)](kxc::Args args, kxc::RetValue* rv) {
            constexpr size_t Arity = Traits::Arity;
            // 内部的 unpack_call 会处理参数数量检查
            unpack_call<Traits>(f_moved, args, rv, std::make_index_sequence<Arity>{});
        });
    }
} // namespace detail
template<typename F>
PackedFunc ToPackedFunc(F&& f) {
    return detail::Wrap(std::forward<F>(f));
}

} // namespace kxc
