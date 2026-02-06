#pragma once
#include <functional>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <iostream>
#include "object.h"
#include "tensor.h"

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
    kHandle = 5
};
/*
@brief 声明联合体：Value，在同一时间每个实例只存在一个成员，省内存不出错，类型切换方便
*/
union Value {
    int64_t v_int;
    double v_float;
    const char* v_str;
    const Object* v_object; // 指向 const Object*
    void* v_handle;
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
class PackedFuncObj : public Object {
public:
    std::function<void(Args, RetValue*)> func_;
    PackedFuncObj(const std::function<void(Args, RetValue*)>& f)
        : func_(f) {}
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(PackedFuncObj)

class PackedFunc : public ObjectRef {
public:
    PackedFunc() {}
    PackedFunc(std::function<void(Args, RetValue*)> f) : ObjectRef(new PackedFuncObj(f)) {
    }

    void operator()(Args args, RetValue* rv) const {
        const PackedFuncObj* obj = static_cast<const PackedFuncObj*>(object_);
        if (obj && obj->func_) {
            obj->func_(args, rv);
        } else {
            throw std::runtime_error("PackedFunc is empty");
        }
    }
    
    // Variadic call template
    template <typename... Args>
    RetValue operator()(Args&&... args) const;

    using ContainerType = PackedFuncObj;
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
        value_ = other.value_;
        type_code_ = other.type_code_;
        str_holder_ = other.str_holder_;
        if (type_code_ == kObjectRef) {
            obj_holder_ = other.obj_holder_;
            value_.v_object = obj_holder_.get();
        }
        return *this;
    }

    RetValue& operator=(int64_t v) {
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        value_.v_int = v;
        type_code_ = kInt;
        return *this;
    }
    RetValue& operator=(int v) {
        return operator=((int64_t)v);
    }
    RetValue& operator=(bool v) {
        return operator=((int64_t)v);
    }
    RetValue& operator=(double v) {
        // 清理旧的 ObjectRef
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        value_.v_float = v;
        type_code_ = kFloat;
        return *this;
    }
    RetValue& operator=(const ObjectRef& v) {
        obj_holder_ = v;
        value_.v_object = obj_holder_.get();
        type_code_ = kObjectRef;
        return *this;
    }
    RetValue& operator=(const std::string& v) {
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        str_holder_ = v;
        value_.v_str = str_holder_.c_str();
        type_code_ = kString;
        return *this;
    }
    RetValue& operator=(const char* v) {
        if (type_code_ == kObjectRef) obj_holder_ = ObjectRef(); 
        str_holder_ = v;
        value_.v_str = str_holder_.c_str();
        type_code_ = kString;
        return *this;
    }

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
        if (type_code_ == kObjectRef) return obj_holder_;
        if (type_code_ == kNull) return ObjectRef(nullptr);
        throw std::runtime_error("Type mismatch: expected ObjectRef");
    }
    
    template<typename T>
    T As() const;
};

template<>
inline int64_t RetValue::As<int64_t>() const {
    return (int64_t)(*this);
}
template<>
inline double RetValue::As<double>() const {
    return (double)(*this);
}
template<>
inline std::string RetValue::As<std::string>() const {
    return (std::string)(*this);
}
template<>
inline ObjectRef RetValue::As<ObjectRef>() const {
    return (ObjectRef)(*this);
}
template<>
inline const char* RetValue::As<const char*>() const {
    if (type_code_ == kString) return value_.v_str;
    throw std::runtime_error("Type mismatch: expected string");
}

template<typename T>
inline typename std::enable_if<std::is_base_of<ObjectRef, T>::value, T>::type 
CastTo(const RetValue& rv) {
    ObjectRef ref = rv.As<ObjectRef>();
    return T(ref);
}

// ArgConverter
template <typename T, typename = void>
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
template<> struct ArgConverter<bool> {
    static bool From(const Value& v, TypeCode t) {
        if (t != kInt) throw std::runtime_error("Type mismatch, expected bool (int)");
        return (bool)v.v_int;
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
        return std::string(v.v_str);
    }
};
template<> struct ArgConverter<ObjectRef> {
    static ObjectRef From(const Value& v, TypeCode t) {
        if (t != kObjectRef) throw std::runtime_error("Type mismatch, expected ObjectRef");
        return ObjectRef(v.v_object);
    }
};
template<> struct ArgConverter<const char*> {
    static const char* From(const Value& v, TypeCode t) {
        if (t != kString) throw std::runtime_error("Type mismatch, expected const char*");
        return v.v_str;
    }
};
// Subclass of ObjectRef
template <typename T>
struct ArgConverter<T, typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type> {
    static T From(const Value& v, TypeCode t) {
        if (t != kObjectRef) throw std::runtime_error("Type mismatch, expected ObjectRef");
        // Ensure T has constructor from ObjectRef or implicit conversion
        return T(ObjectRef(v.v_object));
    }
};
// Vector (Handle)
template <typename T>
struct ArgConverter<std::vector<T>> {
    static std::vector<T> From(const Value& v, TypeCode t) {
        if (t != kHandle) throw std::runtime_error("Type mismatch, expected Handle for vector");
        return *(std::vector<T>*)v.v_handle;
    }
};

template<> struct ArgConverter<Tensor> {
    static Tensor From(const Value& v, TypeCode t) {
        if (t != kObjectRef) throw std::runtime_error("Type mismatch, expected ObjectRef for Tensor");
        return Tensor(runtime::NDArray(v.v_object));
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

    template<typename Traits, typename F, size_t... I>
    void unpack_call(F&& f, Args args, RetValue* rv, std::index_sequence<I...>) {
        if (args.num_args != Traits::Arity) {
             throw std::runtime_error("Function arity mismatch");
        }
        if constexpr (std::is_void_v<typename Traits::ReturnType>) {
            std::invoke(std::forward<F>(f),
                ArgConverter<typename std::decay<typename Traits::template ArgType<I>>::type>::From(args[I], args.type_code(I))...
            );
            *rv = kNull;
        } else {
            *rv = std::invoke(std::forward<F>(f),
                ArgConverter<typename std::decay<typename Traits::template ArgType<I>>::type>::From(args[I], args.type_code(I))...
            );
        }
    }

    template<typename F>
    PackedFunc Wrap(F&& f) {
        using Traits = typename TFunctionTraits<std::decay_t<F>>::type;
        return PackedFunc([f_moved = std::forward<F>(f)](kxc::Args args, kxc::RetValue* rv) {
            constexpr size_t Arity = Traits::Arity;
            unpack_call<Traits>(f_moved, args, rv, std::make_index_sequence<Arity>{});
        });
    }

    // ArgsSetter
    struct ArgsSetter {
        Value* values;
        TypeCode* type_codes;
        ArgsSetter(Value* values, TypeCode* type_codes) : values(values), type_codes(type_codes) {}

        template <typename T>
        void operator()(size_t i, T&& value) const {
            Set(i, std::forward<T>(value));
        }
        
        void Set(size_t i, int v) const { values[i].v_int = v; type_codes[i] = kInt; }
        void Set(size_t i, int64_t v) const { values[i].v_int = v; type_codes[i] = kInt; }
        void Set(size_t i, bool v) const { values[i].v_int = v; type_codes[i] = kInt; }
        void Set(size_t i, double v) const { values[i].v_float = v; type_codes[i] = kFloat; }
        void Set(size_t i, const char* v) const { values[i].v_str = v; type_codes[i] = kString; }
        void Set(size_t i, const std::string& v) const { values[i].v_str = v.c_str(); type_codes[i] = kString; }
        
        template <typename T>
        typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type
        Set(size_t i, const T& v) const {
            values[i].v_object = v.get();
            type_codes[i] = kObjectRef;
        }
        
        // Handle vector
        template<typename T>
        void Set(size_t i, const std::vector<T>& v) const {
            values[i].v_handle = (void*)&v;
            type_codes[i] = kHandle;
        }

        void Set(size_t i, const Tensor& v) const {
            values[i].v_object = v.data_.get();
            type_codes[i] = kObjectRef;
        }
    };
    
    template <typename Setter, typename... Args>
    void for_each(Setter& setter, Args&&... args) {
        size_t i = 0;
        (setter(i++, std::forward<Args>(args)), ...);
    }
}

template <typename... Args>
RetValue PackedFunc::operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    Value values[kArraySize];
    TypeCode type_codes[kArraySize];
    detail::ArgsSetter setter(values, type_codes);
    detail::for_each(setter, std::forward<Args>(args)...);
    RetValue rv;
    (*this)(kxc::Args(values, type_codes, kNumArgs), &rv);
    return rv;
}

template<typename F>
PackedFunc ToPackedFunc(F&& f) {
    return detail::Wrap(std::forward<F>(f));
}

} // namespace kxc
