/*! \file include/kxc/ffi/detail/packed_func_inl.h
 * \brief Visible template implementation for PackedFunc adapters.
 *
 * This file is included by packedfunc.h and is not a stable direct-include API.
 */

#pragma once

namespace kxc {

template<typename T>
inline typename std::enable_if<std::is_base_of<ObjectRef, T>::value, T>::type
CastTo(const RetValue& rv) {
    ObjectRef ref = rv.As<ObjectRef>();
    return T(ref);
}

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
    static int From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<int64_t> {
    static int64_t From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<bool> {
    static bool From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<double> {
    static double From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<std::string> {
    static std::string From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<ObjectRef> {
    static ObjectRef From(const Value& v, TypeCode t);
};
template<> struct ArgConverter<const char*> {
    static const char* From(const Value& v, TypeCode t);
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
        ArgsSetter(Value* values, TypeCode* type_codes);

        template <typename T>
        void operator()(size_t i, T&& value) const {
            Set(i, std::forward<T>(value));
        }

        void Set(size_t i, int v) const;
        void Set(size_t i, int64_t v) const;
        void Set(size_t i, bool v) const;
        void Set(size_t i, double v) const;
        void Set(size_t i, const char* v) const;
        void Set(size_t i, const std::string& v) const;

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

}  // namespace kxc
