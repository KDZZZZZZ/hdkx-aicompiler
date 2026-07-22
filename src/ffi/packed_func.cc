/*! \file src/ffi/packed_func.cc
 * \brief Implements the non-template PackedFunc ABI core and value conversions.
 */

#include "kxc/ffi/packed_func.h"

#include "kxc/support/object_registration.h"

namespace kxc {

Args::Args(const Value* values, const TypeCode* type_codes, int num_args)
    : values(values), type_codes(type_codes), num_args(num_args) {}

const Value& Args::operator[](int index) const {
    if (index < 0 || index >= num_args) throw std::out_of_range("index out of range");
    return values[index];
}

TypeCode Args::type_code(int index) const {
    if (index < 0 || index >= num_args) throw std::out_of_range("index out of range");
    return type_codes[index];
}

PackedFuncObj::PackedFuncObj(const std::function<void(Args, RetValue*)>& f) : func_(f) {}

KXC_OBJECT_DEFINE(PackedFuncObj)

PackedFunc::PackedFunc() = default;

PackedFunc::PackedFunc(std::function<void(Args, RetValue*)> f)
    : ObjectRef(new PackedFuncObj(f)) {}

void PackedFunc::operator()(Args args, RetValue* rv) const {
    const auto* obj = static_cast<const PackedFuncObj*>(object_);
    if (!obj || !obj->func_) throw std::runtime_error("PackedFunc is empty");
    obj->func_(args, rv);
}

RetValue::RetValue() : type_code_(kNull) {}

RetValue::~RetValue() = default;

RetValue::RetValue(const RetValue& other)
    : value_(other.value_),
      type_code_(other.type_code_),
      str_holder_(other.str_holder_) {
    if (type_code_ == kObjectRef) {
        obj_holder_ = other.obj_holder_;
        value_.v_object = obj_holder_.get();
    }
}

RetValue& RetValue::operator=(const RetValue& other) {
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

RetValue& RetValue::operator=(int64_t v) {
    if (type_code_ == kObjectRef) obj_holder_ = ObjectRef();
    value_.v_int = v;
    type_code_ = kInt;
    return *this;
}

RetValue& RetValue::operator=(int v) {
    return operator=(static_cast<int64_t>(v));
}

RetValue& RetValue::operator=(bool v) {
    return operator=(static_cast<int64_t>(v));
}

RetValue& RetValue::operator=(double v) {
    if (type_code_ == kObjectRef) obj_holder_ = ObjectRef();
    value_.v_float = v;
    type_code_ = kFloat;
    return *this;
}

RetValue& RetValue::operator=(const ObjectRef& v) {
    obj_holder_ = v;
    value_.v_object = obj_holder_.get();
    type_code_ = kObjectRef;
    return *this;
}

RetValue& RetValue::operator=(const std::string& v) {
    if (type_code_ == kObjectRef) obj_holder_ = ObjectRef();
    str_holder_ = v;
    value_.v_str = str_holder_.c_str();
    type_code_ = kString;
    return *this;
}

RetValue& RetValue::operator=(const char* v) {
    if (type_code_ == kObjectRef) obj_holder_ = ObjectRef();
    str_holder_ = v;
    value_.v_str = str_holder_.c_str();
    type_code_ = kString;
    return *this;
}

RetValue::operator int64_t() const {
    if (type_code_ == kInt) return value_.v_int;
    if (type_code_ == kFloat) return static_cast<int64_t>(value_.v_float);
    throw std::runtime_error("Type mismatch: expected int");
}

RetValue::operator double() const {
    if (type_code_ == kFloat) return value_.v_float;
    if (type_code_ == kInt) return static_cast<double>(value_.v_int);
    throw std::runtime_error("Type mismatch: expected double");
}

RetValue::operator std::string() const {
    if (type_code_ == kString) return value_.v_str;
    throw std::runtime_error("Type mismatch: expected string");
}

RetValue::operator ObjectRef() const {
    if (type_code_ == kObjectRef) return obj_holder_;
    if (type_code_ == kNull) return ObjectRef(nullptr);
    throw std::runtime_error("Type mismatch: expected ObjectRef");
}

template<> int64_t RetValue::As<int64_t>() const { return static_cast<int64_t>(*this); }
template<> double RetValue::As<double>() const { return static_cast<double>(*this); }
template<> std::string RetValue::As<std::string>() const {
    return static_cast<std::string>(*this);
}
template<> ObjectRef RetValue::As<ObjectRef>() const { return static_cast<ObjectRef>(*this); }
template<> const char* RetValue::As<const char*>() const {
    if (type_code_ == kString) return value_.v_str;
    throw std::runtime_error("Type mismatch: expected string");
}

int ArgConverter<int>::From(const Value& v, TypeCode t) {
    if (t != kInt) throw std::runtime_error("Type mismatch, expected int");
    return static_cast<int>(v.v_int);
}

int64_t ArgConverter<int64_t>::From(const Value& v, TypeCode t) {
    if (t != kInt) throw std::runtime_error("Type mismatch, expected int64_t");
    return v.v_int;
}

bool ArgConverter<bool>::From(const Value& v, TypeCode t) {
    if (t != kInt) throw std::runtime_error("Type mismatch, expected bool (int)");
    return static_cast<bool>(v.v_int);
}

double ArgConverter<double>::From(const Value& v, TypeCode t) {
    if (t != kFloat) throw std::runtime_error("Type mismatch, expected double");
    return v.v_float;
}

std::string ArgConverter<std::string>::From(const Value& v, TypeCode t) {
    if (t != kString) throw std::runtime_error("Type mismatch, expected string");
    return std::string(v.v_str);
}

ObjectRef ArgConverter<ObjectRef>::From(const Value& v, TypeCode t) {
    if (t != kObjectRef) throw std::runtime_error("Type mismatch, expected ObjectRef");
    return ObjectRef(v.v_object);
}

const char* ArgConverter<const char*>::From(const Value& v, TypeCode t) {
    if (t != kString) throw std::runtime_error("Type mismatch, expected const char*");
    return v.v_str;
}

namespace detail {

ArgsSetter::ArgsSetter(Value* values, TypeCode* type_codes)
    : values(values), type_codes(type_codes) {}

void ArgsSetter::Set(size_t i, int v) const {
    values[i].v_int = v;
    type_codes[i] = kInt;
}

void ArgsSetter::Set(size_t i, int64_t v) const {
    values[i].v_int = v;
    type_codes[i] = kInt;
}

void ArgsSetter::Set(size_t i, bool v) const {
    values[i].v_int = v;
    type_codes[i] = kInt;
}

void ArgsSetter::Set(size_t i, double v) const {
    values[i].v_float = v;
    type_codes[i] = kFloat;
}

void ArgsSetter::Set(size_t i, const char* v) const {
    values[i].v_str = v;
    type_codes[i] = kString;
}

void ArgsSetter::Set(size_t i, const std::string& v) const {
    values[i].v_str = v.c_str();
    type_codes[i] = kString;
}

}  // namespace detail

}  // namespace kxc
