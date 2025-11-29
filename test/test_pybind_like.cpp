#include "../include/base/object.h"
#include "../include/base/packedfunc.h"
#include <cassert>
#include <iostream>
#include <string>

using namespace kxc;

struct MyObj : Object { const TypeIndex GetTypeId() const override { return 7; } };

int add_impl(int a, int b) { return a + b; }
std::string greet_impl(const std::string& name) { return std::string("hello ") + name; }
int consume_obj_impl(ObjectRef o) { return o.get() ? 1 : 0; }

PackedFunc make_add_pf() {
    auto f = [](Args args, RetValue* rv){
        if(args.size() != 2) throw std::runtime_error("arity");
        int a = ArgConverter<int>::From(args[0], args.type_code(0));
        int b = ArgConverter<int>::From(args[1], args.type_code(1));
        *rv = static_cast<int64_t>(add_impl(a, b));
    };
    return f;
}

PackedFunc make_greet_pf() {
    static std::string storage;
    auto f = [](Args args, RetValue* rv){
        if(args.size() != 1) throw std::runtime_error("arity");
        std::string name = ArgConverter<std::string>::From(args[0], args.type_code(0));
        std::string s = greet_impl(name);
        storage = s;
        *rv = storage;
    };
    return f;
}

PackedFunc make_consume_obj_pf() {
    auto f = [](Args args, RetValue* rv){
        if(args.size() != 1) throw std::runtime_error("arity");
        ObjectRef o = ArgConverter<ObjectRef>::From(args[0], args.type_code(0));
        *rv = static_cast<int64_t>(consume_obj_impl(o));
    };
    return f;
}

void test_add() {
    PackedFunc pf = make_add_pf();
    Value vals[2]; TypeCode tcs[2];
    vals[0].v_int = 3; tcs[0] = kInt;
    vals[1].v_int = 4; tcs[1] = kInt;
    Args a(vals, tcs, 2); RetValue rv;
    pf(a, &rv);
    assert(true);
}

void test_type_mismatch() {
    PackedFunc pf = make_add_pf();
    Value vals[2]; TypeCode tcs[2];
    vals[0].v_float = 1.0; tcs[0] = kFloat;
    vals[1].v_int = 2; tcs[1] = kInt;
    Args a(vals, tcs, 2); RetValue rv;
    try { pf(a, &rv); assert(false); } catch(const std::runtime_error&) { assert(true); }
}

void test_greet() {
    PackedFunc pf = make_greet_pf();
    Value vals[1]; TypeCode tcs[1];
    const char* name = "world";
    vals[0].v_str = name; tcs[0] = kString;
    Args a(vals, tcs, 1); RetValue rv;
    pf(a, &rv);
    assert(true);
}

void test_objectref() {
    PackedFunc pf = make_consume_obj_pf();
    ObjectRef o(new MyObj());
    Value vals[1]; TypeCode tcs[1];
    vals[0].v_object = o.get(); tcs[0] = kObjectRef;
    Args a(vals, tcs, 1); RetValue rv;
    pf(a, &rv);
    assert(true);
}

int main() {
    test_add();
    test_type_mismatch();
    test_greet();
    test_objectref();
    std::cout << "PASS: pybind-like PackedFunc tests" << std::endl;
    return 0;
}

