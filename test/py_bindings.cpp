#include <string>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/base/registry.h"
#include "../include/base/packedfunc.h"
#include "../include/base/object.h"
#include "../include/base/py_runtime.h"

namespace py = pybind11;
using namespace kxc;

// 用于测试的 MyObj 也需要被 pybind11 感知
struct MyObj : public kxc::Object {
    const kxc::TypeIndex GetTypeId() const override { return 7; }
    // 可以添加一些 Python 可调用的方法
    ::std::string name = "MyTestObject";
    ::std::string Greet() { return "Hello from " + name; }
};

// --- Pybind11 模块定义 ---
PYBIND11_MODULE(kxc_runtime, m) {
    m.doc() = "kxc_runtime: The AI compiler runtime module";

    kxc::InitKXCRuntime(m);

    // 暴露 MyObj 用于测试
    py::class_<MyObj, Object>(m, "MyObj")
        .def(py::init<>())
        .def_readwrite("name", &MyObj::name)
        .def("greet", &MyObj::Greet);
}

// 在 C++ 端注册一些函数以供测试
int add(int a, int b) { return a + b; }
std::string greet(const std::string& name) { return "Hello, " + name; }
ObjectRef process_obj(ObjectRef obj) {
    if (auto* my_obj = dynamic_cast<const MyObj*>(obj.get())) {
        std::cout << "Processing object in C++: " << my_obj->name << std::endl;
    }
    return obj; // 返回相同的对象
}

// 注册函数
// KXC_REGISTER_GLOBAL(add);
// KXC_REGISTER_GLOBAL(greet);
// KXC_REGISTER_GLOBAL(process_obj);
// Update to new registry syntax
KXC_REGISTER_GLOBAL("add").set_body(ToPackedFunc(add));
KXC_REGISTER_GLOBAL("greet").set_body(ToPackedFunc(greet));
KXC_REGISTER_GLOBAL("process_obj").set_body(ToPackedFunc(process_obj));
