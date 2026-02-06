#include "base/py_runtime.h"

PYBIND11_MODULE(kxc_runtime, m) {
    m.doc() = "KXC Runtime Python Bindings";
    kxc::InitKXCRuntime(m);
}
