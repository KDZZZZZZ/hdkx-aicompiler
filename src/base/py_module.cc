/*! \file src/base/py_module.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/py_runtime.h"

PYBIND11_MODULE(kxc_runtime, m) {
    m.doc() = "KXC Runtime Python Bindings";
    kxc::InitKXCRuntime(m);
}
