/*! \file include/runtime/kernel_runner.h
 * \brief 定义 adaptive runtime、shape predictor、kernel cache 和后台编译器。
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include "api/compiler.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

/*! \brief Kernel 执行器，封装当前可执行模块并支持线程安全热替换。 */
class KernelRunner {
public:
    KernelRunner() = default;

    /*! \brief 替换当前执行用 kernel。 */
    void SwapKernel(std::shared_ptr<api::CompiledModule> module);

    /*! \brief 使用当前 kernel 执行一次调用。 */
    void Run(const std::vector<void*>& args);

    /*! \brief 判断当前是否已有可执行 kernel。 */
    bool HasKernel() const;

private:
    std::shared_ptr<api::CompiledModule> module_;
    mutable std::mutex mu_;
};

}  // namespace runtime
}  // namespace kxc
