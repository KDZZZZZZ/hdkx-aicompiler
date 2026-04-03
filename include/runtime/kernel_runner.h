#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include "api/compiler.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

// Kernel执行器：支持热替换
// Run路径使用shared_ptr + atomic load，几乎无锁
class KernelRunner {
public:
    KernelRunner() = default;

    // 热替换kernel（线程安全）
    void SwapKernel(std::shared_ptr<api::CompiledModule> module);

    // 执行kernel
    void Run(const std::vector<void*>& args);

    // 是否有可用kernel
    bool HasKernel() const;

private:
    std::shared_ptr<api::CompiledModule> module_;
    mutable std::mutex mu_;
};

}  // namespace runtime
}  // namespace kxc
