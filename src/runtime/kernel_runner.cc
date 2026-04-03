#include "runtime/kernel_runner.h"

#include <stdexcept>

namespace kxc {
namespace runtime {

void KernelRunner::SwapKernel(std::shared_ptr<api::CompiledModule> module) {
    std::lock_guard<std::mutex> lock(mu_);
    module_ = std::move(module);
}

void KernelRunner::Run(const std::vector<void*>& args) {
    // 快速路径：读取shared_ptr（原子操作）
    std::shared_ptr<api::CompiledModule> local_module;
    {
        std::lock_guard<std::mutex> lock(mu_);
        local_module = module_;
    }

    if (!local_module) {
        throw std::runtime_error("KernelRunner: no kernel available");
    }

    local_module->Run(args);
}

bool KernelRunner::HasKernel() const {
    std::lock_guard<std::mutex> lock(mu_);
    return module_ != nullptr;
}

}  // namespace runtime
}  // namespace kxc
