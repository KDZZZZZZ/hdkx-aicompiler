/*! \file src/base/op.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "relay/op.h"
#include <mutex>

namespace kxc {
namespace relay {

// Implementation of Op Registry
class OpRegistry {
public:
    static OpRegistry* Global() {
        static OpRegistry inst;
        return &inst;
    }

    const Op& Get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = op_map_.find(name);
        if (it != op_map_.end()) {
            return it->second;
        }
        // Create new if not exists (auto-registration style for simplicity here)
        // In TVM, this would throw error if not registered via TVM_REGISTER_OP
        Op new_op(name);
        op_map_.insert({name, new_op});
        return op_map_.at(name);
    }

private:
    std::unordered_map<std::string, Op> op_map_;
    std::mutex mutex_;
};

const Op& Op::Get(const std::string& name) {
    return OpRegistry::Global()->Get(name);
}

} // namespace relay
} // namespace kxc
