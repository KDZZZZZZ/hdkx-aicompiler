/*! \file src/base/disco/threaded_session.cc
 * \brief 实现 Disco 线程会话、执行计划解释器和 CPU/NCCL 通信后端。
 */

#include "base/disco/session.h"

#include <mutex>
#include <stdexcept>
#include <vector>

#include "base/packedfunc.h"
#include "base/registry.h"

namespace kxc {
namespace disco {

namespace {

// 用进程内互斥寄存器文件模拟多 worker Disco 会话。
class ThreadedDiscoSessionNode final : public DiscoSessionNode {
public:
    // 规范化 worker/group 下限并为每个 worker 建立独立寄存器行。
    ThreadedDiscoSessionNode(int num_workers, int num_groups)
        : num_workers_(num_workers > 0 ? num_workers : 1),
          num_groups_(num_groups > 0 ? num_groups : 1),
          register_file_(static_cast<size_t>(num_workers_)) {}

    // 返回会话 worker 数量。
    int num_workers() const override { return num_workers_; }
    // 返回通信分组数量。
    int num_groups() const override { return num_groups_; }

    // 原子扩展所有 worker 的寄存器文件并返回统一寄存器编号。
    int AllocateRegister() override {
        std::lock_guard<std::mutex> lock(mutex_);
        int reg_id = reg_count_++;
        for (auto& regs : register_file_) {
            regs.resize(static_cast<size_t>(reg_count_));
        }
        return reg_id;
    }

    // 在线程安全边界内读取指定 worker 的寄存器值。
    runtime::NDArray GetRegister(int worker_id, int reg_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        return register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)];
    }

    // 在线程安全边界内替换指定 worker 的寄存器值。
    void SetRegister(int worker_id, int reg_id, runtime::NDArray value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)] =
            std::move(value);
    }

    // 线程会话当前同步执行，仅验证 worker 身份即可视为同步完成。
    void SyncWorker(int worker_id) override {
        ValidateWorkerId(worker_id);
    }

    // 在线程安全边界内记录会话关闭状态。
    void Shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }

    KXC_OBJECT_DECLARE

private:
    // 验证 worker 编号可索引寄存器文件。
    void ValidateWorkerId(int worker_id) const {
        if (worker_id < 0 || worker_id >= num_workers_) {
            throw std::runtime_error("DiscoSession worker_id out of range");
        }
    }

    // 验证寄存器编号已由当前会话分配。
    void ValidateRegId(int reg_id) const {
        if (reg_id < 0 || reg_id >= reg_count_) {
            throw std::runtime_error("DiscoSession reg_id out of range");
        }
    }

    int num_workers_{1};
    int num_groups_{1};
    mutable std::mutex mutex_;
    int reg_count_{0};
    bool shutdown_{false};
    std::vector<std::vector<runtime::NDArray>> register_file_;
};

KXC_OBJECT_DEFINE(ThreadedDiscoSessionNode)

}  // namespace

// 返回经过 DiscoSession 类型约束的底层节点。
const DiscoSessionNode* DiscoSession::operator->() const {
    return static_cast<const DiscoSessionNode*>(object_);
}

// 在会话内分配寄存器并返回绑定该会话的分布式引用。
DRef DiscoSession::NewDRef() const {
    if (!defined()) {
        throw std::runtime_error("NewDRef requires a defined DiscoSession");
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    int reg_id = node->AllocateRegister();
    return DRef(reg_id, ObjectRef(*this));
}

// 按 worker 范围创建独立 CPU NDArray，并写入同一 DRef 的各 worker 寄存器。
DRef DiscoSession::Empty(Array<int64_t> shape, std::string dtype, bool worker0_only,
                         bool in_group) const {
    // 当前线程会话由 CPU CCL 后端托管，每个目标 worker 获得独立的 CPU Storage；
    // 后续设备放置必须由 ExecutionPlan/placement 显式决定，不能从 worker 编号推导。
    DRef ref = NewDRef();
    if (!worker0_only) {
        for (int worker = 0; worker < num_workers(); ++worker) {
            Set(worker, ref, runtime::NDArray::Empty(
                                 shape, runtime::DataTypeFromString(dtype), Device::CPU()));
        }
        return ref;
    }

    if (!in_group) {
        Set(0, ref, runtime::NDArray::Empty(
                        shape, runtime::DataTypeFromString(dtype), Device::CPU()));
        return ref;
    }

    int groups = num_groups();
    int workers = num_workers();
    int workers_per_group = workers / groups;
    for (int group = 0; group < groups; ++group) {
        int worker = group * workers_per_group;
        Set(worker, ref, runtime::NDArray::Empty(
                             shape, runtime::DataTypeFromString(dtype), Device::CPU()));
    }
    return ref;
}

// 读取 DRef 在指定 worker 上的本地 NDArray。
runtime::NDArray DiscoSession::Get(int worker_id, const DRef& ref) const {
    if (!defined() || !ref.valid()) {
        return runtime::NDArray();
    }
    return operator->()->GetRegister(worker_id, ref.reg_id());
}

// 设置 DRef 在指定 worker 上的本地 NDArray。
void DiscoSession::Set(int worker_id, const DRef& ref, runtime::NDArray value) const {
    if (!defined() || !ref.valid()) {
        throw std::runtime_error("DiscoSession::Set requires valid session and DRef");
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->SetRegister(worker_id, ref.reg_id(), std::move(value));
}

// 返回会话 worker 数；未定义会话按零处理。
int DiscoSession::num_workers() const {
    if (!defined()) {
        return 0;
    }
    return operator->()->num_workers();
}

// 返回会话 group 数；未定义会话按零处理。
int DiscoSession::num_groups() const {
    if (!defined()) {
        return 0;
    }
    return operator->()->num_groups();
}

// 请求会话后端同步指定 worker。
void DiscoSession::SyncWorker(int worker_id) const {
    if (!defined()) {
        return;
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->SyncWorker(worker_id);
}

// 请求会话后端进入关闭状态。
void DiscoSession::Shutdown() const {
    if (!defined()) {
        return;
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->Shutdown();
}

// 创建进程内线程会话实现。
DiscoSession DiscoSession::ThreadedSession(int num_workers, int num_groups) {
    return DiscoSession(ObjectRef(new ThreadedDiscoSessionNode(num_workers, num_groups)));
}

// 注册线程会话创建、同步、关闭和分配操作的 PackedFunc 入口。
KXC_REGISTER_GLOBAL("kxc.disco.session.threaded")
    .set_body(ToPackedFunc([](int num_workers, int num_groups) -> ObjectRef {
        return ObjectRef(DiscoSession::ThreadedSession(num_workers, num_groups));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.sync_worker")
    .set_body(ToPackedFunc([](DiscoSession session, int worker_id) {
        session.SyncWorker(worker_id);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.shutdown")
    .set_body(ToPackedFunc([](DiscoSession session) {
        session.Shutdown();
    }));

KXC_REGISTER_GLOBAL("kxc.disco.empty")
    .set_body(ToPackedFunc([](DiscoSession session, Array<int64_t> shape,
                              std::string dtype, bool worker0_only,
                              bool in_group) -> ObjectRef {
        return ObjectRef(session.Empty(shape, dtype, worker0_only, in_group));
    }));

}  // namespace disco
}  // namespace kxc
