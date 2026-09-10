/*! \file src/distributed/threaded_session.cc
 * \brief 实现进程内 Disco 寄存器、引用保活与会话生命周期。
 */

#include "kxc/distributed/session.h"
#include "kxc/support/object_registration.h"

#include <mutex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"

namespace kxc {
namespace disco {

KXC_OBJECT_DEFINE(DiscoSessionNode)

namespace {

// 用进程内互斥寄存器文件模拟多 worker Disco 会话。
class ThreadedDiscoSessionNode final : public DiscoSessionNode {
public:
    // Groups are contiguous, equal-sized worker ranges.
    ThreadedDiscoSessionNode(int num_workers, int num_groups)
        : num_workers_(num_workers), num_groups_(num_groups) {
        if (num_workers <= 0 || num_groups <= 0 || num_groups > num_workers || num_workers % num_groups != 0) {
            throw std::invalid_argument("DiscoSession requires positive workers divisible by positive groups");
        }
        register_file_.resize(static_cast<size_t>(num_workers));
    }

    // 返回会话 worker 数量。
    int num_workers() const override { return num_workers_; }
    // 返回通信分组数量。
    int num_groups() const override { return num_groups_; }

    // 原子扩展所有 worker 的寄存器文件并返回统一寄存器编号。
    int AllocateRegister() override {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireActive();
        if (reg_count_ == std::numeric_limits<int>::max()) throw std::overflow_error("DiscoSession register space exhausted");
        const int reg_id = reg_count_;
        for (auto& regs : register_file_) {
            regs.resize(static_cast<size_t>(reg_count_) + 1);
        }
        register_refs_.push_back(kUnowned);
        ++reg_count_;
        return reg_id;
    }

    void RetainRegister(int reg_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireActive(); ValidateRegId(reg_id);
        auto& refs = register_refs_[static_cast<size_t>(reg_id)];
        if (refs == kUnowned) refs = 1;
        else {
            if (refs == kUnowned - 1) throw std::overflow_error("DiscoSession DRef count overflow");
            ++refs;
        }
    }
    void ReleaseRegister(int reg_id) noexcept override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_ || reg_id < 0 || reg_id >= reg_count_) return;
            auto& refs = register_refs_[static_cast<size_t>(reg_id)];
            if (refs == 0 || refs == kUnowned || --refs != 0) return;
        }
        // The dead id cannot be rebound. Retire one worker slot at a time,
        // without allocating in this noexcept destructor path. External Storage
        // deleters can call back into the session, so destruction is outside the lock.
        for (int worker = 0; worker < num_workers_; ++worker) {
            runtime::NDArray retired;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (shutdown_) return;
                retired = std::move(register_file_[static_cast<size_t>(worker)][static_cast<size_t>(reg_id)]);
            }
        }
    }

    // 在线程安全边界内读取指定 worker 的寄存器值。
    runtime::NDArray GetRegister(int worker_id, int reg_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireActive();
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        return register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)];
    }

    // 在线程安全边界内替换指定 worker 的寄存器值。
    void SetRegister(int worker_id, int reg_id, runtime::NDArray value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireActive();
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        // The parameter outlives this lock guard and destroys the old tensor
        // after unlocking, including any external deleter callback.
        std::swap(register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)], value);
    }

    // 线程会话当前同步执行，仅验证 worker 身份即可视为同步完成。
    void SyncWorker(int worker_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        RequireActive();
        ValidateWorkerId(worker_id);
    }

    // 在线程安全边界内记录会话关闭状态。
    void Shutdown() override {
        std::vector<std::vector<runtime::NDArray>> retired;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            retired.swap(register_file_);
            register_refs_.clear();
            reg_count_ = 0;
        }
    }

    KXC_OBJECT_DECLARE

private:
    void RequireActive() const {
        if (shutdown_) throw std::runtime_error("DiscoSession is shut down");
    }
    // 验证 worker 编号可索引寄存器文件。
    void ValidateWorkerId(int worker_id) const {
        if (worker_id < 0 || worker_id >= num_workers_) {
            throw std::runtime_error("DiscoSession worker_id out of range");
        }
    }

    // 验证寄存器编号已由当前会话分配。
    void ValidateRegId(int reg_id) const {
        if (reg_id < 0 || reg_id >= reg_count_ || register_refs_[static_cast<size_t>(reg_id)] == 0) {
            throw std::runtime_error("DiscoSession reg_id out of range");
        }
    }

    int num_workers_{1};
    int num_groups_{1};
    mutable std::mutex mutex_;
    int reg_count_{0};
    // ponytail: monotonically allocated ids prevent stale DRef resurrection;
    // released slots drop payloads, but metadata grows until session shutdown.
    // Add generation-tagged recycling only if long-session metadata is a bottleneck.
    bool shutdown_{false};
    static constexpr size_t kUnowned = std::numeric_limits<size_t>::max();
    std::vector<size_t> register_refs_;
    std::vector<std::vector<runtime::NDArray>> register_file_;
};

KXC_OBJECT_DEFINE(ThreadedDiscoSessionNode)

}  // namespace

// 返回经过 DiscoSession 类型约束的底层节点。
const DiscoSessionNode* DiscoSession::operator->() const {
    const auto* node = As<DiscoSessionNode>();
    if (!node) throw std::invalid_argument("expected DiscoSession");
    return node;
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
    if (!defined() || !ref.valid() || ref->session.get() != get()) {
        throw std::invalid_argument("DiscoSession::Get requires a DRef owned by this session");
    }
    return operator->()->GetRegister(worker_id, ref.reg_id());
}

// 设置 DRef 在指定 worker 上的本地 NDArray。
void DiscoSession::Set(int worker_id, const DRef& ref, runtime::NDArray value) const {
    if (!defined() || !ref.valid() || ref->session.get() != get()) {
        throw std::invalid_argument("DiscoSession::Set requires a DRef owned by this session");
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

namespace kxc::builtin_anchor {
void DistributedSession() {}
}  // namespace kxc::builtin_anchor
