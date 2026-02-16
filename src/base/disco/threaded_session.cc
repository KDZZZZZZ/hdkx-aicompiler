#include "base/disco/session.h"

#include <mutex>
#include <stdexcept>
#include <vector>

#include "base/packedfunc.h"
#include "base/registry.h"

namespace kxc {
namespace disco {

namespace {

class ThreadedDiscoSessionNode final : public DiscoSessionNode {
public:
    ThreadedDiscoSessionNode(int num_workers, int num_groups)
        : num_workers_(num_workers > 0 ? num_workers : 1),
          num_groups_(num_groups > 0 ? num_groups : 1),
          register_file_(static_cast<size_t>(num_workers_)) {}

    int num_workers() const override { return num_workers_; }
    int num_groups() const override { return num_groups_; }

    int AllocateRegister() override {
        std::lock_guard<std::mutex> lock(mutex_);
        int reg_id = reg_count_++;
        for (auto& regs : register_file_) {
            regs.resize(static_cast<size_t>(reg_count_));
        }
        return reg_id;
    }

    runtime::NDArray GetRegister(int worker_id, int reg_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        return register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)];
    }

    void SetRegister(int worker_id, int reg_id, runtime::NDArray value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ValidateWorkerId(worker_id);
        ValidateRegId(reg_id);
        register_file_[static_cast<size_t>(worker_id)][static_cast<size_t>(reg_id)] =
            std::move(value);
    }

    void SyncWorker(int worker_id) override {
        ValidateWorkerId(worker_id);
    }

    void Shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }

    KXC_OBJECT_DECLARE

private:
    void ValidateWorkerId(int worker_id) const {
        if (worker_id < 0 || worker_id >= num_workers_) {
            throw std::runtime_error("DiscoSession worker_id out of range");
        }
    }

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

const DiscoSessionNode* DiscoSession::operator->() const {
    return static_cast<const DiscoSessionNode*>(object_);
}

DRef DiscoSession::NewDRef() const {
    if (!defined()) {
        throw std::runtime_error("NewDRef requires a defined DiscoSession");
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    int reg_id = node->AllocateRegister();
    return DRef(reg_id, ObjectRef(*this));
}

DRef DiscoSession::Empty(Array<int64_t> shape, std::string dtype, bool worker0_only,
                         bool in_group) const {
    DRef ref = NewDRef();
    if (!worker0_only) {
        for (int worker = 0; worker < num_workers(); ++worker) {
            Set(worker, ref, runtime::NDArray(shape, dtype));
        }
        return ref;
    }

    if (!in_group) {
        Set(0, ref, runtime::NDArray(shape, dtype));
        return ref;
    }

    int groups = num_groups();
    int workers = num_workers();
    int workers_per_group = workers / groups;
    for (int group = 0; group < groups; ++group) {
        int worker = group * workers_per_group;
        Set(worker, ref, runtime::NDArray(shape, dtype));
    }
    return ref;
}

runtime::NDArray DiscoSession::Get(int worker_id, const DRef& ref) const {
    if (!defined() || !ref.valid()) {
        return runtime::NDArray();
    }
    return operator->()->GetRegister(worker_id, ref.reg_id());
}

void DiscoSession::Set(int worker_id, const DRef& ref, runtime::NDArray value) const {
    if (!defined() || !ref.valid()) {
        throw std::runtime_error("DiscoSession::Set requires valid session and DRef");
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->SetRegister(worker_id, ref.reg_id(), std::move(value));
}

int DiscoSession::num_workers() const {
    if (!defined()) {
        return 0;
    }
    return operator->()->num_workers();
}

int DiscoSession::num_groups() const {
    if (!defined()) {
        return 0;
    }
    return operator->()->num_groups();
}

void DiscoSession::SyncWorker(int worker_id) const {
    if (!defined()) {
        return;
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->SyncWorker(worker_id);
}

void DiscoSession::Shutdown() const {
    if (!defined()) {
        return;
    }
    DiscoSessionNode* node = const_cast<DiscoSessionNode*>(operator->());
    node->Shutdown();
}

DiscoSession DiscoSession::ThreadedSession(int num_workers, int num_groups) {
    return DiscoSession(ObjectRef(new ThreadedDiscoSessionNode(num_workers, num_groups)));
}

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
    .set_body(ToPackedFunc([](DiscoSession session, std::vector<int64_t> shape_vec,
                              std::string dtype, bool worker0_only,
                              bool in_group) -> ObjectRef {
        Array<int64_t> shape;
        for (int64_t dim : shape_vec) {
            shape.push_back(dim);
        }
        return ObjectRef(session.Empty(shape, dtype, worker0_only, in_group));
    }));

}  // namespace disco
}  // namespace kxc
