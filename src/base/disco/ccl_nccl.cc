#include "base/disco/ccl_backend.h"

#include <memory>
#include <stdexcept>

namespace kxc {
namespace disco {

class NcclCCLBackend final : public CCLBackend {
public:
    void Copy(const DiscoSession&, const DRef&, const DRef&, int, int) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void AllReduce(const DiscoSession&, const DRef&, const DRef&, const std::string&,
                   bool) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void BroadcastFromWorker0(const DiscoSession&, const DRef&, const DRef&, bool) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void ScatterFromWorker0(const DiscoSession&, const DRef&, const DRef&, bool) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void GatherToWorker0(const DiscoSession&, const DRef&, const DRef&, bool) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void SendToWorker(const DiscoSession&, const DRef&, const DRef&, int) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void RecvFromWorker(const DiscoSession&, const DRef&, const DRef&, int) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
    void SyncWorker(const DiscoSession&, int) override {
        throw std::runtime_error("NCCL backend is not implemented in this phase");
    }
};

}  // namespace disco
}  // namespace kxc

