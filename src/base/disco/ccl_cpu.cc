#include "base/disco/ccl_backend.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/packedfunc.h"
#include "base/registry.h"

namespace kxc {
namespace disco {

namespace {

std::string DTypeToString(const DLDataType& dtype) {
    if (dtype.lanes != 1) {
        throw std::runtime_error("CpuCCLBackend only supports lanes=1");
    }
    if (dtype.code == kDLFloat && dtype.bits == 32) return "float32";
    if (dtype.code == kDLFloat && dtype.bits == 64) return "float64";
    if (dtype.code == kDLInt && dtype.bits == 32) return "int32";
    if (dtype.code == kDLInt && dtype.bits == 64) return "int64";
    if (dtype.code == kDLInt && dtype.bits == 8) return "int8";
    throw std::runtime_error("Unsupported dtype in CpuCCLBackend");
}

int64_t NumElements(const runtime::NDArray& array) {
    if (!array.defined()) {
        return 0;
    }
    int64_t n = 1;
    for (int64_t dim : array->shape) {
        n *= dim;
    }
    return n;
}

size_t NBytes(const runtime::NDArray& array) {
    int64_t numel = NumElements(array);
    int64_t lane = array->dl_tensor.dtype.lanes;
    int64_t bytes_per_elem = (array->dl_tensor.dtype.bits / 8) * lane;
    return static_cast<size_t>(numel * bytes_per_elem);
}

runtime::NDArray CloneArray(const runtime::NDArray& src) {
    if (!src.defined()) {
        return runtime::NDArray();
    }
    Array<int64_t> shape;
    for (int64_t dim : src->shape) {
        shape.push_back(dim);
    }
    runtime::NDArray dst(shape, DTypeToString(src->dl_tensor.dtype));
    std::memcpy(dst->dl_tensor.data, src->dl_tensor.data, NBytes(src));
    return dst;
}

void AddInPlace(const runtime::NDArray& acc, const runtime::NDArray& other) {
    if (!acc.defined() || !other.defined()) {
        throw std::runtime_error("AllReduce requires defined arrays");
    }
    if (NumElements(acc) != NumElements(other)) {
        throw std::runtime_error("AllReduce requires equal tensor size");
    }

    const DLDataType dtype = acc->dl_tensor.dtype;
    int64_t numel = NumElements(acc);
    if (dtype.code == kDLFloat && dtype.bits == 32) {
        float* a = static_cast<float*>(acc->dl_tensor.data);
        const float* b = static_cast<const float*>(other->dl_tensor.data);
        for (int64_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLFloat && dtype.bits == 64) {
        double* a = static_cast<double*>(acc->dl_tensor.data);
        const double* b = static_cast<const double*>(other->dl_tensor.data);
        for (int64_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLInt && dtype.bits == 32) {
        int32_t* a = static_cast<int32_t*>(acc->dl_tensor.data);
        const int32_t* b = static_cast<const int32_t*>(other->dl_tensor.data);
        for (int64_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLInt && dtype.bits == 64) {
        int64_t* a = static_cast<int64_t*>(acc->dl_tensor.data);
        const int64_t* b = static_cast<const int64_t*>(other->dl_tensor.data);
        for (int64_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    throw std::runtime_error("Unsupported dtype for sum allreduce");
}

runtime::NDArray SliceFirstDim(const runtime::NDArray& src, int64_t shard_start,
                               int64_t shard_len) {
    if (!src.defined()) {
        return runtime::NDArray();
    }
    if (src->shape.empty()) {
        throw std::runtime_error("Scatter expects rank >= 1 tensor");
    }

    int64_t dim0 = src->shape[0];
    if (shard_start < 0 || shard_len < 0 || shard_start + shard_len > dim0) {
        throw std::runtime_error("Scatter shard out of bounds");
    }

    Array<int64_t> out_shape;
    out_shape.push_back(shard_len);
    for (size_t i = 1; i < src->shape.size(); ++i) {
        out_shape.push_back(src->shape[i]);
    }

    runtime::NDArray dst(out_shape, DTypeToString(src->dl_tensor.dtype));
    int64_t inner = 1;
    for (size_t i = 1; i < src->shape.size(); ++i) {
        inner *= src->shape[i];
    }
    size_t bytes_per_elem = static_cast<size_t>(src->dl_tensor.dtype.bits / 8);
    size_t copy_bytes = static_cast<size_t>(shard_len * inner) * bytes_per_elem;
    size_t src_offset = static_cast<size_t>(shard_start * inner) * bytes_per_elem;
    std::memcpy(dst->dl_tensor.data, static_cast<const uint8_t*>(src->dl_tensor.data) + src_offset,
                copy_bytes);
    return dst;
}

runtime::NDArray ConcatFirstDim(const std::vector<runtime::NDArray>& arrays) {
    if (arrays.empty()) {
        return runtime::NDArray();
    }
    const runtime::NDArray& first = arrays[0];
    if (!first.defined()) {
        return runtime::NDArray();
    }
    if (first->shape.empty()) {
        throw std::runtime_error("Gather expects rank >= 1 tensor");
    }

    int64_t total_dim0 = 0;
    for (const auto& arr : arrays) {
        if (!arr.defined()) {
            throw std::runtime_error("Gather source contains undefined shard");
        }
        total_dim0 += arr->shape[0];
    }

    Array<int64_t> out_shape;
    out_shape.push_back(total_dim0);
    for (size_t i = 1; i < first->shape.size(); ++i) {
        out_shape.push_back(first->shape[i]);
    }
    runtime::NDArray dst(out_shape, DTypeToString(first->dl_tensor.dtype));

    int64_t inner = 1;
    for (size_t i = 1; i < first->shape.size(); ++i) {
        inner *= first->shape[i];
    }
    size_t bytes_per_elem = static_cast<size_t>(first->dl_tensor.dtype.bits / 8);
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst->dl_tensor.data);
    size_t offset = 0;
    for (const auto& arr : arrays) {
        size_t bytes = static_cast<size_t>(arr->shape[0] * inner) * bytes_per_elem;
        std::memcpy(dst_ptr + offset, arr->dl_tensor.data, bytes);
        offset += bytes;
    }
    return dst;
}

class CpuCCLBackend final : public CCLBackend {
public:
    void Copy(const DiscoSession& session, const DRef& src, const DRef& dst, int src_worker,
              int dst_worker) override {
        runtime::NDArray src_array = session.Get(src_worker, src);
        if (!src_array.defined()) {
            throw std::runtime_error("kxc.disco.copy source is undefined");
        }
        session.Set(dst_worker, dst, CloneArray(src_array));
    }

    void AllReduce(const DiscoSession& session, const DRef& src, const DRef& dst,
                   const std::string& reduce_kind, bool in_group) override {
        (void)in_group;
        if (reduce_kind != "sum") {
            throw std::runtime_error("CpuCCLBackend supports only sum allreduce");
        }
        runtime::NDArray acc;
        for (int worker = 0; worker < session.num_workers(); ++worker) {
            runtime::NDArray value = session.Get(worker, src);
            if (!value.defined()) continue;
            if (!acc.defined()) {
                acc = CloneArray(value);
            } else {
                AddInPlace(acc, value);
            }
        }
        if (!acc.defined()) {
            throw std::runtime_error("AllReduce has no source tensors");
        }
        for (int worker = 0; worker < session.num_workers(); ++worker) {
            session.Set(worker, dst, CloneArray(acc));
        }
    }

    void BroadcastFromWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                              bool in_group) override {
        runtime::NDArray root = session.Get(0, src);
        if (!root.defined()) {
            throw std::runtime_error("Broadcast source on worker0 is undefined");
        }
        if (!in_group) {
            for (int worker = 0; worker < session.num_workers(); ++worker) {
                session.Set(worker, dst, CloneArray(root));
            }
            return;
        }
        int groups = session.num_groups();
        int workers = session.num_workers();
        int workers_per_group = workers / groups;
        for (int group = 0; group < groups; ++group) {
            int group_root = group * workers_per_group;
            runtime::NDArray value = session.Get(group_root, src);
            if (!value.defined()) {
                value = root;
            }
            for (int i = 0; i < workers_per_group; ++i) {
                session.Set(group * workers_per_group + i, dst, CloneArray(value));
            }
        }
    }

    void ScatterFromWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                            bool in_group) override {
        runtime::NDArray root = session.Get(0, src);
        if (!root.defined()) {
            throw std::runtime_error("Scatter source on worker0 is undefined");
        }

        int receivers = in_group ? (session.num_workers() / session.num_groups())
                                 : session.num_workers();
        if (root->shape.empty() || root->shape[0] % receivers != 0) {
            throw std::runtime_error(
                "Scatter expects first dimension divisible by receiver count");
        }
        int64_t shard = root->shape[0] / receivers;

        if (!in_group) {
            for (int worker = 0; worker < session.num_workers(); ++worker) {
                session.Set(worker, dst, SliceFirstDim(root, worker * shard, shard));
            }
            return;
        }

        int workers_per_group = session.num_workers() / session.num_groups();
        for (int group = 0; group < session.num_groups(); ++group) {
            int root_worker = group * workers_per_group;
            runtime::NDArray group_src = session.Get(root_worker, src);
            if (!group_src.defined()) {
                group_src = root;
            }
            for (int i = 0; i < workers_per_group; ++i) {
                int worker = root_worker + i;
                session.Set(worker, dst, SliceFirstDim(group_src, i * shard, shard));
            }
        }
    }

    void GatherToWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                         bool in_group) override {
        if (!in_group) {
            std::vector<runtime::NDArray> shards;
            for (int worker = 0; worker < session.num_workers(); ++worker) {
                shards.push_back(session.Get(worker, src));
            }
            session.Set(0, dst, ConcatFirstDim(shards));
            return;
        }

        int workers_per_group = session.num_workers() / session.num_groups();
        for (int group = 0; group < session.num_groups(); ++group) {
            int root_worker = group * workers_per_group;
            std::vector<runtime::NDArray> shards;
            for (int i = 0; i < workers_per_group; ++i) {
                shards.push_back(session.Get(root_worker + i, src));
            }
            session.Set(root_worker, dst, ConcatFirstDim(shards));
        }
    }

    void SendToWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                      int receiver_worker) override {
        runtime::NDArray value = session.Get(0, src);
        if (!value.defined()) {
            throw std::runtime_error("SendToWorker expects source on worker0");
        }
        session.Set(receiver_worker, dst, CloneArray(value));
    }

    void RecvFromWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                        int sender_worker) override {
        runtime::NDArray value = session.Get(sender_worker, src);
        if (!value.defined()) {
            throw std::runtime_error("RecvFromWorker source is undefined");
        }
        session.Set(0, dst, CloneArray(value));
    }

    void SyncWorker(const DiscoSession& session, int worker_id) override {
        session.SyncWorker(worker_id);
    }
};

std::shared_ptr<CCLBackend> GlobalCpuBackend() {
    static std::shared_ptr<CCLBackend> backend = std::make_shared<CpuCCLBackend>();
    return backend;
}

}  // namespace

std::shared_ptr<CCLBackend> CreateCpuCCLBackend() {
    return GlobalCpuBackend();
}

KXC_REGISTER_GLOBAL("kxc.disco.copy")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, int src_worker,
                              int dst_worker) {
        GlobalCpuBackend()->Copy(session, src, dst, src_worker, dst_worker);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.allreduce")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, std::string reduce_kind,
                              bool in_group) {
        GlobalCpuBackend()->AllReduce(session, src, dst, reduce_kind, in_group);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.broadcast_from_worker0")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, bool in_group) {
        GlobalCpuBackend()->BroadcastFromWorker0(session, src, dst, in_group);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.scatter_from_worker0")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, bool in_group) {
        GlobalCpuBackend()->ScatterFromWorker0(session, src, dst, in_group);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.gather_to_worker0")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, bool in_group) {
        GlobalCpuBackend()->GatherToWorker0(session, src, dst, in_group);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.send_to_worker")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, int receiver_worker) {
        GlobalCpuBackend()->SendToWorker(session, src, dst, receiver_worker);
    }));

KXC_REGISTER_GLOBAL("kxc.disco.recv_from_worker")
    .set_body(ToPackedFunc([](DiscoSession session, DRef src, DRef dst, int sender_worker) {
        GlobalCpuBackend()->RecvFromWorker(session, src, dst, sender_worker);
    }));

}  // namespace disco
}  // namespace kxc
