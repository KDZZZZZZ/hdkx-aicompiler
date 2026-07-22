/*! \file src/distributed/ccl_cpu.cc
 * \brief 实现 Disco 线程会话、执行计划解释器和 CPU/NCCL 通信后端。
 */

#include "kxc/distributed/ccl_backend.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"

namespace kxc {
namespace disco {

namespace {

// 统一检查 CPU collective 只接收已定义、连续且位于 cpu:0 的张量。
void RequireCPU(const runtime::NDArray& array) {
    if (!array.defined() || array.device() != Device::CPU() || !array.IsContiguous()) {
        throw std::runtime_error("CpuCCLBackend requires contiguous cpu:0 arrays");
    }
}

// 返回 CPU 张量视图的首字节；零元素张量合法返回 nullptr。
void* RawData(const runtime::NDArray& array) {
    RequireCPU(array);
    void* base = array.storage().data();
    if (!base) {
        if (array.NBytes() == 0) return nullptr;
        throw std::runtime_error("non-empty CPU NDArray has null storage");
    }
    return static_cast<uint8_t*>(base) + array->byte_offset;
}

// 完整比较 dtype 的 code、bits 和 lanes，防止按错误元素宽度访问。
bool SameDType(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

// 比较 collective 输入的完整逻辑 shape。
bool SameShape(const runtime::NDArray& lhs, const runtime::NDArray& rhs) {
    return lhs->shape_storage == rhs->shape_storage;
}

// 分配独立 CPU Storage 并复制张量，避免不同 worker 意外共享可变地址。
runtime::NDArray CloneArray(const runtime::NDArray& src) {
    if (!src.defined()) {
        return runtime::NDArray();
    }
    Array<int64_t> shape;
    for (int64_t dim : src->shape_storage) {
        shape.push_back(dim);
    }
    runtime::NDArray dst = runtime::NDArray::Empty(shape, src.dtype(), Device::CPU());
    dst.CopyFrom(src);
    return dst;
}

// 对受支持的标量 dtype 执行原地求和，并在解引用前校验完整布局契约。
void AddInPlace(const runtime::NDArray& acc, const runtime::NDArray& other) {
    if (!acc.defined() || !other.defined()) {
        throw std::runtime_error("AllReduce requires defined arrays");
    }
    RequireCPU(acc);
    RequireCPU(other);
    if (!SameShape(acc, other) || !SameDType(acc.dtype(), other.dtype())) {
        throw std::runtime_error("AllReduce requires equal shape and dtype");
    }

    const DLDataType dtype = acc->dl_tensor.dtype;
    if (dtype.lanes != 1) {
        throw std::runtime_error("AllReduce only supports dtype lanes=1");
    }
    const size_t element_bytes =
        static_cast<size_t>(dtype.bits / 8) * dtype.lanes;
    const size_t numel = acc.NBytes() / element_bytes;
    if (numel == 0) return;
    if (dtype.code == kDLFloat && dtype.bits == 32) {
        float* a = static_cast<float*>(RawData(acc));
        const float* b = static_cast<const float*>(RawData(other));
        for (size_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLFloat && dtype.bits == 64) {
        double* a = static_cast<double*>(RawData(acc));
        const double* b = static_cast<const double*>(RawData(other));
        for (size_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLInt && dtype.bits == 32) {
        int32_t* a = static_cast<int32_t*>(RawData(acc));
        const int32_t* b = static_cast<const int32_t*>(RawData(other));
        for (size_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    if (dtype.code == kDLInt && dtype.bits == 64) {
        int64_t* a = static_cast<int64_t*>(RawData(acc));
        const int64_t* b = static_cast<const int64_t*>(RawData(other));
        for (size_t i = 0; i < numel; ++i) a[i] += b[i];
        return;
    }
    throw std::runtime_error("Unsupported dtype for sum allreduce");
}

// 沿首维复制连续分片，供 scatter 为目标 worker 创建独立 Storage。
runtime::NDArray SliceFirstDim(const runtime::NDArray& src, int64_t shard_start,
                               int64_t shard_len) {
    if (!src.defined()) {
        return runtime::NDArray();
    }
    RequireCPU(src);
    if (src->shape_storage.empty()) {
        throw std::runtime_error("Scatter expects rank >= 1 tensor");
    }

    int64_t dim0 = src->shape_storage[0];
    if (shard_start < 0 || shard_len < 0 || shard_start > dim0 ||
        shard_len > dim0 - shard_start) {
        throw std::runtime_error("Scatter shard out of bounds");
    }

    Array<int64_t> out_shape;
    out_shape.push_back(shard_len);
    for (size_t i = 1; i < src->shape_storage.size(); ++i) {
        out_shape.push_back(src->shape_storage[i]);
    }

    runtime::NDArray dst = runtime::NDArray::Empty(out_shape, src.dtype(), Device::CPU());
    const size_t copy_bytes = dst.NBytes();
    if (copy_bytes == 0) return dst;
    // 连续张量按首维均分字节跨度，并在地址计算前完成溢出及容量校验。
    size_t src_offset = 0;
    if (dim0 != 0) {
        const size_t leading_dim = static_cast<size_t>(dim0);
        if (src.NBytes() % leading_dim != 0) {
            throw std::runtime_error("Scatter source byte size is inconsistent");
        }
        const size_t row_bytes = src.NBytes() / leading_dim;
        const size_t start = static_cast<size_t>(shard_start);
        if (row_bytes != 0 && start > std::numeric_limits<size_t>::max() / row_bytes) {
            throw std::overflow_error("Scatter source offset overflow");
        }
        src_offset = start * row_bytes;
    }
    if (src_offset > src.NBytes() || copy_bytes > src.NBytes() - src_offset) {
        throw std::out_of_range("Scatter source byte range exceeds tensor");
    }
    std::memcpy(RawData(dst), static_cast<const uint8_t*>(RawData(src)) + src_offset,
                copy_bytes);
    return dst;
}

// 校验各分片 rank、dtype 和尾随维度后，沿首维连续拼接。
runtime::NDArray ConcatFirstDim(const std::vector<runtime::NDArray>& arrays) {
    if (arrays.empty()) {
        return runtime::NDArray();
    }
    const runtime::NDArray& first = arrays[0];
    if (!first.defined()) {
        return runtime::NDArray();
    }
    RequireCPU(first);
    if (first->shape_storage.empty()) {
        throw std::runtime_error("Gather expects rank >= 1 tensor");
    }

    int64_t total_dim0 = 0;
    for (const auto& arr : arrays) {
        if (!arr.defined()) {
            throw std::runtime_error("Gather source contains undefined shard");
        }
        RequireCPU(arr);
        if (arr->shape_storage.size() != first->shape_storage.size() ||
            !SameDType(arr.dtype(), first.dtype())) {
            throw std::runtime_error("Gather requires equal rank and dtype");
        }
        for (size_t i = 1; i < first->shape_storage.size(); ++i) {
            if (arr->shape_storage[i] != first->shape_storage[i]) {
                throw std::runtime_error("Gather requires equal trailing dimensions");
            }
        }
        if (arr->shape_storage[0] >
            std::numeric_limits<int64_t>::max() - total_dim0) {
            throw std::overflow_error("Gather leading dimension overflow");
        }
        total_dim0 += arr->shape_storage[0];
    }

    Array<int64_t> out_shape;
    out_shape.push_back(total_dim0);
    for (size_t i = 1; i < first->shape_storage.size(); ++i) {
        out_shape.push_back(first->shape_storage[i]);
    }
    runtime::NDArray dst = runtime::NDArray::Empty(out_shape, first.dtype(), Device::CPU());
    if (dst.NBytes() == 0) return dst;

    uint8_t* dst_ptr = static_cast<uint8_t*>(RawData(dst));
    size_t offset = 0;
    for (const auto& arr : arrays) {
        size_t bytes = arr.NBytes();
        if (offset > dst.NBytes() || bytes > dst.NBytes() - offset) {
            throw std::overflow_error("Gather byte range overflow");
        }
        std::memcpy(dst_ptr + offset, RawData(arr), bytes);
        offset += bytes;
    }
    return dst;
}

// 使用进程内 CPU NDArray 实现 Disco 的复制与基础 collective 契约。
class CpuCCLBackend final : public CCLBackend {
public:
    // 在两个 worker 槽位之间复制张量，并保持 Storage 相互独立。
    void Copy(const DiscoSession& session, const DRef& src, const DRef& dst, int src_worker,
              int dst_worker) override {
        runtime::NDArray src_array = session.Get(src_worker, src);
        if (!src_array.defined()) {
            throw std::runtime_error("kxc.disco.copy source is undefined");
        }
        session.Set(dst_worker, dst, CloneArray(src_array));
    }

    // 汇总所有 worker 的张量并把 sum 结果复制回每个 worker。
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

    // 从全局或各 group 的根 worker 广播独立张量副本。
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

    // 将根张量沿首维等分到全局 worker 或各 group 内 worker。
    void ScatterFromWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                            bool in_group) override {
        runtime::NDArray root = session.Get(0, src);
        if (!root.defined()) {
            throw std::runtime_error("Scatter source on worker0 is undefined");
        }

        int receivers = in_group ? (session.num_workers() / session.num_groups())
                                 : session.num_workers();
        if (root->shape_storage.empty() || root->shape_storage[0] % receivers != 0) {
            throw std::runtime_error(
                "Scatter expects first dimension divisible by receiver count");
        }
        int64_t shard = root->shape_storage[0] / receivers;

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

    // 收集 worker 分片并沿首维拼接到全局或各 group 的根 worker。
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

    // 把 worker0 的源张量发送为指定 worker 的独立副本。
    void SendToWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                      int receiver_worker) override {
        runtime::NDArray value = session.Get(0, src);
        if (!value.defined()) {
            throw std::runtime_error("SendToWorker expects source on worker0");
        }
        session.Set(receiver_worker, dst, CloneArray(value));
    }

    // 从指定 worker 接收张量并写入 worker0。
    void RecvFromWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                        int sender_worker) override {
        runtime::NDArray value = session.Get(sender_worker, src);
        if (!value.defined()) {
            throw std::runtime_error("RecvFromWorker source is undefined");
        }
        session.Set(0, dst, CloneArray(value));
    }

    // 将同步请求转发给会话的真实 worker 队列。
    void SyncWorker(const DiscoSession& session, int worker_id) override {
        session.SyncWorker(worker_id);
    }
};

// 返回供 C++ 和 PackedFunc 注册入口共享的进程级 CPU collective 后端。
std::shared_ptr<CCLBackend> GlobalCpuBackend() {
    static std::shared_ptr<CCLBackend> backend = std::make_shared<CpuCCLBackend>();
    return backend;
}

}  // namespace

// 向执行计划层暴露 CPU collective 后端工厂。
std::shared_ptr<CCLBackend> CreateCpuCCLBackend() {
    return GlobalCpuBackend();
}

// PackedFunc 入口仅负责参数解包，所有数据与设备校验保留在强类型后端中。
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

namespace kxc::builtin_anchor {
void DistributedCclCpu() {}
}  // namespace kxc::builtin_anchor
