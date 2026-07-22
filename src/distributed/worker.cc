/*! \file src/distributed/worker.cc
 * \brief 实现 Disco 线程会话、执行计划解释器和 CPU/NCCL 通信后端。
 */

#include "kxc/distributed/worker.h"

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"

namespace kxc {
namespace disco {

namespace {
thread_local int tls_worker_id = 0;
}  // namespace

int CurrentWorkerId() {
    return tls_worker_id;
}

void SetCurrentWorkerId(int worker_id) {
    tls_worker_id = worker_id;
}

KXC_REGISTER_GLOBAL("kxc.disco.worker_id")
    .set_body(ToPackedFunc([]() -> int { return CurrentWorkerId(); }));

}  // namespace disco
}  // namespace kxc

namespace kxc::builtin_anchor {
void DistributedWorker() {}
}  // namespace kxc::builtin_anchor
