#include "base/disco/worker.h"

#include "base/packedfunc.h"
#include "base/registry.h"

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

