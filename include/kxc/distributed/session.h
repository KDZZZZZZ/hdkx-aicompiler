/*! \file include/kxc/distributed/session.h
 * \brief 定义 Disco 分布式会话、DRef、执行器和通信后端接口。
 */

#pragma once

#include <string>

#include "kxc/support/container.h"
#include "kxc/distributed/dref.h"
#include "kxc/runtime/ndarray.h"

namespace kxc {
namespace disco {

class DiscoSessionNode : public Object {
public:
    virtual ~DiscoSessionNode() = default;

    virtual int num_workers() const = 0;
    virtual int num_groups() const = 0;
    virtual int AllocateRegister() = 0;
    virtual runtime::NDArray GetRegister(int worker_id, int reg_id) const = 0;
    virtual void SetRegister(int worker_id, int reg_id, runtime::NDArray value) = 0;
    virtual void SyncWorker(int worker_id) = 0;
    virtual void Shutdown() = 0;

    KXC_OBJECT_DECLARE
};

class DiscoSession : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit DiscoSession(const ObjectRef& ref) : ObjectRef(ref) {}

    const DiscoSessionNode* operator->() const;
    DRef NewDRef() const;
    DRef Empty(Array<int64_t> shape, std::string dtype, bool worker0_only = false,
               bool in_group = true) const;
    runtime::NDArray Get(int worker_id, const DRef& ref) const;
    void Set(int worker_id, const DRef& ref, runtime::NDArray value) const;
    int num_workers() const;
    int num_groups() const;
    void SyncWorker(int worker_id) const;
    void Shutdown() const;

    static DiscoSession ThreadedSession(int num_workers, int num_groups = 1);
};

}  // namespace disco
}  // namespace kxc
