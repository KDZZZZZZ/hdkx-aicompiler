/*! \file include/base/disco/ccl_backend.h
 * \brief 定义 Disco 分布式会话、DRef、执行器和通信后端接口。
 */

#pragma once

#include <memory>
#include <string>

#include "base/disco/dref.h"
#include "base/disco/session.h"

namespace kxc {
namespace disco {

class CCLBackend {
public:
    virtual ~CCLBackend() = default;

    virtual void Copy(const DiscoSession& session, const DRef& src, const DRef& dst, int src_worker,
                      int dst_worker) = 0;
    virtual void AllReduce(const DiscoSession& session, const DRef& src, const DRef& dst,
                           const std::string& reduce_kind, bool in_group) = 0;
    virtual void BroadcastFromWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                                      bool in_group) = 0;
    virtual void ScatterFromWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                                    bool in_group) = 0;
    virtual void GatherToWorker0(const DiscoSession& session, const DRef& src, const DRef& dst,
                                 bool in_group) = 0;
    virtual void SendToWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                              int receiver_worker) = 0;
    virtual void RecvFromWorker(const DiscoSession& session, const DRef& src, const DRef& dst,
                                int sender_worker) = 0;
    virtual void SyncWorker(const DiscoSession& session, int worker_id) = 0;
};

std::shared_ptr<CCLBackend> CreateCpuCCLBackend();

}  // namespace disco
}  // namespace kxc

