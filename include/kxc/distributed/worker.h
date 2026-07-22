/*! \file include/kxc/distributed/worker.h
 * \brief 定义 Disco 分布式会话、DRef、执行器和通信后端接口。
 */

#pragma once

namespace kxc {
namespace disco {

int CurrentWorkerId();
void SetCurrentWorkerId(int worker_id);

}  // namespace disco
}  // namespace kxc
