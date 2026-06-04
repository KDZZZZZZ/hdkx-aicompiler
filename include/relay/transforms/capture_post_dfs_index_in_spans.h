/*! \file include/relay/transforms/capture_post_dfs_index_in_spans.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "relay/relay.h"

namespace kxc {
namespace relay {

Function CapturePostDfsIndexInSpansPass(const Function& func);

}  // namespace relay
}  // namespace kxc

