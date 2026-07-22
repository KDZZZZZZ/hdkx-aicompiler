/*! \file include/kxc/support/arena.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <cstddef>

namespace kxc {

namespace detail {
class ArenaState;

}  // namespace detail

class Arena {
public:
    explicit Arena(size_t size_bytes = 4 * 1024 * 1024);
    ~Arena();

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* Allocate(size_t size, size_t alignment) noexcept;

    // Object 分配成功后调用，使该对象独立保活 ArenaState。
    detail::ArenaState* RetainState() noexcept;

private:
    detail::ArenaState* state_;
};

// TLS 只决定 new 从哪里分配；delete 根据对象自己的分配头判断来源。
extern thread_local Arena* current_arena;

/*!
 * \brief 在当前线程临时启用 Arena，并在退出时恢复上一个 Arena。
 *
 * 支持嵌套 scope；ArenaScope 本身必须先于传入的 Arena 析构。
 */
class ArenaScope {
public:
    explicit ArenaScope(Arena& arena) noexcept;
    ~ArenaScope();

    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;

private:
    Arena* previous_;
};

}  // namespace kxc
