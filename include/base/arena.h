/*! \file include/base/arena.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kxc {

namespace detail {

/*!
 * \brief Arena 内存块的共享生命周期控制器。
 *
 * 引用由 Arena owner 和每个尚未析构的 Arena 对象共同持有。因此 Arena
 * owner 离开作用域后，逃逸的 ObjectRef 仍能保证底层 block 有效。
 */
class ArenaState {
public:
    explicit ArenaState(size_t size_bytes) : memory_block_(size_bytes) {
        if (size_bytes == 0) {
            throw std::invalid_argument("Arena size must be greater than zero");
        }
    }

    ArenaState(const ArenaState&) = delete;
    ArenaState& operator=(const ArenaState&) = delete;

    // 单线程 bump allocation；对象可以在其他线程析构，但同一 Arena 不并发分配。
    void* Allocate(size_t size, size_t alignment) noexcept {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            return nullptr;
        }

        const size_t capacity = memory_block_.size();
        if (offset_ > capacity) {
            return nullptr;
        }

        const uintptr_t current =
            reinterpret_cast<uintptr_t>(memory_block_.data()) + offset_;
        if (current > std::numeric_limits<uintptr_t>::max() - (alignment - 1)) {
            return nullptr;
        }

        const uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
        const size_t padding = static_cast<size_t>(aligned - current);
        const size_t remaining = capacity - offset_;
        if (padding > remaining || size > remaining - padding) {
            return nullptr;
        }

        offset_ += padding + size;
        return reinterpret_cast<void*>(aligned);
    }

    void Retain() noexcept {
        references_.fetch_add(1, std::memory_order_relaxed);
    }

    void Release() noexcept {
        if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

private:
    ~ArenaState() = default;

    // 初始的 1 属于 Arena owner，成功的 Arena 对象分配各增加 1。
    std::atomic<size_t> references_{1};
    std::vector<std::byte> memory_block_;
    size_t offset_{0};
};

}  // namespace detail

class Arena {
public:
    explicit Arena(size_t size_bytes = 4 * 1024 * 1024)
        : state_(new detail::ArenaState(size_bytes)) {}

    // 这里只释放 owner 引用；逃逸对象仍持有 state，不会提前释放 block。
    ~Arena() { state_->Release(); }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* Allocate(size_t size, size_t alignment) noexcept {
        return state_->Allocate(size, alignment);
    }

    // Object 分配成功后调用，使该对象独立保活 ArenaState。
    detail::ArenaState* RetainState() noexcept {
        state_->Retain();
        return state_;
    }

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
    explicit ArenaScope(Arena& arena) noexcept : previous_(current_arena) {
        current_arena = &arena;
    }

    ~ArenaScope() { current_arena = previous_; }

    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;

private:
    Arena* previous_;
};

}  // namespace kxc
