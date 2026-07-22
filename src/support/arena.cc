/*! \file src/support/arena.cc
 * \brief Implements Arena allocation and thread-local scope switching.
 */

#include "kxc/support/arena.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "internal/arena_state.h"

namespace kxc {

namespace detail {

ArenaState::ArenaState(size_t size_bytes) : memory_block_(size_bytes) {
    if (size_bytes == 0) {
        throw std::invalid_argument("Arena size must be greater than zero");
    }
}

void* ArenaState::Allocate(size_t size, size_t alignment) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return nullptr;

    const size_t capacity = memory_block_.size();
    if (offset_ > capacity) return nullptr;

    const uintptr_t current =
        reinterpret_cast<uintptr_t>(memory_block_.data()) + offset_;
    if (current > std::numeric_limits<uintptr_t>::max() - (alignment - 1)) {
        return nullptr;
    }

    const uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
    const size_t padding = static_cast<size_t>(aligned - current);
    const size_t remaining = capacity - offset_;
    if (padding > remaining || size > remaining - padding) return nullptr;

    offset_ += padding + size;
    return reinterpret_cast<void*>(aligned);
}

void ArenaState::Retain() noexcept {
    references_.fetch_add(1, std::memory_order_relaxed);
}

void ArenaState::Release() noexcept {
    if (references_.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
}

}  // namespace detail

thread_local Arena* current_arena = nullptr;

Arena::Arena(size_t size_bytes) : state_(new detail::ArenaState(size_bytes)) {}

Arena::~Arena() {
    state_->Release();
}

void* Arena::Allocate(size_t size, size_t alignment) noexcept {
    return state_->Allocate(size, alignment);
}

detail::ArenaState* Arena::RetainState() noexcept {
    state_->Retain();
    return state_;
}

ArenaScope::ArenaScope(Arena& arena) noexcept : previous_(current_arena) {
    current_arena = &arena;
}

ArenaScope::~ArenaScope() {
    current_arena = previous_;
}

}  // namespace kxc
