/*! \file src/support/internal/arena_state.h
 * \brief Private shared lifetime state for Arena allocations.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace kxc::detail {

class ArenaState {
public:
    explicit ArenaState(size_t size_bytes);

    ArenaState(const ArenaState&) = delete;
    ArenaState& operator=(const ArenaState&) = delete;

    void* Allocate(size_t size, size_t alignment) noexcept;
    void Retain() noexcept;
    void Release() noexcept;

private:
    ~ArenaState() = default;

    std::atomic<size_t> references_{1};
    std::vector<std::byte> memory_block_;
    size_t offset_{0};
};

}  // namespace kxc::detail
