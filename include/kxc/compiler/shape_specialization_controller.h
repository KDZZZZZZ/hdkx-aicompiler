/*! \file include/kxc/compiler/shape_specialization_controller.h
 * \brief Explicit exact-specializing compiler control plane.
 *
 * Acquire is lookup-only. CompileAndPublish is the sole operation with compile
 * intent; neither operation executes a graph or allocates runtime buffers.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/compiler/shape_control.h"

namespace kxc::api::experimental::shape_control::v1 {

namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;

class ShapeSpecializationController final {
public:
    // config is the sole compile target; max_profiles is a fixed hard limit.
    ShapeSpecializationController(
        restricted::PreparedRestrictedSymbolicTemplate prepared,
        CompileConfig config, std::size_t max_profiles);

    [[nodiscard]] std::optional<PublishedExactVariant> Acquire(
        const std::vector<std::vector<int64_t>>& input_shapes) const;

    [[nodiscard]] PublishedExactVariant CompileAndPublish(
        const std::vector<std::vector<int64_t>>& input_shapes);

private:
    std::size_t max_profiles_;
    CompileConfig config_;
    restricted::PreparedRestrictedSymbolicTemplate prepared_;
    ExactProfileRouteTable routes_;
    // ponytail: serialize profile compilation until measured demand justifies
    // per-profile coordination.
    std::mutex compile_mutex_;
};

}  // namespace kxc::api::experimental::shape_control::v1
