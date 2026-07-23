/*! \file src/compiler/internal/prepared_static_graph.h
 * \brief Immutable graph work shared by static compilation and shape exact. */
#pragma once

#include <cstddef>

#include "compilation_unit.h"
#include "kxc/target/target.h"

namespace kxc::api::internal {

struct PreparedStaticGraph final {
    PartitionedGraph partitioned;
    Device device;
    Target target;
    String pipeline_fingerprint;
    size_t capability_boundary_checks{0};
    size_t value_graph_builds{0};
    size_t partitions{0};
};

}  // namespace kxc::api::internal
