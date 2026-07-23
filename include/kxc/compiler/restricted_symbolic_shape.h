/*! \file include/kxc/compiler/restricted_symbolic_shape.h
 * \brief Default-off W3 restricted symbolic Shape control plane.
 *
 * This experimental-v1 surface mints validated shape dispatch decisions.  It
 * deliberately does not allocate dynamic buffers, compile/cache artifacts,
 * execute guarded plans, or lower control flow. RuntimeSession remains the
 * static-exact execution boundary. Enabling this adapter also requires the
 * default-off exact representative gate.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/shape_exact.h"
#include "kxc/shape/guarded_specialization.h"

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {

inline constexpr uint32_t kRestrictedSymbolicShapeVersion = 1;

struct InputAxisSymbol final {
    size_t parameter_index{0};
    size_t axis{0};
    std::string symbol;
    int64_t lower{0};
    int64_t upper{0};
    int64_t divisible_by{1};
};

enum class DispatchKind { kExact, kBucket, kPolymorphic };

struct RestrictedDispatchDecision final {
    DispatchKind kind{DispatchKind::kExact};
    shape::experimental::v1::ExactOracle exact_oracle;
    std::vector<shape::experimental::v1::UnitSpecializationRequest> exact_requests;
    std::vector<shape::experimental::v1::GuardedUnitSpecializationRequest> guarded_requests;
    std::string canonical_identity;
};

class PreparedRestrictedSymbolicTemplate final {
public:
    PreparedRestrictedSymbolicTemplate();
    ~PreparedRestrictedSymbolicTemplate();
    PreparedRestrictedSymbolicTemplate(const PreparedRestrictedSymbolicTemplate&);
    PreparedRestrictedSymbolicTemplate& operator=(const PreparedRestrictedSymbolicTemplate&);
    PreparedRestrictedSymbolicTemplate(PreparedRestrictedSymbolicTemplate&&) noexcept;
    PreparedRestrictedSymbolicTemplate& operator=(PreparedRestrictedSymbolicTemplate&&) noexcept;

    const shape::experimental::v1::GraphTemplate& graph_template() const;
    const std::vector<InputAxisSymbol>& input_axis_symbols() const;
    // The frozen representative is retained solely for static-exact fallback;
    // it is never a dynamic RuntimeSession plan.
    const kxc::api::experimental::shape_exact::v1::PreparedGraphTemplate& representative() const;

private:
    struct Impl;
    explicit PreparedRestrictedSymbolicTemplate(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

class RestrictedSymbolicShapeAdapter final {
public:
    static bool IsEnabled() noexcept;

    // Deep-freezes the representative through the exact adapter, then exposes
    // only fixed-rank relu/sqrt and equal-shape add/mul formulas. Constants,
    // broadcasts, aliases, control flow, and unsupported Relay structure fail closed.
    static PreparedRestrictedSymbolicTemplate Prepare(
        Function representative, CompileConfig config,
        std::vector<InputAxisSymbol> input_axis_symbols);

    // These mint complete requests before any lowering/cache authority is
    // touched. A guard miss, stale template, or invalid tail proof throws.
    static RestrictedDispatchDecision MintExact(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const shape::experimental::v1::BindingSet& bindings);
    static RestrictedDispatchDecision MintBucket(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const shape::experimental::v1::BindingSet& bindings,
        const shape::experimental::v1::BucketPolicy& policy);
    static RestrictedDispatchDecision MintPolymorphic(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const shape::experimental::v1::BindingSet& bindings,
        const shape::experimental::v1::PolymorphicPolicy& policy);

    // Returns precisely the units whose complete boundary artifact request
    // changed. This is a comparison of canonical requests, never a cache.
    static std::vector<size_t> ChangedUnitIndices(
        const RestrictedDispatchDecision& previous,
        const RestrictedDispatchDecision& next);
};

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
