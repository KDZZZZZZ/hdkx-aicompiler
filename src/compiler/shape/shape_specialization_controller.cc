#include "kxc/compiler/shape_specialization_controller.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "../internal/primitive_cache.h"

namespace kxc::api::experimental::shape_control::v1 {
namespace {

std::size_t RequireMaxProfiles(std::size_t max_profiles) {
    if (max_profiles == 0) {
        throw std::invalid_argument(
            "shape specialization controller requires max_profiles > 0");
    }
    return max_profiles;
}

std::string TargetFingerprint(const CompileConfig& config) {
    config.Validate();
    return internal::BuildTargetCapabilityFingerprint(config->target);
}

restricted::RestrictedDispatchDecision Decide(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    const std::vector<std::vector<int64_t>>& input_shapes) {
    return restricted::RestrictedSymbolicShapeAdapter::MintExact(
        prepared,
        restricted::RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
            prepared, input_shapes));
}

}  // namespace

ShapeSpecializationController::ShapeSpecializationController(
    restricted::PreparedRestrictedSymbolicTemplate prepared,
    CompileConfig config, std::size_t max_profiles)
    : max_profiles_(RequireMaxProfiles(max_profiles)),
      config_(std::move(config)),
      prepared_(std::move(prepared)),
      routes_(prepared_.graph_template(), TargetFingerprint(config_)) {}

std::optional<PublishedExactVariant>
ShapeSpecializationController::Acquire(
    const std::vector<std::vector<int64_t>>& input_shapes) const {
    const restricted::RestrictedDispatchDecision decision =
        Decide(prepared_, input_shapes);
    return routes_.TryLookup(decision.exact_oracle());
}

PublishedExactVariant ShapeSpecializationController::CompileAndPublish(
    const std::vector<std::vector<int64_t>>& input_shapes) {
    const restricted::RestrictedDispatchDecision decision =
        Decide(prepared_, input_shapes);
    if (auto published = routes_.TryLookup(decision.exact_oracle())) {
        return std::move(*published);
    }

    std::lock_guard<std::mutex> lock(compile_mutex_);
    if (auto published = routes_.TryLookup(decision.exact_oracle())) {
        return std::move(*published);
    }
    if (routes_.size() >= max_profiles_) {
        throw std::length_error(
            "shape specialization controller max_profiles reached");
    }

    Function materialized =
        restricted::RestrictedSymbolicShapeAdapter::MaterializeExactFunction(
            prepared_, decision);
    CompiledGraph compiled =
        Compiler::Compile(std::move(materialized), config_);
    restricted::RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
        prepared_, decision, compiled);
    const PlanAbiFingerprint plan_abi = BuildPlanAbiFingerprint(compiled);
    routes_.Publish(decision.exact_oracle(), std::move(compiled), plan_abi);
    return routes_.Lookup(decision.exact_oracle());
}

}  // namespace kxc::api::experimental::shape_control::v1
