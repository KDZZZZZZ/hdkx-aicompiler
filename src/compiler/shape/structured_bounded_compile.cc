/*! \file src/compiler/shape/structured_bounded_compile.cc
 * \brief Region-aware bounded admission for structured control graphs (PR3).
 *
 * Pipeline (per the M10 C3 PR3 design finalization):
 *   1. Resolve restricted shape values on the representative. The resolver's
 *      post-order unit operation list and per-unit output dimension proofs are
 *      the single symbolic authority, exactly as in the linear bounded path.
 *   2. Lower the resolver's rewritten program to a ControlPlan (dense units and
 *      regions). Its units must align positionally with the resolver's units;
 *      misalignment is rejected, never silently accepted.
 *   3. Build a ShapeProgram over all values, a GraphTemplate whose units are the
 *      control units (with topology-produced values marked synthesized), and a
 *      DynamicUnitShapeContract per unit.
 *   4. Build graph-input guards from declared symbols and a StructuredSchedule
 *      referencing call indices.
 */

#include "../internal/structured_bounded_compile.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../control_flow/internal_lowering.h"
#include "../internal/relay_program.h"
#include "shape_value_resolver.h"
#include "kxc/relay/transforms/infer_type.h"

namespace kxc::api::internal {
namespace {

namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;
using specialization::DimExpr;
namespace shape = kxc::shape::experimental::v1;
namespace shape_resolution = restricted::shape_resolution;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("PrepareStructuredBoundedCompile: " + message);
}

std::string ValueName(ValueId id) { return "value." + std::to_string(id); }

shape::NamedTensorContract MakeContract(ValueId id,
                                        const std::vector<DimExpr>& dims) {
    return shape::NamedTensorContract{
        ValueName(id),
        shape::TensorShapeContract(shape::LogicalShape(dims),
                                   shape::PhysicalCapacity(dims),
                                   shape::ValidExtent(dims))};
}

std::vector<DimExpr> StaticDims(const TensorTypeNode& type,
                                const char* context) {
    std::vector<DimExpr> dims;
    for (int64_t extent : type.shape) {
        if (extent < 0) {
            Reject(std::string(context) +
                   " has a wildcard extent with no symbolic proof");
        }
        dims.push_back(DimExpr::Const(extent));
    }
    return dims;
}

}  // namespace

StructuredBoundedCompilation::StructuredBoundedCompilation(
    PartitionedGraph partitioned_graph,
    std::vector<DynamicUnitShapeContract> unit_shape_contracts,
    std::vector<runtime::GraphInputAxisGuard> graph_input_guards,
    runtime::StructuredSchedule schedule, GraphSemanticKey graph_semantic_key,
    CompileConfig config)
    : partitioned_graph_(std::move(partitioned_graph)),
      unit_shape_contracts_(std::move(unit_shape_contracts)),
      graph_input_guards_(std::move(graph_input_guards)),
      schedule_(std::move(schedule)),
      graph_semantic_key_(std::move(graph_semantic_key)),
      config_(std::move(config)) {}

const PartitionedGraph& StructuredBoundedCompilation::partitioned_graph()
    const noexcept { return partitioned_graph_; }
const std::vector<DynamicUnitShapeContract>&
StructuredBoundedCompilation::unit_shape_contracts() const noexcept {
    return unit_shape_contracts_;
}
const std::vector<runtime::GraphInputAxisGuard>&
StructuredBoundedCompilation::graph_input_guards() const noexcept {
    return graph_input_guards_;
}
const runtime::StructuredSchedule& StructuredBoundedCompilation::schedule()
    const noexcept { return schedule_; }
const GraphSemanticKey& StructuredBoundedCompilation::graph_semantic_key()
    const noexcept { return graph_semantic_key_; }
const CompileConfig& StructuredBoundedCompilation::config() const noexcept {
    return config_;
}

StructuredBoundedCompilation PrepareStructuredBoundedCompile(
    Function representative, CompileConfig config,
    const std::vector<restricted::InputAxisSymbol>& input_axis_symbols) {
    config.Validate();
    if (input_axis_symbols.empty()) {
        Reject("at least one explicit input-axis symbol is required");
    }
    if (!(config->target->kind == "llvm" && config->target->device_type == kCPU &&
          config->target->device_id == 0)) {
        Reject("structured bounded compilation requires CPU:0/LLVM");
    }

    // 1. Type the representative and confirm residual control. The resolver
    //    walks a non-ANF typed graph (ANF introduces Let-bound variables the
    //    resolver does not model); the control lowerer ANF-normalizes the
    //    resolver's rewritten output itself, matching the linear restricted path.
    PreparedRelayProgram prepared = PrepareRelayProgram(
        representative, config, ControlFlowPolicy::NativeExact());
    if (!prepared.residual_profile().requires_control_topology()) {
        Reject("the representative has no residual control topology");
    }

    // 2. Symbolic shape resolution is the single authority for unit dims.
    const Function snapshot = relay::InferTypePass(representative);
    const shape_resolution::Resolution resolution =
        shape_resolution::ResolveShapeValues(snapshot, input_axis_symbols);
    const std::vector<std::string>& operations = resolution.registry_operations;
    if (operations.empty()) {
        Reject("the representative has no resolved compute units");
    }

    // 3. Lower the resolver's rewritten program to a ControlPlan. Its units must
    //    align with the resolver's post-order unit sequence; fail closed.
    ControlPlanLowering lowered =
        LowerRelayToControlPlanWithSidecar(resolution.rewritten);
    const ControlPlan& plan = lowered.plan;
    const std::vector<PrimitiveUnit>& units = lowered.primitive_units;
    if (units.size() != operations.size() ||
        resolution.unit_output_dimensions.size() != units.size()) {
        Reject("control units do not match the resolved operation sequence");
    }
    for (size_t index = 0; index < units.size(); ++index) {
        const std::string ops_name =
            operations[index] == "nn_relu" ? "relu" : operations[index];
        if (ops_name != std::string(units[index].call.spec.name)) {
            Reject("resolver and control lowering disagree on unit order");
        }
        if (units[index].output_value_ids.size() != 1) {
            Reject("structured bounded v1 requires one output per unit");
        }
    }

    // 3. Establish symbolic dims for every control-plan value.
    std::map<ValueId, std::vector<DimExpr>> dims_by_value;
    std::set<std::string> declared_symbols;
    std::vector<shape::Constraint> constraints;
    {
        std::map<std::string, std::pair<int64_t, int64_t>> range;
        std::map<std::string, int64_t> divisor;
        for (const auto& symbol : input_axis_symbols) {
            declared_symbols.insert(symbol.symbol);
            const auto inserted = range.emplace(
                symbol.symbol, std::make_pair(symbol.lower, symbol.upper));
            if (!inserted.second &&
                (inserted.first->second.first != symbol.lower ||
                 inserted.first->second.second != symbol.upper ||
                 divisor.at(symbol.symbol) != symbol.divisible_by)) {
                Reject("a repeated symbol has inconsistent bounds");
            }
            divisor.emplace(symbol.symbol, symbol.divisible_by);
        }
        for (const auto& item : range) {
            constraints.push_back(shape::Constraint::Range(
                DimExpr::Symbol(item.first), item.second.first,
                item.second.second));
            constraints.push_back(shape::Constraint::DivisibleBy(
                DimExpr::Symbol(item.first), divisor.at(item.first)));
        }
    }
    // Parameters.
    if (plan.graph_inputs.size() != resolution.rewritten->params.size()) {
        Reject("control plan inputs differ from the representative parameters");
    }
    std::map<size_t, std::map<size_t, std::string>> axis_symbol_by_param;
    for (const auto& symbol : input_axis_symbols) {
        axis_symbol_by_param[symbol.parameter_index][symbol.axis] = symbol.symbol;
    }
    for (size_t parameter = 0; parameter < resolution.rewritten->params.size();
         ++parameter) {
        const auto* type = resolution.rewritten->params[parameter]
                               ->type_annotation.As<TensorTypeNode>();
        if (!type) Reject("representative parameter lacks a TensorType");
        std::vector<DimExpr> dims;
        const auto by_axis = axis_symbol_by_param.find(parameter);
        for (size_t axis = 0; axis < type->shape.size(); ++axis) {
            if (by_axis != axis_symbol_by_param.end() &&
                by_axis->second.count(axis) != 0) {
                dims.push_back(DimExpr::Symbol(by_axis->second.at(axis)));
            } else {
                dims.push_back(DimExpr::Const(type->shape[axis]));
            }
        }
        dims_by_value[plan.graph_inputs[parameter]] = std::move(dims);
    }
    // Constants.
    for (ValueId id : plan.constant_values) {
        const auto* type =
            plan.values.at(static_cast<size_t>(id)).checked_type.As<TensorTypeNode>();
        if (!type) Reject("a constant lacks a TensorType");
        dims_by_value[id] = StaticDims(*type, "constant");
    }
    // Unit outputs (resolver-proven dims, positional).
    for (size_t index = 0; index < units.size(); ++index) {
        dims_by_value[units[index].output_value_ids[0]] =
            resolution.unit_output_dimensions[index];
    }
    // Phi and loop-carried topology values.
    for (const auto& region : plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.kind == ControlTaskKind::kBranch) {
                for (const auto& phi : task.branch.phis) {
                    const auto then_it = dims_by_value.find(phi.then_value);
                    const auto else_it = dims_by_value.find(phi.else_value);
                    if (then_it == dims_by_value.end() ||
                        else_it == dims_by_value.end() ||
                        then_it->second != else_it->second) {
                        Reject("If arms must agree on symbolic dims");
                    }
                    dims_by_value[phi.result] = then_it->second;
                }
            } else if (task.kind == ControlTaskKind::kLoop) {
                for (const auto& carried : task.loop.carried) {
                    const auto initial = dims_by_value.find(carried.initial);
                    const auto backedge = dims_by_value.find(carried.backedge);
                    if (initial == dims_by_value.end() ||
                        backedge == dims_by_value.end() ||
                        initial->second != backedge->second) {
                        Reject("loop carried shape must be invariant across the body");
                    }
                    dims_by_value[carried.body_argument] = initial->second;
                    dims_by_value[carried.result] = initial->second;
                }
            }
        }
    }
    for (const auto& value : plan.values) {
        if (dims_by_value.find(value.id) == dims_by_value.end()) {
            Reject("value " + ValueName(value.id) +
                   " has no symbolic dimension proof");
        }
    }

    // 4. ShapeProgram: inputs are graph inputs; every other value is an output
    //    slot (matching the linear bounded template convention).
    std::vector<shape::NamedTensorContract> program_inputs;
    std::vector<shape::NamedTensorContract> program_outputs;
    std::unordered_set<ValueId> graph_inputs(plan.graph_inputs.begin(),
                                             plan.graph_inputs.end());
    for (ValueId id : plan.graph_inputs) {
        program_inputs.push_back(MakeContract(id, dims_by_value.at(id)));
    }
    std::vector<std::string> synthesized;
    for (const auto& value : plan.values) {
        if (graph_inputs.count(value.id)) continue;
        program_outputs.push_back(MakeContract(value.id, dims_by_value.at(value.id)));
        if (value.origin == LogicalValueOrigin::kPhi ||
            value.origin == LogicalValueOrigin::kLoopCarried) {
            synthesized.push_back(ValueName(value.id));
        }
    }
    shape::ShapeProgram program(
        std::vector<std::string>(declared_symbols.begin(), declared_symbols.end()),
        std::move(program_inputs), std::move(program_outputs),
        std::move(constraints));

    // 5. GraphTemplate units and per-unit contracts.
    std::vector<specialization::UnitSkeleton> skeletons;
    skeletons.reserve(units.size());
    for (const auto& unit : units) {
        std::vector<std::string> unit_inputs;
        for (int64_t id : unit.boundary_input_value_ids) {
            unit_inputs.push_back(ValueName(id));
        }
        std::vector<std::string> unit_outputs;
        for (int64_t id : unit.output_value_ids) {
            unit_outputs.push_back(ValueName(id));
        }
        skeletons.push_back(specialization::UnitSkeleton{
            specialization::GraphLocalCallLocator(ValueName(unit.output_value_ids[0])),
            unit.semantic_key, std::move(unit_inputs), std::move(unit_outputs)});
    }
    specialization::GraphTemplate graph_template(
        Compiler::BuildGraphSemanticKey(resolution.rewritten), std::move(program),
        std::move(skeletons), std::move(synthesized));

    std::vector<DynamicUnitShapeContract> contracts =
        BuildDynamicUnitShapeContracts(
            graph_template, operations,
            DecodeValueExpressionOverrides(resolution.unit_value_expressions));

    // 6. PartitionedGraph from the control plan (dense units, one call each).
    //    Boundary values use the logical-boundary representation: a non-constant
    //    symbolic axis becomes -1, exactly as BuildBoundedValueGraph produces,
    //    so bounded unit lowering and the dynamic plan agree.
    const auto boundary_shape = [](const std::vector<DimExpr>& dims) {
        Array<int64_t> shape;
        for (const DimExpr& dim : dims) {
            if (dim.kind() == DimExpr::Kind::kConst) {
                shape.push_back(dim.Evaluate(specialization::BindingSet()));
            } else {
                shape.push_back(-1);
            }
        }
        return shape;
    };
    PartitionedGraph partitioned;
    partitioned.units = units;
    // Retarget each unit's resolved output leaf types to the logical-boundary
    // representation so ValidatePrimitiveUnit agrees with the bounded values.
    for (auto& unit : partitioned.units) {
        for (size_t index = 0; index < unit.output_value_ids.size(); ++index) {
            const auto* type = unit.call.output_leaf_types[index].As<TensorTypeNode>();
            if (!type) Reject("a unit output leaf is not a TensorType");
            unit.call.output_leaf_types[index] = TensorType(
                boundary_shape(dims_by_value.at(unit.output_value_ids[index])),
                type->dtype);
        }
    }
    partitioned.calls = Array<runtime::KernelCall>();
    for (const auto& unit : units) {
        partitioned.calls.push_back(runtime::KernelCall(
            unit.symbol, unit.boundary_input_value_ids, unit.output_value_ids));
    }
    partitioned.input_value_ids = Array<int64_t>(plan.graph_inputs);
    partitioned.constant_value_ids = Array<int64_t>(plan.constant_values);
    partitioned.output_value_ids = Array<int64_t>(plan.graph_outputs);
    for (const auto& value : plan.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        if (!type) Reject("a structured bounded plan value has no TensorType");
        LogicalValueContract bounded = value;
        bounded.checked_type = TensorType(boundary_shape(dims_by_value.at(value.id)),
                                          type->dtype);
        partitioned.value_graph.values.push_back(std::move(bounded));
    }
    partitioned.value_graph.function = resolution.rewritten;
    partitioned.value_graph.input_value_ids = partitioned.input_value_ids;
    partitioned.value_graph.constant_value_ids = partitioned.constant_value_ids;
    partitioned.value_graph.output_value_ids = partitioned.output_value_ids;

    // 7. Graph input guards from declared symbols (prerequisite shape).
    std::vector<runtime::GraphInputAxisGuard> guards;
    {
        std::map<std::string, std::pair<int64_t, int64_t>> range;
        std::map<std::string, int64_t> divisor;
        for (const auto& symbol : input_axis_symbols) {
            range.emplace(symbol.symbol,
                          std::make_pair(symbol.lower, symbol.upper));
            divisor.emplace(symbol.symbol, symbol.divisible_by);
        }
        std::map<std::string, size_t> anchor;
        for (size_t input = 0; input < plan.graph_inputs.size(); ++input) {
            const ValueId id = plan.graph_inputs[input];
            const std::vector<DimExpr>& dims = dims_by_value.at(id);
            for (size_t axis = 0; axis < dims.size(); ++axis) {
                if (dims[axis].kind() == DimExpr::Kind::kConst) continue;
                const std::vector<std::string> symbols = dims[axis].Symbols();
                if (symbols.size() != 1 || dims[axis].kind() != DimExpr::Kind::kSymbol) {
                    Reject("a wildcard input axis must be a single direct symbol");
                }
                runtime::GraphInputAxisGuard guard;
                guard.input_index = input;
                guard.axis = axis;
                guard.lower = range.at(symbols[0]).first;
                guard.upper = range.at(symbols[0]).second;
                guard.divisible_by = divisor.at(symbols[0]);
                const runtime::GraphInputAxisReference current{input, axis};
                const auto inserted = anchor.emplace(symbols[0], guards.size());
                if (!inserted.second) {
                    guard.equal_to = runtime::GraphInputAxisReference{
                        guards[inserted.first->second].input_index,
                        guards[inserted.first->second].axis};
                }
                guards.push_back(std::move(guard));
            }
        }
    }
    // Every wildcard input axis must be guarded (validated by the plan).
    {
        std::unordered_set<uint64_t> guarded;
        for (const auto& guard : guards) {
            guarded.insert((static_cast<uint64_t>(guard.input_index) << 32) |
                           guard.axis);
        }
        for (size_t input = 0; input < plan.graph_inputs.size(); ++input) {
            const auto* type = plan.values.at(static_cast<size_t>(
                plan.graph_inputs[input])).checked_type.As<TensorTypeNode>();
            for (size_t axis = 0; axis < type->shape.size(); ++axis) {
                if (type->shape[axis] != -1) continue;
                if (guarded.count((static_cast<uint64_t>(input) << 32) | axis) == 0) {
                    Reject("a bounded graph input axis lacks a guard");
                }
            }
        }
    }

    // 8. StructuredSchedule: map control regions/tasks to the runtime schema.
    std::map<internal::RegionId, int64_t> region_index;
    for (size_t index = 0; index < plan.region_order.size(); ++index) {
        region_index[plan.region_order[index]] = static_cast<int64_t>(index);
    }
    std::map<internal::PrimitiveUnitId, int64_t> unit_index;
    for (size_t index = 0; index < units.size(); ++index) {
        unit_index[units[index].id] = static_cast<int64_t>(index);
    }
    runtime::StructuredSchedule schedule;
    schedule.schema_version = runtime::StructuredSchedule::kSchemaVersion;
    schedule.entry_region = region_index.at(plan.entry_region);
    schedule.region_order.assign(plan.region_order.size(), 0);
    for (size_t index = 0; index < plan.region_order.size(); ++index) {
        schedule.region_order[index] = static_cast<int64_t>(index);
    }
    for (const auto& region : plan.regions) {
        runtime::StructuredRegion target;
        target.id = region_index.at(region.id);
        target.live_ins = region.live_ins;
        target.live_outs = region.live_outs;
        for (const auto& task : region.tasks) {
            runtime::StructuredTask bound;
            bound.id = task.id;
            bound.inputs = task.inputs;
            bound.outputs = task.outputs;
            switch (task.kind) {
                case ControlTaskKind::kKernel: {
                    const auto found = unit_index.find(task.primitive_unit_id);
                    if (found == unit_index.end()) {
                        Reject("a kernel task references an unknown unit");
                    }
                    bound.kind = runtime::StructuredTaskKind::kKernel;
                    bound.call_index = found->second;
                    break;
                }
                case ControlTaskKind::kBranch:
                    bound.kind = runtime::StructuredTaskKind::kBranch;
                    bound.branch.predicate = task.branch.predicate;
                    bound.branch.then_region = region_index.at(task.branch.then_region);
                    bound.branch.else_region = region_index.at(task.branch.else_region);
                    for (const auto& phi : task.branch.phis) {
                        bound.branch.phis.push_back(
                            {phi.result, phi.then_value, phi.else_value});
                    }
                    break;
                case ControlTaskKind::kLoop:
                    bound.kind = runtime::StructuredTaskKind::kLoop;
                    bound.loop.condition_region =
                        region_index.at(task.loop.condition_region);
                    bound.loop.body_region = region_index.at(task.loop.body_region);
                    bound.loop.condition_value = task.loop.condition_value;
                    bound.loop.max_trip_count = task.loop.max_trip_count;
                    for (const auto& carried : task.loop.carried) {
                        bound.loop.carried.push_back(
                            {carried.result, carried.initial, carried.body_argument,
                             carried.backedge});
                    }
                    break;
            }
            target.tasks.push_back(std::move(bound));
        }
        schedule.regions.push_back(std::move(target));
    }

    return StructuredBoundedCompilation(
        std::move(partitioned), std::move(contracts), std::move(guards),
        std::move(schedule), graph_template.key(), config);
}

runtime::ExecutablePlan BuildStructuredBoundedExecutablePlan(
    const StructuredBoundedCompilation& compilation) {
    const PartitionedGraph& graph = compilation.partitioned_graph();
    Array<runtime::ValueSpec> value_specs;
    std::unordered_set<ValueId> graph_inputs(graph.input_value_ids.begin(),
                                             graph.input_value_ids.end());
    std::unordered_set<ValueId> graph_outputs(graph.output_value_ids.begin(),
                                              graph.output_value_ids.end());
    for (const LogicalValueContract& value : graph.value_graph.values) {
        const auto* type = value.checked_type.As<TensorTypeNode>();
        if (!type) Reject("a structured bounded plan value has no TensorType");
        value_specs.push_back(runtime::ValueSpec(
            value.id, value.id, type->shape,
            runtime::DataTypeFromString(type->dtype), value.device,
            graph_inputs.count(value.id) != 0,
            value.origin == LogicalValueOrigin::kConstant,
            graph_outputs.count(value.id) != 0));
    }
    runtime::ExecutablePlan plan(
        value_specs, graph.calls, graph.input_value_ids, graph.constant_value_ids,
        graph.output_value_ids, {}, runtime::ExecutablePlanMode::kDynamicFreshOutputV1,
        compilation.graph_input_guards(), {}, -1, {},
        std::optional<runtime::RequestBatchingContract>{},
        std::optional<runtime::StructuredSchedule>(compilation.schedule()));
    plan.Validate();
    return plan;
}

}  // namespace kxc::api::internal
