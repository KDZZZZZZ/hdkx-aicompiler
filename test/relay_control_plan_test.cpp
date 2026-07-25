/*! \file test/relay_control_plan_test.cpp
 * \brief Standalone Relay-to-ControlPlan preparation checks.
 */

#include <any>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/compiler/internal/value_graph.h"
#include "../src/compiler/control_flow/internal_lowering.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/transforms/infer_type.h"
#include "support/control_plan_reference_executor.h"

namespace {

#define TEST_CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

using kxc::Call;
using kxc::Device;
using kxc::Expr;
using kxc::Function;
using kxc::If;
using kxc::Let;
using kxc::TensorType;
using kxc::Tuple;
using kxc::TupleGetItem;
using kxc::Var;
using kxc::While;
using kxc::api::internal::ControlPlan;
using kxc::api::internal::ControlTask;
using kxc::api::internal::ControlTaskKind;
using kxc::api::internal::test_support::ControlPlanReferenceExecutor;
using kxc::api::internal::test_support::FakeValue;

const TensorType kI64({}, "int64");

kxc::api::internal::ControlPlanLowering LowerRelayProgram(
    Function function) {
    return kxc::api::internal::LowerRelayToControlPlanWithSidecar(
        std::move(function));
}

ControlPlan LowerRelayToControlPlan(Function function) {
    return LowerRelayProgram(std::move(function)).plan;
}

Call Add(const Expr& lhs, const Expr& rhs) {
    return Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

Call Mul(const Expr& lhs, const Expr& rhs) {
    return Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
}

const kxc::relay::Op& TestPredicateOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static bool registered = false;
    if (!registered) {
        OperatorSpec spec;
        spec.name = "test_control_predicate";
        spec.category = "test";
        spec.input_arity.num_inputs = 1;
        spec.output_arity = 1;
        spec.type_relation_key = "FInferType";
        spec.lowering_kind = OperatorLoweringKind::kSingleTE;
        spec.lowering_key = "FRelayToTE";
        Op op = Op::Register(spec);
        auto* node = const_cast<OpNode*>(op.operator->());
        node->attrs.emplace("FInferType", FInferType([](const Attrs&, const Array<Type>&) {
            return TensorType({}, "bool");
        }));
        node->attrs.emplace("FRelayToTE", FRelayToTE(
            [](const Attrs&, const Array<te::Tensor>&, const Type&) { return te::Tensor(); }));
        registered = true;
    }
    return kxc::relay::Op::Get("test_control_predicate");
}

const kxc::relay::Op& UnresolvedNestedOutputOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static bool registered = false;
    if (!registered) {
        OperatorSpec spec;
        spec.name = "test_control_unresolved_nested_output";
        spec.category = "test";
        spec.input_arity.num_inputs = 1;
        spec.output_arity = 3;
        spec.type_relation_key = "FInferType";
        spec.lowering_kind = OperatorLoweringKind::kMultiTE;
        spec.lowering_key = "FRelayToTEMulti";
        Op op = Op::Register(spec);
        auto* node = const_cast<OpNode*>(op.operator->());
        node->attrs.emplace(
            "FInferType",
            FInferType([](const Attrs&, const Array<Type>& inputs) {
                return TupleType(
                    {inputs[0], TupleType({inputs[0], inputs[0]})});
            }));
        node->attrs.emplace(
            "FRelayToTEMulti",
            FRelayToTEMulti([](const Attrs&, const Array<te::Tensor>&,
                               const Type&) { return Array<te::Tensor>{}; }));
        registered = true;
    }
    return kxc::relay::Op::Get("test_control_unresolved_nested_output");
}

std::string ErrorText(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& error) { return error.what(); }
    return "";
}

bool SameIds(const kxc::Array<std::int64_t>& left,
             const kxc::Array<std::int64_t>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

kxc::api::internal::test_support::FakeKernelCallback ArithmeticKernels(
    const std::vector<kxc::api::internal::PrimitiveUnit>& units) {
    std::unordered_map<kxc::api::internal::PrimitiveUnitId, std::string>
        operator_names;
    for (const auto& unit : units) {
        operator_names.emplace(unit.id, unit.call.spec.name);
    }
    return [operator_names = std::move(operator_names)](
               const ControlTask& task,
               const std::vector<FakeValue>& args) {
        const auto found = operator_names.find(task.primitive_unit_id);
        if (found == operator_names.end()) {
            throw std::invalid_argument("unknown fake PrimitiveUnit");
        }
        if (found->second == "add") {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer + args[1].integer)};
        }
        if (found->second == "mul") {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer * args[1].integer)};
        }
        if (found->second == "test_control_predicate") {
            return std::vector<FakeValue>{FakeValue::Bool(args[0].integer < 2)};
        }
        throw std::invalid_argument("unexpected fake kernel reference");
    };
}

bool TestCanonicalAndRepeatedArguments() {
    Var x("x", kI64), y("y", kI64), shared("shared", kI64);
    Function first({x, y}, Let(shared, Add(x, y), Add(shared, shared)));
    ControlPlan plan = LowerRelayToControlPlan(first);
    TEST_CHECK(plan.regions.size() == 1 && plan.regions[0].tasks.size() == 2,
               "Let must not create a task and a shared Call must be produced once");
    const ControlTask& repeated = plan.regions[0].tasks[1];
    TEST_CHECK(repeated.inputs.size() == 1 && repeated.argument_values.size() == 2 &&
                   repeated.argument_values[0] == repeated.argument_values[1] &&
                   repeated.primitive_unit_id >= 0,
               "kernel boundary inputs must be unique and reference a PrimitiveUnit");

    Var x2("x", kI64), y2("y", kI64), shared2("shared", kI64);
    Function equivalent({x2, y2}, Let(shared2, Add(x2, y2), Add(shared2, shared2)));
    TEST_CHECK(plan.CanonicalText() ==
                   LowerRelayToControlPlan(equivalent).CanonicalText(),
               "equivalent Relay must have deterministic ControlPlan text");

    Var px("px", kI64), py("py", kI64), pz("pz", kI64);
    Tuple nested({Add(px, py), Tuple({Add(py, pz), Add(pz, px)})});
    Function projection({px, py, pz}, TupleGetItem(nested, 1));
    const auto projection_lowering = LowerRelayProgram(projection);
    const ControlPlan& projection_plan = projection_lowering.plan;
    TEST_CHECK(projection_plan.graph_outputs.size() == 2 &&
                   projection_plan.regions[0].tasks.size() == 3,
               "nested tuple projection must preserve selected leaves and pure dead work");
    const auto projection_result =
        ControlPlanReferenceExecutor(
            ArithmeticKernels(projection_lowering.primitive_units)).Execute(
            projection_plan,
            {{projection_plan.graph_inputs[0], FakeValue::I64(1)},
             {projection_plan.graph_inputs[1], FakeValue::I64(2)},
             {projection_plan.graph_inputs[2], FakeValue::I64(3)}});
    TEST_CHECK(
        projection_result.values.at(projection_plan.graph_outputs[0]).integer == 5 &&
            projection_result.values.at(projection_plan.graph_outputs[1]).integer == 4,
        "nested tuple projection must execute the complete selected field");

    Var nested_input("nested_input", kI64);
    ControlPlan nested_output_plan = LowerRelayToControlPlan(
        Function({nested_input}, Call(UnresolvedNestedOutputOp(), {nested_input})));
    TEST_CHECK(
            nested_output_plan.graph_outputs.size() == 3 &&
            nested_output_plan.regions[0].tasks.size() == 1 &&
            nested_output_plan.regions[0].tasks[0].primitive_unit_id >= 0,
        "nested Call leaves must reference a shared PrimitiveUnit");
    return true;
}

bool TestSharedLogicalValueContracts() {
    Var lhs("lhs", kI64), rhs("rhs", kI64);
    Function typed =
        kxc::relay::InferTypePass(Function({lhs, rhs}, Add(lhs, rhs)));
    const kxc::api::internal::PartitionedGraph static_plan =
        kxc::api::internal::PartitionValueGraph(
            kxc::api::internal::BuildValueGraph(typed, Device::CPU()));
    const auto control_lowering = LowerRelayProgram(typed);
    const auto& static_graph = static_plan.value_graph;
    const ControlPlan& control_plan = control_lowering.plan;
    TEST_CHECK(static_graph.values.size() == control_plan.values.size(),
               "static and control preparation must produce the same logical leaf count");
    for (std::size_t index = 0; index < static_graph.values.size(); ++index) {
        const auto& static_value = static_graph.values[index];
        const auto& control_value = control_plan.values[index];
        TEST_CHECK(
            static_value.id == control_value.id &&
                static_value.origin == control_value.origin &&
                kxc::api::internal::SameLogicalValueContract(
                    static_value, control_value),
            "static and control preparation must share one value/type/device contract");
    }
    TEST_CHECK(
        static_plan.units.size() == 1 &&
            control_lowering.primitive_units.size() == 1 &&
            static_plan.units[0].semantic_key ==
                control_lowering.primitive_units[0].semantic_key &&
            SameIds(static_plan.units[0].argument_value_ids,
                    control_lowering.primitive_units[0]
                        .argument_value_ids) &&
            SameIds(static_plan.units[0].boundary_input_value_ids,
                    control_lowering.primitive_units[0]
                        .boundary_input_value_ids),
        "static and control topology must build the same PrimitiveUnit contract");
    return true;
}

bool TestIfExecutionAndNestedTuplePhi() {
    Var p("p", TensorType({}, "bool")), q("q", TensorType({}, "bool"));
    Var x("x", kI64), y("y", kI64);
    Function simple({p, x, y}, If(p, Add(x, y), Mul(x, y)));
    const auto simple_lowering = LowerRelayProgram(simple);
    const ControlPlan& simple_plan = simple_lowering.plan;
    ControlPlanReferenceExecutor executor(
        ArithmeticKernels(simple_lowering.primitive_units));
    const auto yes = executor.Execute(simple_plan, {{0, FakeValue::Bool(true)},
                                                     {1, FakeValue::I64(2)},
                                                     {2, FakeValue::I64(3)}});
    const auto no = executor.Execute(simple_plan, {{0, FakeValue::Bool(false)},
                                                    {1, FakeValue::I64(2)},
                                                    {2, FakeValue::I64(3)}});
    TEST_CHECK(yes.values.at(simple_plan.graph_outputs[0]).integer == 5 &&
                   no.values.at(simple_plan.graph_outputs[0]).integer == 6,
               "true and false Relay If must select distinct branch work");

    kxc::runtime::NDArray constant_data = kxc::runtime::NDArray::Zeros(
        {}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
    kxc::Constant constant(constant_data);
    Function captured({p, x}, If(p, Add(x, constant), Mul(x, constant)));
    const auto captured_lowering = LowerRelayProgram(captured);
    const ControlPlan& captured_plan = captured_lowering.plan;
    TEST_CHECK(captured_plan.constant_values.size() == 1,
               "branch constant capture must remain an explicit plan source");
    const auto captured_result =
        ControlPlanReferenceExecutor(
            ArithmeticKernels(captured_lowering.primitive_units)).Execute(
        captured_plan,
        {{captured_plan.graph_inputs[0], FakeValue::Bool(true)},
         {captured_plan.graph_inputs[1], FakeValue::I64(3)},
         {captured_plan.constant_values[0], FakeValue::I64(4)}});
    TEST_CHECK(captured_result.values.at(captured_plan.graph_outputs[0]).integer == 7,
               "captured constant must execute through the frozen source contract");

    Expr inner = If(q, Tuple({Add(x, y), Mul(x, y)}),
                    Tuple({Mul(x, y), Add(x, y)}));
    Function nested({p, q, x, y}, If(p, inner, Tuple({Add(x, x), Mul(y, y)})));
    ControlPlan nested_plan = LowerRelayToControlPlan(nested);
    std::size_t branches = 0;
    for (const auto& region : nested_plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.kind == ControlTaskKind::kBranch) {
                ++branches;
                TEST_CHECK(task.branch.phis.size() == task.outputs.size(),
                           "each flattened If leaf requires one PhiBinding");
            }
        }
    }
    TEST_CHECK(branches == 2 && nested_plan.graph_outputs.size() == 2,
               "nested If and tuple results must remain structured branch-local work");
    return true;
}

bool TestRelaySourceWhileExecution() {
    Var initial("initial", kI64), increment("increment", kI64), state("state");
    const Expr initial_state = Tuple(kxc::Array<Expr>{initial});
    const Expr element = TupleGetItem(state, 0);
    const Expr condition = Call(TestPredicateOp(), {element});
    const Expr body = Tuple(kxc::Array<Expr>{Add(element, increment)});
    const Function loop({initial, increment},
                        While(initial_state, state, condition, body, 3));
    const auto loop_lowering = LowerRelayProgram(loop);
    const ControlPlan& plan = loop_lowering.plan;
    const auto run = [&plan, &loop_lowering](std::int64_t start) {
        return ControlPlanReferenceExecutor(
            ArithmeticKernels(loop_lowering.primitive_units)).Execute(
            plan, {{plan.graph_inputs[0], FakeValue::I64(start)},
                   {plan.graph_inputs[1], FakeValue::I64(1)}});
    };
    const auto zero = run(2);
    const auto one = run(1);
    const auto many = run(0);
    TEST_CHECK(zero.values.at(plan.graph_outputs[0]).integer == 2 &&
                   one.values.at(plan.graph_outputs[0]).integer == 2 &&
                   many.values.at(plan.graph_outputs[0]).integer == 2,
               "Relay-source While must use condition-before-body for zero, one, and multi trips");
    const auto loops = [](
        const kxc::api::internal::test_support::ControlTrace& trace) {
        std::size_t count = 0;
        for (const std::string& event : trace.events) if (event.find("loop:") == 0) ++count;
        return count;
    };
    TEST_CHECK(loops(zero.trace) == 0 && loops(one.trace) == 1 && loops(many.trace) == 2,
               "Relay-source While trip counts must be exact");
    Function exhausted({initial, increment},
        While(initial_state, state, condition, body, 1));
    const auto exhausted_lowering = LowerRelayProgram(exhausted);
    const ControlPlan& exhausted_plan = exhausted_lowering.plan;
    const std::string exhausted_error = ErrorText([&] {
        (void)ControlPlanReferenceExecutor(
            ArithmeticKernels(exhausted_lowering.primitive_units)).Execute(
            exhausted_plan, {{exhausted_plan.graph_inputs[0], FakeValue::I64(0)},
                             {exhausted_plan.graph_inputs[1], FakeValue::I64(1)}});
    });
    TEST_CHECK(exhausted_error.find("max_trip_count exhausted") != std::string::npos,
               "true condition after the bound must throw rather than host-unroll");

    const TensorType boolean({}, "bool");
    Var first("first", boolean), second("second", boolean), third("third", boolean);
    Var tail("tail", boolean), value("value", kI64), shifted_state("shifted_state");
    const Expr shifted_initial = Tuple({first, second, third, tail, value});
    const Expr shifted_condition = TupleGetItem(shifted_state, 0);
    const Expr shifted_body = Tuple({TupleGetItem(shifted_state, 1),
                                     TupleGetItem(shifted_state, 2),
                                     TupleGetItem(shifted_state, 3), tail,
                                     Add(TupleGetItem(shifted_state, 4), increment)});
    const auto shifted_lowering = LowerRelayProgram(Function(
        {first, second, third, tail, value, increment},
        While(shifted_initial, shifted_state, shifted_condition, shifted_body, 3)));
    const ControlPlan& shifted_plan = shifted_lowering.plan;
    const auto shifted = ControlPlanReferenceExecutor(
        ArithmeticKernels(shifted_lowering.primitive_units)).Execute(
        shifted_plan,
        {{shifted_plan.graph_inputs[0], FakeValue::Bool(true)},
         {shifted_plan.graph_inputs[1], FakeValue::Bool(true)},
         {shifted_plan.graph_inputs[2], FakeValue::Bool(true)},
         {shifted_plan.graph_inputs[3], FakeValue::Bool(false)},
         {shifted_plan.graph_inputs[4], FakeValue::I64(7)},
         {shifted_plan.graph_inputs[5], FakeValue::I64(1)}});
    TEST_CHECK(shifted_plan.graph_outputs.size() == 5 &&
                   shifted.values.at(shifted_plan.graph_outputs[4]).integer == 10 &&
                   loops(shifted.trace) == 3,
               "tuple-carried Relay While must preserve all leaves across multiple trips");
    return true;
}

bool TestWhileMappingAndGates() {
    Var predicate("predicate", TensorType({}, "bool"));
    Var initial("initial", kI64), increment("increment", kI64);
    Var state("state", kI64);
    Function loop({predicate, initial, increment},
                  While(initial, state, predicate, Add(state, increment), 3));
    const ControlPlan plan = LowerRelayToControlPlan(loop);
    TEST_CHECK(plan.regions.size() == 3 && plan.graph_outputs.size() == 1,
               "While must lower to condition/body regions and one carried result");
    const ControlTask& task = plan.regions[0].tasks.front();
    TEST_CHECK(task.kind == ControlTaskKind::kLoop && task.loop.carried.size() == 1 &&
                   task.loop.max_trip_count == 3 &&
                   task.loop.condition_region != task.loop.body_region,
               "While must map exactly to bounded LoopSpec");
    const auto& carried = task.loop.carried.front();
    TEST_CHECK(carried.result == task.outputs.front() &&
                   carried.initial == plan.graph_inputs[1] &&
                   carried.body_argument != carried.backedge,
               "LoopSpec must retain result/initial/body-argument/backedge roles");
    TEST_CHECK(plan.CanonicalText() == LowerRelayToControlPlan(loop).CanonicalText(),
               "While ControlPlan text must be deterministic");

    Var inner_predicate("inner_predicate", TensorType({}, "bool"));
    Function nested({predicate, inner_predicate, initial, increment},
        While(initial, state, predicate,
              If(inner_predicate, Add(state, increment), state), 3));
    const ControlPlan nested_plan = LowerRelayToControlPlan(nested);
    std::size_t loops = 0, branches = 0;
    for (const auto& region : nested_plan.regions) for (const auto& nested_task : region.tasks) {
        loops += nested_task.kind == ControlTaskKind::kLoop;
        branches += nested_task.kind == ControlTaskKind::kBranch;
    }
    TEST_CHECK(loops == 1 && branches == 1,
               "nested Relay If in a While body must remain structured control regions");

    Var bad_state("bad_state", TensorType({}, "float32"));
    const std::string type_error = ErrorText([&] {
        (void)kxc::relay::InferTypePass(Function(
            {predicate, initial}, While(initial, bad_state, predicate, bad_state, 0)));
    });
    TEST_CHECK(type_error.find("loop_var annotation mismatch") != std::string::npos,
               "While must reject an inexact lexical state binder type");
    const std::string bound_error = ErrorText([&] {
        (void)kxc::relay::InferTypePass(Function(
            {predicate, initial}, While(initial, state, predicate, state, -1)));
    });
    TEST_CHECK(bound_error.find("max_trip_count") != std::string::npos,
               "While must reject a negative mandatory trip bound");

    Function typed_loop = kxc::relay::InferTypePass(loop);
    const std::string graph_error = ErrorText([&] {
        (void)kxc::api::internal::BuildValueGraph(typed_loop);
    });
    TEST_CHECK(graph_error.find("control_flow.loop") != std::string::npos,
               "ordinary ValueGraph compilation must reject While explicitly");
    return true;
}

bool TestStaticAndControlGates() {
    Var dynamic("dynamic", TensorType({-1}, "int64"));
    const std::string dynamic_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function({dynamic}, dynamic));
    });
    TEST_CHECK(dynamic_error.find("static_exact_shape") != std::string::npos,
               "dynamic shapes must retain capability diagnostics");

    Var predicate("predicate", TensorType({}, "bool")), x("x", kI64), y("y", kI64);
    predicate.set_virtual_device(kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    const std::string predicate_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function({predicate, x, y}, If(predicate, x, y)));
    });
    TEST_CHECK(predicate_error.find("branch predicate must be a CPU scalar bool") != std::string::npos,
               "ControlPlan must reject an implicit non-CPU predicate copy");

    Var loop_condition("loop_condition", TensorType({}, "bool"));
    loop_condition.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    Var loop_initial("loop_initial", kI64), loop_state("loop_state", kI64);
    const std::string loop_condition_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function(
            {loop_condition, loop_initial},
            While(loop_initial, loop_state, loop_condition, loop_state, 0)));
    });
    TEST_CHECK(loop_condition_error.find("required capability=cpu_loop_condition_placement") !=
                   std::string::npos,
               "While condition placement must fail in the capability gate before plan lowering");

    Var placement_condition("placement_condition", TensorType({}, "bool"));
    Var placement_initial("placement_initial", kI64),
        placement_state("placement_state", kI64);
    placement_state.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    const std::string loop_state_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function(
            {placement_condition, placement_initial},
            While(placement_initial, placement_state, placement_condition,
                  placement_state, 0)));
    });
    TEST_CHECK(loop_state_error.find("required capability=exact_loop_state_placement") !=
                   std::string::npos,
               "While state placement must fail in the capability gate before plan lowering");

    Var mismatch_predicate("mismatch_predicate", TensorType({}, "bool"));
    Var mismatch_x("mismatch_x", kI64), mismatch_y("mismatch_y", kI64);
    If mismatch(mismatch_predicate, mismatch_x, mismatch_y);
    mismatch.set_virtual_device(kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    const std::string mismatch_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(
            Function({mismatch_predicate, mismatch_x, mismatch_y}, mismatch));
    });
    TEST_CHECK(mismatch_error.find("phi sources/results") != std::string::npos,
               "If results and both Phi sources must have exact device contracts");

    Var cuda_x("cuda_x", kI64), cuda_y("cuda_y", kI64);
    cuda_x.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(0)));
    cuda_y.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(0)));
    Call cross_ordinal = Add(cuda_x, cuda_y);
    cross_ordinal.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(1)));
    const std::string ordinal_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(
            Function({cuda_x, cuda_y}, cross_ordinal));
    });
    TEST_CHECK(ordinal_error.find("input/output device") != std::string::npos,
               "ControlPlan must preserve and compare CUDA device ordinals");

    Var structural_x("structural_x", kI64);
    Var structural_y("structural_y", kI64);
    Tuple placed_tuple({structural_x, structural_y});
    placed_tuple.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(1)));
    const std::string structural_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(
            Function({structural_x, structural_y}, placed_tuple));
    });
    TEST_CHECK(structural_error.find("structural alias placement") !=
                   std::string::npos,
               "placed structural aliases must not erase device identity");

    Var placed_binding("placed_binding", kI64);
    Let placed_let(placed_binding, Add(structural_x, structural_y),
                   placed_binding);
    placed_let.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(1)));
    const std::string let_placement_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(
            Function({structural_x, structural_y}, placed_let));
    });
    TEST_CHECK(let_placement_error.find("structural alias placement") !=
                   std::string::npos,
               "placed Let aliases must not erase device identity");

    const kxc::relay::Op& add = kxc::relay::Op::Get("add");
    auto* add_node = const_cast<kxc::relay::OpNode*>(add.operator->());
    const kxc::relay::OperatorSpec saved_spec = add_node->spec;
    Var gate_x("gate_x", kI64), gate_y("gate_y", kI64);
    add_node->spec.alias_contract = "must_alias";
    const std::string alias_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->spec = saved_spec;
    TEST_CHECK(alias_error.find("non-aliasing kernel") != std::string::npos,
               "aliasing OperatorSpecs must not become kernel tasks");

    add_node->spec.effect = kxc::relay::OperatorEffectKind::kStateful;
    const std::string effect_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->spec = saved_spec;
    TEST_CHECK(effect_error.find("pure deterministic non-aliasing kernel") != std::string::npos,
               "stateful OperatorSpecs must not become kernel tasks");

    const std::string lowering_key = saved_spec.lowering_key;
    const std::any saved_lowering = add_node->attrs.at(lowering_key);
    add_node->attrs[lowering_key] = kxc::relay::FRelayToTE{};
    const std::string empty_hook_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(
            Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->attrs[lowering_key] = saved_lowering;
    TEST_CHECK(empty_hook_error.find("operator_implementation_binding") !=
                   std::string::npos,
               "empty TE hooks must fail before an unresolved task is prepared");

    Var duplicate("duplicate", kI64);
    const std::string duplicate_error = ErrorText([&] {
        (void)LowerRelayToControlPlan(Function({duplicate}, Tuple({duplicate, duplicate})));
    });
    TEST_CHECK(duplicate_error.find("duplicate graph output") != std::string::npos,
               "ControlPlan v2 graph outputs must be unique");

    Var ordinary_predicate("ordinary_predicate", TensorType({}, "bool"));
    Var a("a", kI64), b("b", kI64);
    Function ordinary_if({ordinary_predicate, a, b},
                         If(ordinary_predicate, a, b));
    ordinary_if = kxc::relay::InferTypePass(ordinary_if);
    const std::string graph_error = ErrorText([&] {
        (void)kxc::api::internal::BuildValueGraph(ordinary_if);
    });
    TEST_CHECK(graph_error.find("required capability=if") != std::string::npos,
               "ordinary ValueGraph compilation must still reject Relay If");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"canonical_repeated_arguments", TestCanonicalAndRepeatedArguments},
        {"shared_logical_value_contracts", TestSharedLogicalValueContracts},
        {"if_execution_nested_tuple_phi", TestIfExecutionAndNestedTuplePhi},
        {"relay_source_while_execution", TestRelaySourceWhileExecution},
        {"while_mapping_and_gates", TestWhileMappingAndGates},
        {"static_and_control_gates", TestStaticAndControlGates},
    };
    for (const auto& test : tests) {
        try {
            if (!test.second()) return 1;
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
