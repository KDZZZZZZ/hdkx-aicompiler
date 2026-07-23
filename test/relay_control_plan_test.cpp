/*! \file test/relay_control_plan_test.cpp
 * \brief Standalone Relay-to-ControlPlan preparation checks.
 */

#include <any>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/value_graph.h"
#include "kxc/compiler/control_flow.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/transforms/infer_type.h"
#include "support/control_plan_reference_executor.h"

namespace {

#define TEST_CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

using kxc::Call;
using kxc::Expr;
using kxc::Function;
using kxc::If;
using kxc::Let;
using kxc::TensorType;
using kxc::Tuple;
using kxc::TupleGetItem;
using kxc::Var;
using kxc::runtime::ControlPlan;
using kxc::runtime::ControlTask;
using kxc::runtime::ControlTaskKind;
using kxc::runtime::test_support::ControlPlanReferenceExecutor;
using kxc::runtime::test_support::FakeValue;

const TensorType kI64({}, "int64");

Call Add(const Expr& lhs, const Expr& rhs) {
    return Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

Call Mul(const Expr& lhs, const Expr& rhs) {
    return Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
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

kxc::runtime::test_support::FakeKernelCallback ArithmeticKernels() {
    return [](const ControlTask& task, const std::vector<FakeValue>& args) {
        if (task.kernel_ref.find("name=add;") != std::string::npos) {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer + args[1].integer)};
        }
        if (task.kernel_ref.find("name=mul;") != std::string::npos) {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer * args[1].integer)};
        }
        throw std::invalid_argument("unexpected fake kernel reference");
    };
}

bool TestCanonicalAndRepeatedArguments() {
    Var x("x", kI64), y("y", kI64), shared("shared", kI64);
    Function first({x, y}, Let(shared, Add(x, y), Add(shared, shared)));
    ControlPlan plan = kxc::api::LowerRelayToControlPlan(first);
    TEST_CHECK(plan.regions.size() == 1 && plan.regions[0].tasks.size() == 2,
               "Let must not create a task and a shared Call must be produced once");
    const ControlTask& repeated = plan.regions[0].tasks[1];
    TEST_CHECK(repeated.inputs.size() == 1 && repeated.argument_values.size() == 2 &&
                   repeated.argument_values[0] == repeated.argument_values[1] &&
                   repeated.binding_state ==
                       kxc::runtime::KernelBindingState::kUnresolvedRelayKernel,
               "kernel boundary inputs must be unique and Relay kernels must remain unresolved");

    Var x2("x", kI64), y2("y", kI64), shared2("shared", kI64);
    Function equivalent({x2, y2}, Let(shared2, Add(x2, y2), Add(shared2, shared2)));
    TEST_CHECK(plan.CanonicalText() ==
                   kxc::api::LowerRelayToControlPlan(equivalent).CanonicalText(),
               "equivalent Relay must have deterministic ControlPlan text");

    Var px("px", kI64), py("py", kI64), pz("pz", kI64);
    Tuple nested({Add(px, py), Tuple({Add(py, pz), Add(pz, px)})});
    Function projection({px, py, pz}, TupleGetItem(nested, 1));
    ControlPlan projection_plan =
        kxc::api::LowerRelayToControlPlan(projection);
    TEST_CHECK(projection_plan.graph_outputs.size() == 2 &&
                   projection_plan.regions[0].tasks.size() == 3,
               "nested tuple projection must preserve selected leaves and pure dead work");
    const auto projection_result =
        ControlPlanReferenceExecutor(ArithmeticKernels()).Execute(
            projection_plan,
            {{projection_plan.graph_inputs[0], FakeValue::I64(1)},
             {projection_plan.graph_inputs[1], FakeValue::I64(2)},
             {projection_plan.graph_inputs[2], FakeValue::I64(3)}});
    TEST_CHECK(
        projection_result.values.at(projection_plan.graph_outputs[0]).integer == 5 &&
            projection_result.values.at(projection_plan.graph_outputs[1]).integer == 4,
        "nested tuple projection must execute the complete selected field");

    Var nested_input("nested_input", kI64);
    ControlPlan nested_output_plan = kxc::api::LowerRelayToControlPlan(
        Function({nested_input}, Call(UnresolvedNestedOutputOp(), {nested_input})));
    TEST_CHECK(
        nested_output_plan.graph_outputs.size() == 3 &&
            nested_output_plan.regions[0].tasks.size() == 1 &&
            nested_output_plan.regions[0].tasks[0].binding_state ==
                kxc::runtime::KernelBindingState::kUnresolvedRelayKernel,
        "nested Call leaves may be prepared only as an unresolved Relay kernel");
    return true;
}

bool TestIfExecutionAndNestedTuplePhi() {
    Var p("p", TensorType({}, "bool")), q("q", TensorType({}, "bool"));
    Var x("x", kI64), y("y", kI64);
    Function simple({p, x, y}, If(p, Add(x, y), Mul(x, y)));
    ControlPlan simple_plan = kxc::api::LowerRelayToControlPlan(simple);
    ControlPlanReferenceExecutor executor(ArithmeticKernels());
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
    ControlPlan captured_plan = kxc::api::LowerRelayToControlPlan(captured);
    TEST_CHECK(captured_plan.constant_values.size() == 1,
               "branch constant capture must remain an explicit plan source");
    const auto captured_result = executor.Execute(
        captured_plan,
        {{captured_plan.graph_inputs[0], FakeValue::Bool(true)},
         {captured_plan.graph_inputs[1], FakeValue::I64(3)},
         {captured_plan.constant_values[0], FakeValue::I64(4)}});
    TEST_CHECK(captured_result.values.at(captured_plan.graph_outputs[0]).integer == 7,
               "captured constant must execute through the frozen source contract");

    Expr inner = If(q, Tuple({Add(x, y), Mul(x, y)}),
                    Tuple({Mul(x, y), Add(x, y)}));
    Function nested({p, q, x, y}, If(p, inner, Tuple({Add(x, x), Mul(y, y)})));
    ControlPlan nested_plan = kxc::api::LowerRelayToControlPlan(nested);
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

bool TestStaticAndControlGates() {
    Var dynamic("dynamic", TensorType({-1}, "int64"));
    const std::string dynamic_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(Function({dynamic}, dynamic));
    });
    TEST_CHECK(dynamic_error.find("static_exact_shape") != std::string::npos,
               "dynamic shapes must retain capability diagnostics");

    Var predicate("predicate", TensorType({}, "bool")), x("x", kI64), y("y", kI64);
    predicate.set_virtual_device(kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    const std::string predicate_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(Function({predicate, x, y}, If(predicate, x, y)));
    });
    TEST_CHECK(predicate_error.find("branch predicate must be a CPU scalar bool") != std::string::npos,
               "ControlPlan must reject an implicit non-CPU predicate copy");

    Var mismatch_predicate("mismatch_predicate", TensorType({}, "bool"));
    Var mismatch_x("mismatch_x", kI64), mismatch_y("mismatch_y", kI64);
    If mismatch(mismatch_predicate, mismatch_x, mismatch_y);
    mismatch.set_virtual_device(kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    const std::string mismatch_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(
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
        (void)kxc::api::LowerRelayToControlPlan(
            Function({cuda_x, cuda_y}, cross_ordinal));
    });
    TEST_CHECK(ordinal_error.find("one explicit device") != std::string::npos,
               "ControlPlan must preserve and compare CUDA device ordinals");

    Var structural_x("structural_x", kI64);
    Var structural_y("structural_y", kI64);
    Tuple placed_tuple({structural_x, structural_y});
    placed_tuple.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA(1)));
    const std::string structural_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(
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
        (void)kxc::api::LowerRelayToControlPlan(
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
        (void)kxc::api::LowerRelayToControlPlan(Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->spec = saved_spec;
    TEST_CHECK(alias_error.find("non-aliasing kernel") != std::string::npos,
               "aliasing OperatorSpecs must not become kernel tasks");

    add_node->spec.effect = kxc::relay::OperatorEffectKind::kStateful;
    const std::string effect_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->spec = saved_spec;
    TEST_CHECK(effect_error.find("pure deterministic non-aliasing kernel") != std::string::npos,
               "stateful OperatorSpecs must not become kernel tasks");

    const std::string lowering_key = saved_spec.lowering_key;
    const std::any saved_lowering = add_node->attrs.at(lowering_key);
    add_node->attrs[lowering_key] = kxc::relay::FRelayToTE{};
    const std::string empty_hook_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(
            Function({gate_x, gate_y}, Add(gate_x, gate_y)));
    });
    add_node->attrs[lowering_key] = saved_lowering;
    TEST_CHECK(empty_hook_error.find("wrong type or is empty") !=
                   std::string::npos,
               "empty TE hooks must fail before an unresolved task is prepared");

    Var duplicate("duplicate", kI64);
    const std::string duplicate_error = ErrorText([&] {
        (void)kxc::api::LowerRelayToControlPlan(Function({duplicate}, Tuple({duplicate, duplicate})));
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
        {"if_execution_nested_tuple_phi", TestIfExecutionAndNestedTuplePhi},
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
