/*! \file test/executable_capability_test.cpp
 * \brief Standalone static-exact executable capability and ValueGraph checks.
 */

#include <any>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/executable_capability.h"
#include "../src/compiler/internal/lowered_graph.h"
#include "../src/compiler/internal/value_graph.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/transforms/infer_type.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

std::string ErrorText(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

const kxc::relay::Op& NestedMultiOutputOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static bool registered = false;
    if (!registered) {
        OperatorSpec spec;
        spec.name = "test_nested_multi_output";
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
    return Op::Get("test_nested_multi_output");
}

const kxc::relay::Op& MultiOutputContractOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static bool registered = false;
    if (!registered) {
        OperatorSpec spec;
        spec.name = "test_multi_output_contract";
        spec.category = "test";
        spec.input_arity.num_inputs = 1;
        spec.output_arity = 2;
        spec.type_relation_key = "FInferType";
        spec.lowering_kind = OperatorLoweringKind::kMultiTE;
        spec.lowering_key = "FRelayToTEMulti";
        Op op = Op::Register(spec);
        auto* node = const_cast<OpNode*>(op.operator->());
        node->attrs.emplace(
            "FInferType",
            FInferType([](const Attrs&, const Array<Type>& inputs) {
                return TupleType({inputs[0], inputs[0]});
            }));
        node->attrs.emplace(
            "FRelayToTEMulti",
            FRelayToTEMulti([](const Attrs&, const Array<te::Tensor>& inputs,
                               const Type&) {
                return Array<te::Tensor>{inputs[0], inputs[0]};
            }));
        registered = true;
    }
    return Op::Get("test_multi_output_contract");
}

bool SameIds(const kxc::Array<int64_t>& lhs, const kxc::Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool TestStaticExactDiagnosticsAndIfGate() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");

    Var untyped("untyped");
    const std::string untyped_error = ErrorText([&] {
        VerifyExecutableCapability(Function({untyped}, untyped),
                                    StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(untyped_error.find("required capability=defined_typed_relay") !=
                   std::string::npos,
               "untyped Relay must report the typed capability");

    Var free_var("free", type);
    Function free_function({}, free_var);
    free_function = relay::InferTypePass(free_function);
    const std::string free_error = ErrorText([&] {
        VerifyExecutableCapability(free_function, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(free_error.find("path=function.body") != std::string::npos &&
                   free_error.find("node=Var") != std::string::npos &&
                   free_error.find("required capability=lexically_bound_var") !=
                       std::string::npos,
               "free Var diagnostic must carry deterministic path, kind, and capability");

    Var dynamic("dynamic", TensorType({-1, 4}, "float32"));
    Function dynamic_function({dynamic}, dynamic);
    dynamic_function = relay::InferTypePass(dynamic_function);
    const std::string dynamic_error = ErrorText([&] {
        VerifyExecutableCapability(dynamic_function,
                                    StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(dynamic_error.find("shape[0]") != std::string::npos &&
                   dynamic_error.find("static_exact_shape") != std::string::npos,
               "every negative TensorType dimension must be rejected");

    Var predicate("predicate", TensorType({}, "bool"));
    Var x("x", type);
    Var y("y", type);
    Function conditional({predicate, x, y}, If(predicate, x, y));
    conditional = relay::InferTypePass(conditional);
    const std::string if_error = ErrorText([&] {
        VerifyExecutableCapability(conditional, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(if_error.find("path=function.body") != std::string::npos &&
                   if_error.find("node=If") != std::string::npos &&
                   if_error.find("required capability=if") != std::string::npos,
               "default static-dataflow capability must reject If with a clear gate");

    ExecutableCapabilityOptions with_if = StaticDataflowExecutableCapabilities();
    with_if.allow_if = true;
    VerifyExecutableCapability(conditional, with_if);
    return true;
}

bool TestFunctionValueIsRejected() {
    using namespace kxc;
    using namespace kxc::api::internal;
    Var x("x", TensorType({4}, "float32"));
    Function inner({}, x);
    Function outer({x}, inner);
    outer = relay::InferTypePass(outer);
    const std::string error = ErrorText([&] {
        VerifyExecutableCapability(outer, StaticDataflowExecutableCapabilities());
    });
    TEST_CHECK(error.find("node=Function") != std::string::npos &&
                   error.find("first_order_relay") != std::string::npos,
               "function values and closures must be rejected");
    return true;
}

bool TestValueGraphLetMatchesNestedTopology() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Function nested({x, y}, Add(Add(x, y), y));
    nested = relay::InferTypePass(nested);

    Var lx("x", type);
    Var ly("y", type);
    Var shared("shared", type);
    Function with_let({lx, ly}, Let(shared, Add(lx, ly), Add(shared, ly)));
    with_let = relay::InferTypePass(with_let);

    const ValueGraph nested_graph = BuildValueGraph(nested);
    const ValueGraph let_graph = BuildValueGraph(with_let);
    TEST_CHECK(nested_graph.values.size() == let_graph.values.size(),
               "lexical Let changed value count");
    TEST_CHECK(nested_graph.calls.size() == let_graph.calls.size(),
               "lexical Let changed Call count");
    TEST_CHECK(SameIds(nested_graph.output_value_ids, let_graph.output_value_ids),
               "lexical Let changed stable output value ids");
    TEST_CHECK(let_graph.calls.size() == 2 &&
                   let_graph.calls[1].argument_value_ids.size() == 2 &&
                   let_graph.calls[1].argument_value_ids[0] == 2 &&
                   let_graph.calls[1].argument_value_ids[1] == 1,
               "bound Var must resolve exactly to the value ids of its Let value");

    Var tx("tx", type);
    Var ty("ty", type);
    Var tuple_value("tuple_value", TupleType({type, type}));
    Function tuple_let({tx, ty},
                       Let(tuple_value, Tuple({Add(tx, ty), Add(ty, tx)}),
                           Tuple({TupleGetItem(tuple_value, 1),
                                  TupleGetItem(tuple_value, 0)})));
    tuple_let = relay::InferTypePass(tuple_let);
    const ValueGraph tuple_graph = BuildValueGraph(tuple_let);
    TEST_CHECK(tuple_graph.calls.size() == 2 && tuple_graph.output_value_ids.size() == 2 &&
                   tuple_graph.output_value_ids[0] == 3 && tuple_graph.output_value_ids[1] == 2,
               "Tuple Let and TupleGetItem must retain their resolved leaf ids");
    return true;
}

bool TestPlacementAndOperatorContractsFailClosed() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");

    Var placed("placed", type);
    placed.set_virtual_device(VirtualDevice::ForDevice(Device::CUDA()));
    Function placed_function({placed}, placed);
    placed_function = relay::InferTypePass(placed_function);
    const std::string missing_device = ErrorText(
        [&] { (void)BuildValueGraph(placed_function); });
    TEST_CHECK(missing_device.find("explicit_execution_device") !=
                   std::string::npos,
               "explicit placement without an execution device must fail");
    const std::string mismatched_device = ErrorText(
        [&] { (void)BuildValueGraph(placed_function, Device::CPU()); });
    TEST_CHECK(mismatched_device.find("matching_execution_device") !=
                   std::string::npos,
               "placement must match the static execution device");
    (void)BuildValueGraph(placed_function, Device::CUDA());

    const relay::Op& add = relay::Op::Get("add");
    auto* add_node = const_cast<relay::OpNode*>(add.operator->());
    const relay::OperatorSpec saved_spec = add_node->spec;
    Var x("x", type);
    Var y("y", type);
    Function add_function({x, y}, Add(x, y));
    add_function = relay::InferTypePass(add_function);

    add_node->spec.effect = relay::OperatorEffectKind::kStateful;
    const std::string effect_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->spec = saved_spec;
    TEST_CHECK(effect_error.find("pure_deterministic_no_alias") !=
                   std::string::npos,
               "stateful ordinary operators must not enter static ValueGraph");

    add_node->spec.alias_contract = "must_alias";
    const std::string alias_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->spec = saved_spec;
    TEST_CHECK(alias_error.find("pure_deterministic_no_alias") !=
                   std::string::npos,
               "aliasing ordinary operators must not enter static ValueGraph");

    add_node->spec.output_arity = 2;
    const std::string output_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->spec = saved_spec;
    TEST_CHECK(output_error.find("operator_output_arity") !=
                   std::string::npos,
               "output arity must fail in the capability gate");

    add_node->spec.input_arity.num_inputs = 1;
    const std::string input_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->spec = saved_spec;
    TEST_CHECK(input_error.find("operator_input_arity") !=
                   std::string::npos,
               "input arity must fail in the capability gate");

    const std::string lowering_key = saved_spec.lowering_key;
    const std::any saved_lowering = add_node->attrs.at(lowering_key);
    add_node->attrs.erase(lowering_key);
    const std::string binding_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->attrs.emplace(lowering_key, saved_lowering);
    add_node->attrs[lowering_key] = relay::FRelayToTE{};
    const std::string empty_binding_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->attrs[lowering_key] = relay::FRelayToTEMulti(
        [](const relay::Attrs&, const Array<te::Tensor>&, const Type&) {
            return Array<te::Tensor>{};
        });
    const std::string wrong_binding_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->attrs[lowering_key] = saved_lowering;
    const std::string wrong_lowering_key = "test_wrong_lowering_key";
    add_node->attrs.emplace(wrong_lowering_key, saved_lowering);
    add_node->spec.lowering_key = wrong_lowering_key;
    const std::string wrong_key_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->spec = saved_spec;
    add_node->attrs.erase(wrong_lowering_key);
    const std::string relation_key = saved_spec.type_relation_key;
    const std::any saved_relation = add_node->attrs.at(relation_key);
    add_node->attrs[relation_key] = relay::FInferType{};
    const std::string empty_relation_error = ErrorText(
        [&] { (void)BuildValueGraph(add_function, Device::CPU()); });
    add_node->attrs[relation_key] = saved_relation;
    TEST_CHECK(binding_error.find("operator_implementation_binding") !=
                   std::string::npos,
               "missing lowering bindings must fail in the capability gate");
    TEST_CHECK(empty_binding_error.find("operator_implementation_binding") !=
                   std::string::npos &&
                   empty_binding_error.find("empty") != std::string::npos,
               "empty lowering functions must fail in the capability gate");
    TEST_CHECK(wrong_binding_error.find("operator_implementation_binding") !=
                   std::string::npos,
               "wrong lowering binding types must fail in the capability gate");
    TEST_CHECK(wrong_key_error.find("operator_implementation_binding") !=
                   std::string::npos,
               "non-canonical lowering keys must fail in the capability gate");
    TEST_CHECK(
        empty_relation_error.find("operator_implementation_binding") !=
                std::string::npos &&
            empty_relation_error.find("empty") != std::string::npos,
        "empty type relations must fail in the capability gate");
    return true;
}

bool TestTEOutputContractsFailClosed() {
    using namespace kxc;
    using namespace kxc::api::internal;
    using namespace kxc::relay;
    const TensorType type({4}, "float32");

    const Op& add = Op::Get("add");
    auto* add_node = const_cast<OpNode*>(add.operator->());
    const std::string add_key = add_node->spec.lowering_key;
    const std::any saved_add_lowering = add_node->attrs.at(add_key);
    add_node->attrs[add_key] = FRelayToTE(
        [](const Attrs&, const Array<te::Tensor>&, const Type&) {
            return te::Tensor();
        });
    Var lhs("lhs", type), rhs("rhs", type);
    const std::string single_undefined = ErrorText([&] {
        (void)LowerGraph(Function({lhs, rhs}, Add(lhs, rhs)), Device::CPU());
    });
    add_node->attrs[add_key] = saved_add_lowering;

    const Op& multi = MultiOutputContractOp();
    auto* multi_node = const_cast<OpNode*>(multi.operator->());
    const std::string multi_key = multi_node->spec.lowering_key;
    const std::any saved_multi_lowering = multi_node->attrs.at(multi_key);
    Var input("input", type);
    Function function({input}, Call(multi, {input}));
    const auto lowering_error = [&](FRelayToTEMulti lowering) {
        multi_node->attrs[multi_key] = std::move(lowering);
        return ErrorText(
            [&] { (void)LowerGraph(function, Device::CPU()); });
    };

    const std::string multi_empty_hook =
        lowering_error(FRelayToTEMulti{});
    const std::string multi_empty_outputs = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>&, const Type&) {
            return Array<te::Tensor>{};
        }));
    const std::string multi_count = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{inputs[0]};
        }));
    const std::string multi_undefined = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{inputs[0], te::Tensor()};
        }));
    const std::string multi_dtype = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{
                inputs[0],
                te::placeholder(inputs[0]->shape, tir::DataType::Float(64),
                                "wrong_dtype")};
        }));
    const std::string multi_rank = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{
                inputs[0],
                te::placeholder({tir::IntImm(4), tir::IntImm(1)},
                                inputs[0]->dtype, "wrong_rank")};
        }));
    const std::string multi_shape = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{
                inputs[0],
                te::placeholder({tir::IntImm(5)}, inputs[0]->dtype,
                                "wrong_shape")};
        }));
    const std::string multi_dynamic_shape = lowering_error(FRelayToTEMulti(
        [](const Attrs&, const Array<te::Tensor>& inputs, const Type&) {
            return Array<te::Tensor>{
                inputs[0],
                te::placeholder({tir::Var("n")}, inputs[0]->dtype,
                                "dynamic_shape")};
        }));
    multi_node->attrs[multi_key] = saved_multi_lowering;

    TEST_CHECK(single_undefined.find("undefined tensor") != std::string::npos,
               "single-output lowering must reject an undefined TE tensor");
    TEST_CHECK(multi_empty_hook.find("empty") != std::string::npos,
               "multi-output lowering must reject an empty std::function");
    TEST_CHECK(multi_empty_outputs.find("result count mismatch") !=
                   std::string::npos,
               "multi-output lowering must reject an empty output list");
    TEST_CHECK(multi_count.find("result count mismatch") != std::string::npos,
               "multi-output lowering must reject the wrong flattened count");
    TEST_CHECK(multi_undefined.find("undefined tensor") != std::string::npos,
               "multi-output lowering must reject an undefined TE tensor");
    TEST_CHECK(multi_dtype.find("dtype mismatch") != std::string::npos,
               "TE output dtype must match its stable ValueInfo");
    TEST_CHECK(multi_rank.find("rank mismatch") != std::string::npos,
               "TE output rank must match its stable ValueInfo");
    TEST_CHECK(multi_shape.find("shape mismatch") != std::string::npos,
               "TE output shape must match its stable ValueInfo");
    TEST_CHECK(multi_dynamic_shape.find("not static") != std::string::npos,
               "TE output shape must be static before artifact lowering");
    return true;
}

bool TestNestedTupleGetItemKeepsAllLeaves() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Var z("z", type);
    Tuple nested({Add(x, y), Tuple({Add(y, z), Add(z, x)})});
    Function function({x, y, z}, TupleGetItem(nested, 1));
    function = relay::InferTypePass(function);
    const ValueGraph graph = BuildValueGraph(function, Device::CPU());
    TEST_CHECK(graph.calls.size() == 3 &&
                   SameIds(graph.output_value_ids, {4, 5}),
               "TupleGetItem of a nested field must return every flattened leaf");

    Var input("input", type);
    Function nested_call({input}, Call(NestedMultiOutputOp(), {input}));
    nested_call = relay::InferTypePass(nested_call);
    const std::string nested_call_error = ErrorText(
        [&] { (void)LowerGraph(nested_call, Device::CPU()); });
    TEST_CHECK(nested_call_error.find("flat_multi_tensor_output") !=
                   std::string::npos,
               "nested tuple Call outputs must fail in the static capability gate");
    return true;
}

bool TestTupleParameterCapabilityIsNotOverclaimed() {
    using namespace kxc;
    using namespace kxc::api::internal;
    const TensorType tensor({4}, "float32");
    const TupleType tuple_type({tensor, tensor});
    Var parameter("parameter", tuple_type);
    SetCheckedType(Expr(ObjectRef(parameter)), tuple_type);
    TupleGetItem first(parameter, 0);
    SetCheckedType(first, tensor);
    Function function({parameter}, first);
    SetCheckedType(function, tensor);
    const std::string error = ErrorText([&] {
        VerifyExecutableCapability(function,
                                    StaticDataflowExecutableCapabilities(
                                        Device::CPU()));
    });
    TEST_CHECK(error.find("required capability=tensor_parameter") !=
                   std::string::npos,
               "static ValueGraph verifier must reject tuple parameters itself");
    return true;
}

bool TestValueGraphFreeVarRejects() {
    using namespace kxc;
    using namespace kxc::api::internal;
    Var free_var("free", TensorType({4}, "float32"));
    Function function({}, free_var);
    function = relay::InferTypePass(function);
    const std::string error = ErrorText([&] { (void)BuildValueGraph(function); });
    TEST_CHECK(error.find("lexically_bound_var") != std::string::npos,
               "ValueGraph must capability-reject free Vars before graph construction");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"static_exact_diagnostics_and_if_gate", TestStaticExactDiagnosticsAndIfGate},
        {"function_value_rejected", TestFunctionValueIsRejected},
        {"value_graph_let_matches_nested", TestValueGraphLetMatchesNestedTopology},
        {"placement_and_operator_contracts",
         TestPlacementAndOperatorContractsFailClosed},
        {"te_output_contracts", TestTEOutputContractsFailClosed},
        {"nested_tuple_get_item_leaves", TestNestedTupleGetItemKeepsAllLeaves},
        {"tuple_parameter_gate", TestTupleParameterCapabilityIsNotOverclaimed},
        {"value_graph_free_var_rejected", TestValueGraphFreeVarRejects},
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
