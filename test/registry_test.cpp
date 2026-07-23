/*! \file test/registry_test.cpp
 * \brief Characterize global PackedFunc registry behavior.
 */

#include "kxc/ffi/registry.h"
#include "kxc/relay/op.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, message)                         \
    do {                                                  \
        if (!(cond)) {                                    \
            std::cerr << "[FAIL] " << (message) << '\n'; \
            return false;                                 \
        }                                                 \
    } while (false)

bool TestSetGetAndDuplicateFailure() {
    constexpr const char* kName = "kxc.test.registry.increment";
    kxc::Registry& registry = kxc::Registry::Global();
    registry.Set(kName, kxc::ToPackedFunc([](int value) { return value + 1; }));

    kxc::PackedFunc increment = registry.Get(kName);
    TEST_CHECK(increment.defined(), "registered function should be discoverable");
    TEST_CHECK(increment(41).As<int64_t>() == 42,
               "registered function should preserve its body");
    TEST_CHECK(!registry.Get("kxc.test.registry.missing").defined(),
               "missing registry entry should return an empty function");

    try {
        registry.Set(kName, kxc::ToPackedFunc([](int value) { return value; }));
        TEST_CHECK(false, "duplicate registration should throw");
    } catch (const std::runtime_error&) {
    }
    return true;
}

template <typename Fn>
bool ExpectThrow(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool TestRelayOperatorRegistryLookupAndSpecs() {
    const kxc::relay::Op& add = kxc::relay::Op::Get("add");
    TEST_CHECK(add->name == "add", "registered op lookup should return canonical op");
    TEST_CHECK(add.has_spec(), "registered op should expose OperatorSpec");
    TEST_CHECK(add.spec().type_relation_key == "FInferType",
               "fluent registration should derive type relation key");
    TEST_CHECK(add.spec().lowering_kind == kxc::relay::OperatorLoweringKind::kSingleTE,
               "fluent registration should derive single TE lowering kind");
    TEST_CHECK(add.spec().lowering_key == "FRelayToTE",
               "fluent registration should derive lowering key");

    TEST_CHECK(kxc::relay::Op::TryGet("definitely_missing_op") == nullptr,
               "TryGet should not create missing operators");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Get("definitely_missing_op"); }),
               "plain missing op lookup should throw");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Get("unknown.namespaced_op"); }),
               "namespaced missing op lookup should also throw");

    kxc::Var data("ffi_data", kxc::TensorType({2}, "float32"));
    kxc::Var other("ffi_other", kxc::TensorType({2}, "float32"));
    kxc::Var indices("ffi_indices", kxc::TensorType({1}, "int64"));
    kxc::Var condition("ffi_condition", kxc::TensorType({2}, "bool"));

    const kxc::relay::Op& gather = kxc::relay::Op::Get("gather");
    TEST_CHECK(gather->name == "gather" &&
                   gather.spec().attrs_type_key == "GatherAttrs",
               "gather should expose its canonical attrs contract");
    const kxc::PackedFunc make_gather =
        kxc::Registry::Global().Get("kxc.relay.op._make.gather");
    TEST_CHECK(make_gather.defined(),
               "gather canonical FFI helper should be registered");
    const kxc::Call ffi_gather =
        kxc::CastTo<kxc::Call>(make_gather(data, indices, -1));
    const auto* ffi_gather_attrs = ffi_gather->attrs.As<kxc::relay::GatherAttrsNode>();
    TEST_CHECK(ffi_gather->args.size() == 2 && ffi_gather->args[0].get() == data.get() &&
                   ffi_gather->args[1].get() == indices.get() && ffi_gather_attrs &&
                   ffi_gather_attrs->axis == -1,
               "gather FFI invocation must preserve argument order and axis attrs");
    const std::string gather_attrs = kxc::relay::SerializeAttrs(
        kxc::relay::GatherAttrs::Create(-1));
    TEST_CHECK(kxc::relay::SerializeAttrs(kxc::relay::Attrs(ffi_gather->attrs)) ==
                   gather_attrs,
               "gather FFI attrs must equal direct canonical attrs");
    TEST_CHECK(Contains(gather_attrs, "GatherAttrsNode") &&
                   Contains(gather_attrs, "axis"),
               "gather attrs serialization should include the canonical axis");

    const kxc::relay::Op& concatenate = kxc::relay::Op::Get("concatenate");
    TEST_CHECK(concatenate->name == "concatenate" &&
                   concatenate.spec().attrs_type_key == "ConcatenateAttrs" &&
                   concatenate.spec().input_arity.num_inputs == 2,
               "concatenate should expose its binary canonical attrs contract");
    const kxc::PackedFunc make_concatenate =
        kxc::Registry::Global().Get("kxc.relay.op._make.concatenate");
    TEST_CHECK(make_concatenate.defined(),
               "concatenate canonical FFI helper should be registered");
    const kxc::Call ffi_concatenate =
        kxc::CastTo<kxc::Call>(make_concatenate(data, other, -1));
    const auto* ffi_concatenate_attrs =
        ffi_concatenate->attrs.As<kxc::relay::ConcatenateAttrsNode>();
    TEST_CHECK(ffi_concatenate->args.size() == 2 && ffi_concatenate_attrs &&
                   ffi_concatenate_attrs->axis == -1,
               "concatenate FFI invocation must preserve binary axis attrs");
    const std::string concatenate_attrs = kxc::relay::SerializeAttrs(
        kxc::relay::ConcatenateAttrs::Create(-1));
    TEST_CHECK(kxc::relay::SerializeAttrs(
                   kxc::relay::Attrs(ffi_concatenate->attrs)) == concatenate_attrs,
               "concatenate FFI attrs must equal direct canonical attrs");
    TEST_CHECK(Contains(concatenate_attrs, "ConcatenateAttrsNode") &&
                   Contains(concatenate_attrs, "axis"),
               "concatenate attrs serialization should include the canonical axis");

    const kxc::relay::Op& slice = kxc::relay::Op::Get("slice");
    TEST_CHECK(slice->name == "slice" && slice.spec().attrs_type_key == "SliceAttrs" &&
                   slice.spec().input_arity.num_inputs == 1,
               "slice should expose its unary canonical attrs contract");
    const kxc::PackedFunc make_slice =
        kxc::Registry::Global().Get("kxc.relay.op._make.slice");
    TEST_CHECK(make_slice.defined(),
               "slice canonical FFI helper should be registered");
    const kxc::Array<int64_t> starts{0};
    const kxc::Array<int64_t> ends{2};
    const kxc::Array<int64_t> axes{0};
    const kxc::Array<int64_t> steps{1};
    const kxc::Call ffi_slice = kxc::CastTo<kxc::Call>(
        make_slice(data, starts, ends, axes, steps));
    const auto* ffi_slice_attrs = ffi_slice->attrs.As<kxc::relay::SliceAttrsNode>();
    TEST_CHECK(ffi_slice->args.size() == 1 && ffi_slice_attrs &&
                   ffi_slice_attrs->starts[0] == 0 && ffi_slice_attrs->ends[0] == 2 &&
                   ffi_slice_attrs->axes[0] == 0 && ffi_slice_attrs->steps[0] == 1,
               "slice FFI invocation must preserve canonical int64 control attrs");
    const std::string slice_attrs = kxc::relay::SerializeAttrs(
        kxc::relay::SliceAttrs::Create({0}, {2}, {0}, {1}));
    TEST_CHECK(kxc::relay::SerializeAttrs(kxc::relay::Attrs(ffi_slice->attrs)) ==
                   slice_attrs,
               "slice FFI attrs must equal direct canonical attrs");
    TEST_CHECK(Contains(slice_attrs, "SliceAttrsNode") && Contains(slice_attrs, "starts") &&
                   Contains(slice_attrs, "steps"),
               "slice attrs serialization should include canonical arrays");

    const kxc::relay::Op& layer_norm = kxc::relay::Op::Get("nn_layer_norm");
    TEST_CHECK(layer_norm->name == "nn_layer_norm" &&
                   layer_norm.spec().attrs_type_key == "LayerNormAttrs" &&
                   layer_norm.spec().input_arity.num_inputs == 3,
               "LayerNorm should expose its three-input canonical attrs contract");
    const kxc::PackedFunc make_layer_norm =
        kxc::Registry::Global().Get("kxc.relay.op._make.nn_layer_norm");
    TEST_CHECK(make_layer_norm.defined(),
               "LayerNorm canonical FFI helper should be registered");
    const kxc::Call ffi_layer_norm = kxc::CastTo<kxc::Call>(make_layer_norm(
        data, other, other, -1, 0.125, std::string("float64")));
    const auto* ffi_layer_norm_attrs =
        ffi_layer_norm->attrs.As<kxc::relay::LayerNormAttrsNode>();
    TEST_CHECK(ffi_layer_norm->args.size() == 3 && ffi_layer_norm_attrs &&
                   ffi_layer_norm_attrs->axis == -1 &&
                   ffi_layer_norm_attrs->epsilon == 0.125f &&
                   ffi_layer_norm_attrs->accumulation_dtype == "float64",
               "LayerNorm FFI invocation must preserve stable argument and attrs ABI");
    const std::string layer_norm_attrs = kxc::relay::SerializeAttrs(
        kxc::relay::LayerNormAttrs::Create(-1, 0.125f, "float64"));
    TEST_CHECK(kxc::relay::SerializeAttrs(
                   kxc::relay::Attrs(ffi_layer_norm->attrs)) == layer_norm_attrs,
               "LayerNorm FFI attrs must equal direct canonical attrs");
    TEST_CHECK(Contains(layer_norm_attrs, "LayerNormAttrsNode") &&
                   Contains(layer_norm_attrs, "axis") &&
                   Contains(layer_norm_attrs, "epsilon") &&
                   Contains(layer_norm_attrs, "accumulation_dtype"),
               "LayerNorm attrs serialization should include all canonical fields");

    const kxc::relay::Op& where = kxc::relay::Op::Get("where");
    TEST_CHECK(where->name == "where" && where.spec().attrs_type_key.empty() &&
                   where.spec().input_arity.num_inputs == 3,
               "where should expose its fieldless three-input canonical contract");
    const kxc::PackedFunc make_where =
        kxc::Registry::Global().Get("kxc.relay.op._make.where");
    TEST_CHECK(make_where.defined(),
               "where canonical FFI helper should be registered");
    const kxc::Call ffi_where =
        kxc::CastTo<kxc::Call>(make_where(condition, data, other));
    TEST_CHECK(ffi_where->args.size() == 3 &&
                   ffi_where->args[0].get() == condition.get() &&
                   ffi_where->args[1].get() == data.get() &&
                   ffi_where->args[2].get() == other.get() &&
                   !ffi_where->attrs.defined(),
               "Where FFI invocation must preserve fieldless attrs and argument order");

    const kxc::relay::Op& copy = kxc::relay::Op::Get("device.copy");
    TEST_CHECK(copy.has_spec(), "explicitly registered control op should expose metadata");
    TEST_CHECK(copy.spec().effect == kxc::relay::OperatorEffectKind::kDeviceCommunication,
               "device external op should be marked as device communication");
    TEST_CHECK(copy.spec().lowering_kind == kxc::relay::OperatorLoweringKind::kExecPlan,
               "device external op should be routed through exec_plan lowering");

    kxc::relay::CheckOperatorRegistry();
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::Op::Register("kxc.test.registry.after_freeze");
               }),
               "operator registration after registry freeze should throw");
    std::vector<kxc::relay::OperatorSpec> specs = kxc::relay::ListOperatorSpecs();
    TEST_CHECK(!specs.empty(), "registry should list operator specs");

    bool found_add = false;
    for (const auto& spec : specs) {
        if (spec.name == "add") {
            found_add = true;
            std::string serialized = kxc::relay::SerializeOperatorSpec(spec);
            TEST_CHECK(Contains(serialized, "name=add;schema_version=1;category=tensor.math"),
                       "operator spec serialization should be stable and ordered");
            TEST_CHECK(Contains(serialized, "type_relation_key=FInferType"),
                       "operator spec serialization should include type key");
            TEST_CHECK(Contains(serialized, "lowering_kind=single;lowering_key=FRelayToTE"),
                       "operator spec serialization should include lowering key");
        }
    }
    TEST_CHECK(found_add, "ListOperatorSpecs should include add");
    return true;
}

bool TestRelayOperatorDuplicateRegistrationFailure() {
    kxc::relay::OperatorSpec spec;
    spec.name = "kxc.test.registry.duplicate_op";
    spec.category = "test";
    spec.input_arity.num_inputs = 0;
    spec.output_arity = 0;
    kxc::relay::Op::Register(spec);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::Op::Register(spec); }),
               "duplicate Relay op registration should throw");
    return true;
}

}  // namespace

int main() {
    if (!TestSetGetAndDuplicateFailure()) return EXIT_FAILURE;
    if (!TestRelayOperatorDuplicateRegistrationFailure()) return EXIT_FAILURE;
    if (!TestRelayOperatorRegistryLookupAndSpecs()) return EXIT_FAILURE;
    std::cout << "All registry tests passed.\n";
    return EXIT_SUCCESS;
}
