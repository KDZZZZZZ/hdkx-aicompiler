/*! \file src/compiler/analysis/primitive_unit.cc
 * \brief Shared construction and identity for primitive units.
 */

#include "../internal/primitive_unit.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "support/canonical.h"

namespace kxc::api::internal {
namespace {

const LogicalValueContract& Value(
    ValueId id, const std::vector<LogicalValueContract>& values,
    const char* context) {
    if (id < 0 || static_cast<std::size_t>(id) >= values.size() ||
        values[static_cast<std::size_t>(id)].id != id) {
        throw std::invalid_argument(std::string(context) +
                                    " references an invalid logical value");
    }
    return values[static_cast<std::size_t>(id)];
}

std::string SanitizeSymbolPart(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    for (const unsigned char ch : name) {
        result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
    }
    return result.empty() ? "op" : result;
}

void AppendCanonicalField(std::string* canonical, const std::string& name,
                          const std::string& value) {
    support::CanonicalBytesEncoder field;
    field.Field(name, value);
    *canonical += std::move(field).Take();
}

UnitSemanticKey BuildSemanticKey(
    const ResolvedRelayCall& call, const Array<ValueId>& arguments,
    const Array<ValueId>& boundary_inputs, const Array<ValueId>& outputs,
    const std::vector<LogicalValueContract>& values) {
    std::string canonical;
    AppendCanonicalField(&canonical, "kind", "unit-semantic-key-v1");
    AppendCanonicalField(&canonical, "operator",
                         relay::SerializeOperatorSpec(call.spec));

    std::unordered_map<ValueId, std::size_t> boundary_index;
    for (std::size_t index = 0; index < boundary_inputs.size(); ++index) {
        const LogicalValueContract& value =
            Value(boundary_inputs[index], values, "PrimitiveUnit input");
        boundary_index.emplace(value.id, index);
        AppendCanonicalField(
            &canonical, "input_role",
            value.origin == LogicalValueOrigin::kConstant ? "constant"
                                                          : "input");
        AppendCanonicalField(&canonical, "input_type",
                             TypeToString(value.checked_type));
    }
    for (const ValueId id : arguments) {
        const auto boundary = boundary_index.find(id);
        if (boundary == boundary_index.end()) {
            throw std::invalid_argument(
                "PrimitiveUnit logical argument is outside its boundary");
        }
        AppendCanonicalField(&canonical, "logical_input",
                             std::to_string(boundary->second));
    }
    for (const ValueId id : outputs) {
        AppendCanonicalField(
            &canonical, "output_type",
            TypeToString(Value(id, values, "PrimitiveUnit output").checked_type));
    }
    AppendCanonicalField(
        &canonical, "attrs",
        call.attrs.defined() ? relay::SerializeAttrs(call.attrs) : "<none>");
    return UnitSemanticKey(std::move(canonical));
}

Array<ValueId> OrderBoundaryInputs(
    const Array<ValueId>& arguments,
    const std::vector<LogicalValueContract>& values) {
    Array<ValueId> unique;
    std::unordered_set<ValueId> seen;
    for (const ValueId id : arguments) {
        (void)Value(id, values, "PrimitiveUnit argument");
        if (seen.insert(id).second) unique.push_back(id);
    }
    Array<ValueId> ordered;
    for (const ValueId id : unique) {
        if (Value(id, values, "PrimitiveUnit input").origin !=
            LogicalValueOrigin::kConstant) {
            ordered.push_back(id);
        }
    }
    for (const ValueId id : unique) {
        if (Value(id, values, "PrimitiveUnit input").origin ==
            LogicalValueOrigin::kConstant) {
            ordered.push_back(id);
        }
    }
    return ordered;
}

bool SameIds(const Array<ValueId>& left, const Array<ValueId>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

UnitSemanticKey RegionSemanticKey(const PrimitiveUnit& unit,
                                  const std::vector<LogicalValueContract>& values) {
    const auto& producer = *unit.producer;
    support::CanonicalBytesEncoder bytes("unit-static-add-sqrt-v1");
    bytes.Field("producer", BuildSemanticKey(producer.call, producer.argument_value_ids,
        unit.boundary_input_value_ids, producer.output_value_ids, values).canonical_bytes());
    bytes.Field("consumer", BuildSemanticKey(unit.call, unit.argument_value_ids,
        producer.output_value_ids, unit.output_value_ids, values).canonical_bytes());
    return UnitSemanticKey(std::move(bytes).Take());
}

bool PlainPure(const ResolvedRelayCall& call, const char* name) {
    return call.spec.name == name && call.spec.effect == relay::OperatorEffectKind::kPure &&
        call.spec.deterministic && call.spec.alias_contract == "none" &&
        call.spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE &&
        call.spec.output_arity == 1 && !call.attrs.defined();
}

}  // namespace

PrimitiveUnit BuildPrimitiveUnit(
    PrimitiveUnitId id, ResolvedRelayCall call,
    Array<ValueId> argument_value_ids, Array<ValueId> output_value_ids,
    const std::vector<LogicalValueContract>& values) {
    if (id < 0 || !call.call.As<CallNode>() || output_value_ids.empty()) {
        throw std::invalid_argument(
            "BuildPrimitiveUnit requires an id, resolved Call, and outputs");
    }
    Array<ValueId> boundary_inputs =
        OrderBoundaryInputs(argument_value_ids, values);
    const Device device =
        Value(output_value_ids[0], values, "PrimitiveUnit output").device;
    PrimitiveUnit unit{
        id,
        String("kxc_unit_" + std::to_string(id) + "_" +
               SanitizeSymbolPart(call.spec.name)),
        std::move(call),
        std::move(argument_value_ids),
        std::move(boundary_inputs),
        std::move(output_value_ids),
        device,
        UnitSemanticKey()};
    unit.semantic_key = BuildSemanticKey(
        unit.call, unit.argument_value_ids, unit.boundary_input_value_ids,
        unit.output_value_ids, values);
    ValidatePrimitiveUnit(unit, values);
    return unit;
}

bool CanFuseAddSqrt(const PrimitiveUnit& producer, const PrimitiveUnit& consumer,
                   const std::vector<LogicalValueContract>& values) {
    if (producer.producer || consumer.producer || !PlainPure(producer.call, "add") ||
        !PlainPure(consumer.call, "sqrt") || producer.argument_value_ids.size() != 2 ||
        producer.output_value_ids.size() != 1 || consumer.argument_value_ids.size() != 1 ||
        consumer.output_value_ids.size() != 1 ||
        consumer.argument_value_ids[0] != producer.output_value_ids[0] ||
        producer.device != consumer.device || producer.device != Device::CPU()) return false;
    const auto& output = Value(consumer.output_value_ids[0], values, "fusion output");
    const auto* type = output.checked_type.As<TensorTypeNode>();
    if (!type || (type->dtype != "float32" && type->dtype != "float64")) return false;
    for (const auto dim : type->shape) if (dim < 0) return false;
    Array<ValueId> required;
    for (const ValueId id : producer.argument_value_ids) required.push_back(id);
    required.push_back(producer.output_value_ids[0]);
    for (const ValueId id : required) {
        const auto& value = Value(id, values, "fusion input");
        if (value.device != output.device || !TypeEqual(value.checked_type, output.checked_type)) return false;
    }
    return true;
}

PrimitiveUnit FuseAddSqrt(const PrimitiveUnit& producer, const PrimitiveUnit& consumer,
                         const std::vector<LogicalValueContract>& values) {
    ValidatePrimitiveUnit(producer, values);
    ValidatePrimitiveUnit(consumer, values);
    if (!CanFuseAddSqrt(producer, consumer, values)) {
        throw std::invalid_argument("PrimitiveUnit fusion requires static pure CPU add -> sqrt with identical tensor types");
    }
    PrimitiveUnit region = consumer;
    region.id = producer.id;
    region.symbol = String("kxc_unit_" + std::to_string(region.id) + "_add_sqrt");
    region.boundary_input_value_ids = producer.boundary_input_value_ids;
    region.producer = PrimitiveUnitProducer{producer.call, producer.argument_value_ids, producer.output_value_ids};
    region.semantic_key = RegionSemanticKey(region, values);
    ValidatePrimitiveUnit(region, values);
    return region;
}

void ValidatePrimitiveUnit(
    const PrimitiveUnit& unit,
    const std::vector<LogicalValueContract>& values) {
    if (unit.id < 0 || std::string(unit.symbol).empty() ||
        !unit.call.call.As<CallNode>() || !unit.semantic_key.defined() ||
        unit.output_value_ids.empty() || !unit.device.defined()) {
        throw std::invalid_argument("PrimitiveUnit has an incomplete contract");
    }
    if (unit.producer) {
        const auto& first = *unit.producer;
        const PrimitiveUnit producer = BuildPrimitiveUnit(unit.id, first.call,
            first.argument_value_ids, first.output_value_ids, values);
        const PrimitiveUnit consumer = BuildPrimitiveUnit(unit.id, unit.call,
            unit.argument_value_ids, unit.output_value_ids, values);
        if (!CanFuseAddSqrt(producer, consumer, values) ||
            unit.semantic_key != RegionSemanticKey(unit, values)) {
            throw std::invalid_argument("PrimitiveUnit region contract or semantic key drifted");
        }
    }
    const Array<ValueId> expected_inputs = OrderBoundaryInputs(
        unit.producer ? unit.producer->argument_value_ids : unit.argument_value_ids, values);
    if (!SameIds(expected_inputs, unit.boundary_input_value_ids)) {
        throw std::invalid_argument(
            "PrimitiveUnit boundary input ordering is not canonical");
    }
    if (unit.output_value_ids.size() != unit.call.output_leaf_types.size()) {
        throw std::invalid_argument(
            "PrimitiveUnit outputs drifted from ResolvedRelayCall");
    }
    for (std::size_t index = 0; index < unit.output_value_ids.size(); ++index) {
        const LogicalValueContract& output =
            Value(unit.output_value_ids[index], values, "PrimitiveUnit output");
        if (output.device != unit.device ||
            !TypeEqual(output.checked_type, unit.call.output_leaf_types[index])) {
            throw std::invalid_argument(
                "PrimitiveUnit output type/device contract mismatch");
        }
    }
    for (const ValueId id : unit.boundary_input_value_ids) {
        if (Value(id, values, "PrimitiveUnit input").device != unit.device) {
            throw std::invalid_argument(
                "PrimitiveUnit requires one explicit input/output device");
        }
    }
}

std::string PrimitiveUnitOperatorIdentity(const PrimitiveUnit& unit) {
    const auto identity = [](const ResolvedRelayCall& call) {
        return call.spec.name + "@v" + std::to_string(call.spec.schema_version);
    };
    return unit.producer ? identity(unit.producer->call) + "+" + identity(unit.call) : identity(unit.call);
}

}  // namespace kxc::api::internal
