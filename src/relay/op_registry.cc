/*! \file src/relay/op_registry.cc
 * \brief Relay operator registry, spec validation, and stable spec serialization.
 */

#include "kxc/relay/op.h"
#include "kxc/ffi/registration.h"
#include "generated/relay_op_contract.inc"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kxc {
namespace relay {

namespace {

void ValidateSpecFields(const OperatorSpec& spec) {
    if (spec.name.empty()) {
        throw std::runtime_error("OperatorSpec name must not be empty");
    }
    if (spec.schema_version <= 0) {
        throw std::runtime_error("OperatorSpec schema_version must be positive for op: " +
                                 spec.name);
    }
    if (spec.category.empty()) {
        throw std::runtime_error("OperatorSpec category is missing for op: " + spec.name);
    }
    if (spec.input_arity.num_inputs < -1) {
        throw std::runtime_error("OperatorSpec num_inputs is invalid for op: " + spec.name);
    }
    if (spec.input_arity.num_inputs == -1) {
        if (spec.input_arity.min_inputs < 0 || spec.input_arity.max_inputs < 0 ||
            spec.input_arity.min_inputs > spec.input_arity.max_inputs) {
            throw std::runtime_error("OperatorSpec variable arity range is invalid for op: " +
                                     spec.name);
        }
    }
    if (spec.output_arity < 0) {
        throw std::runtime_error("OperatorSpec output_arity is invalid for op: " + spec.name);
    }
    if (spec.effect == OperatorEffectKind::kPure && !spec.deterministic) {
        throw std::runtime_error("pure OperatorSpec must be deterministic for op: " +
                                 spec.name);
    }
    if (spec.lowering_kind == OperatorLoweringKind::kSingleTE && spec.lowering_key.empty()) {
        throw std::runtime_error("OperatorSpec single TE lowering key is missing for op: " +
                                 spec.name);
    }
    if (spec.lowering_kind == OperatorLoweringKind::kMultiTE && spec.lowering_key.empty()) {
        throw std::runtime_error("OperatorSpec multi TE lowering key is missing for op: " +
                                 spec.name);
    }
    if (spec.lowering_kind != OperatorLoweringKind::kNone && spec.type_relation_key.empty()) {
        throw std::runtime_error("OperatorSpec type relation key is missing for op: " +
                                 spec.name);
    }
}

void ValidateRegisteredSpec(const OpNode* node) {
    if (!node) {
        throw std::runtime_error("operator registry contains a null op node");
    }
    if (!node->has_spec) {
        throw std::runtime_error("operator has no OperatorSpec: " + node->name);
    }
    ValidateSpecFields(node->spec);
    if (node->spec.name != node->name) {
        throw std::runtime_error("OperatorSpec name mismatch for op: " + node->name);
    }
    if (node->num_inputs != -1 && node->spec.input_arity.num_inputs != node->num_inputs) {
        throw std::runtime_error("OperatorSpec arity conflicts with legacy num_inputs for op: " +
                                 node->name);
    }
}

class OpRegistry {
public:
    static OpRegistry* Global() {
        static OpRegistry inst;
        return &inst;
    }

    Op Register(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frozen_) {
            throw std::runtime_error("Relay operator registry is frozen");
        }
        if (op_map_.count(name) != 0) {
            throw std::runtime_error("Relay op is already registered: " + name);
        }
        Op op(name);
        OpNode* node = const_cast<OpNode*>(op.operator->());
        node->spec = op_contract_generated::Spec(name);
        node->num_inputs = node->spec.input_arity.num_inputs;
        node->arguments = node->spec.arguments;
        node->has_spec = true;
        ValidateRegisteredSpec(node);
        op_map_.insert({name, op});
        return op_map_.at(name);
    }

    Op Register(OperatorSpec spec) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frozen_) {
            throw std::runtime_error("Relay operator registry is frozen");
        }
        if (spec.name.empty()) {
            throw std::runtime_error("OperatorSpec name must not be empty");
        }
        if (op_map_.count(spec.name) != 0) {
            throw std::runtime_error("Relay op is already registered: " + spec.name);
        }
        Op op(spec.name);
        OpNode* node = const_cast<OpNode*>(op.operator->());
        node->num_inputs = spec.input_arity.num_inputs;
        node->arguments = spec.arguments;
        node->spec = std::move(spec);
        node->has_spec = true;
        ValidateRegisteredSpec(node);
        op_map_.insert({node->name, op});
        return op_map_.at(node->name);
    }

    const Op* TryGet(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = op_map_.find(name);
        return it == op_map_.end() ? nullptr : &it->second;
    }

    const Op& Get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = op_map_.find(name);
        if (it == op_map_.end()) {
            throw std::runtime_error("Relay op is not registered: " + name);
        }
        return it->second;
    }

    std::vector<OperatorSpec> ListSpecs() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<OperatorSpec> specs;
        specs.reserve(op_map_.size());
        for (const auto& kv : op_map_) {
            specs.push_back(kv.second->spec);
        }
        std::sort(specs.begin(), specs.end(),
                  [](const OperatorSpec& lhs, const OperatorSpec& rhs) {
                      return lhs.name < rhs.name;
                  });
        return specs;
    }

    void Check() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : op_map_) {
            ValidateRegisteredSpec(kv.second.operator->());
        }
        frozen_ = true;
    }

private:
    std::unordered_map<std::string, Op> op_map_;
    std::mutex mutex_;
    bool frozen_{false};
};

}  // namespace

const char* OperatorEffectKindToString(OperatorEffectKind effect) {
    switch (effect) {
    case OperatorEffectKind::kPure:
        return "pure";
    case OperatorEffectKind::kStateful:
        return "stateful";
    case OperatorEffectKind::kDeviceCommunication:
        return "device_communication";
    }
    return "unknown";
}

const char* OperatorLoweringKindToString(OperatorLoweringKind kind) {
    switch (kind) {
    case OperatorLoweringKind::kNone:
        return "none";
    case OperatorLoweringKind::kSingleTE:
        return "single";
    case OperatorLoweringKind::kMultiTE:
        return "multi";
    case OperatorLoweringKind::kExecPlan:
        return "exec_plan";
    }
    return "unknown";
}

std::string SerializeOperatorSpec(const OperatorSpec& spec) {
    std::ostringstream os;
    os << "name=" << spec.name
       << ";schema_version=" << spec.schema_version
       << ";category=" << spec.category
       << ";num_inputs=" << spec.input_arity.num_inputs
       << ";min_inputs=" << spec.input_arity.min_inputs
       << ";max_inputs=" << spec.input_arity.max_inputs
       << ";attrs=" << spec.attrs_type_key
       << ";output_arity=" << spec.output_arity
       << ";type_relation_key=" << spec.type_relation_key
       << ";effect=" << OperatorEffectKindToString(spec.effect)
       << ";deterministic=" << (spec.deterministic ? "true" : "false")
       << ";alias=" << spec.alias_contract
       << ";lowering_kind=" << OperatorLoweringKindToString(spec.lowering_kind)
       << ";lowering_key=" << spec.lowering_key
       << ";arguments=[";
    for (size_t i = 0; i < spec.arguments.size(); ++i) {
        const ArgumentInfo& arg = spec.arguments[i];
        if (i != 0) {
            os << ",";
        }
        os << arg.name << ":" << arg.type << ":"
           << (arg.is_optional ? "optional" : "required") << ":" << arg.default_value_desc;
    }
    os << "]";
    return os.str();
}

std::vector<OperatorSpec> ListOperatorSpecs() {
    RegisterBuiltins();
    return OpRegistry::Global()->ListSpecs();
}

void CheckOperatorRegistry() {
    RegisterBuiltins();
    OpRegistry::Global()->Check();
}

void ValidateOperatorSpec(const OperatorSpec& spec) {
    ValidateSpecFields(spec);
}

const Op* Op::TryGet(const std::string& name) {
    RegisterBuiltins();
    return OpRegistry::Global()->TryGet(name);
}

Op Op::Register(const std::string& name) {
    return OpRegistry::Global()->Register(name);
}

Op Op::Register(OperatorSpec spec) {
    return OpRegistry::Global()->Register(std::move(spec));
}

const Op& Op::Get(const std::string& name) {
    RegisterBuiltins();
    return OpRegistry::Global()->Get(name);
}

}  // namespace relay
}  // namespace kxc
