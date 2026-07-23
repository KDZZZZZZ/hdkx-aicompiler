/*! \file src/compiler/capability.cc
 * \brief Implements the compiler's fail-closed executable dialect check.
 */

#include "kxc/compiler/capability.h"

#include <algorithm>
#include <any>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "internal/execution_contract.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

namespace kxc::api {
namespace {

Device Placement(const Expr& expr) {
    const auto* relay_node = dynamic_cast<const RelayNode*>(expr.get());
    if (!relay_node || !relay_node->virtual_device_.defined()) return Device::CPU();
    return relay_node->virtual_device_->device;
}

class Verifier final {
public:
    Verifier(const CapabilityRequest& request, bool prove_execution)
        : request_(request), prove_execution_(prove_execution) {
        result_.requested_mode = request.requested_mode;
        result_.pipeline_fingerprint = request.pipeline_fingerprint;
        if (request.target.defined() && request.target.As<TargetNode>()) {
            result_.target_identity =
                request.target->kind + ":" +
                std::to_string(static_cast<int>(request.target->device_type)) +
                ":" + std::to_string(request.target->device_id);
        } else {
            result_.target_identity = "undefined";
        }
        result_.normalized_requirements = {
            "relay.static_exact.v1", "relay.registered_operator_calls",
            "relay.tensor_or_flat_tuple_values", "compiler.per_unit_lowering",
            "compiler.normalized_pipeline.v2", "compiler.target_schedule",
            "compiler.backend_codegen"};
    }

    CapabilityResult Run() {
        VerifyRequest();
        if (!has_structural_issue_ && prove_execution_) ProbeExecutablePipeline();
        if (has_structural_issue_) {
            result_.status = CapabilityStatus::kUnsupported;
        } else if (!prove_execution_ || !result_.issues.empty()) {
            result_.status = CapabilityStatus::kEligibleButNotExecutable;
        } else {
            result_.status = CapabilityStatus::kExecutable;
        }
        result_.supported = result_.status == CapabilityStatus::kExecutable;
        if (!result_.supported && !result_.issues.empty()) {
            result_.diagnostic_locator = result_.issues.front().diagnostic_locator;
        }
        return result_;
    }

private:
    const CapabilityRequest& request_;
    CapabilityResult result_;
    std::unordered_set<const Object*> visited_;
    std::unordered_set<const Object*> bound_vars_;
    bool has_structural_issue_{false};
    bool prove_execution_{false};

    void AddIssue(std::string locator, std::string node_kind,
                  std::string capability, std::string detail,
                  bool structural = true) {
        has_structural_issue_ = has_structural_issue_ || structural;
        if (std::find(result_.missing_capabilities.begin(),
                      result_.missing_capabilities.end(), capability) ==
            result_.missing_capabilities.end()) {
            result_.missing_capabilities.push_back(capability);
        }
        result_.issues.push_back(CapabilityIssue{
            std::move(locator), std::move(node_kind), std::move(capability),
            std::move(detail)});
    }

    void VerifyRequest() {
        const std::string root = request_.graph_locator.empty()
                                     ? "graph"
                                     : request_.graph_locator;
        if (request_.requested_mode != CapabilityMode::kStaticExact) {
            AddIssue(root, "Function", "execution_mode",
                     std::string("requested mode '") +
                         ToString(request_.requested_mode) +
                         "' is not executable; only static_exact is supported");
        }
        if (!request_.function.defined() ||
            !request_.function.As<FunctionNode>() ||
            !request_.function->body.defined()) {
            AddIssue(root, "Function", "defined_function",
                     "validated relay must have a body");
            return;
        }
        for (size_t i = 0; i < request_.function->params.size(); ++i) {
            const Var& parameter = request_.function->params[i];
            const std::string locator = root + "/param[" +
                                        std::to_string(i) + "]";
            if (!parameter.defined() || !parameter.As<VarNode>()) {
                AddIssue(locator, "Var", "typed_parameter",
                         "function parameter is undefined or invalid");
                continue;
            }
            bound_vars_.insert(parameter.get());
            if (!parameter->type_annotation.As<TensorTypeNode>()) {
                AddIssue(locator, "Var", "tensor_parameter",
                         "per-unit compilation requires TensorType function parameters");
            } else {
                CheckType(parameter->type_annotation, locator, "Var", true);
            }
        }
        Visit(request_.function->body, root + "/body");
        VerifyTarget(root);
    }

    void VerifyTarget(const std::string& locator) {
        if (!request_.target.defined() || !request_.target.As<TargetNode>()) {
            AddIssue(locator, "Function", "target", "target is undefined or invalid");
            return;
        }
        const TargetNode* target = request_.target.operator->();
        if (target->kind == "llvm" && target->device_type == kCPU) {
#if !KXC_USE_LLVM
            AddIssue(locator, "Target", "backend.llvm",
                     "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON",
                     false);
#endif
            if (target->attrs.exists == 0) {
                AddIssue(locator, "Target", "target_snapshot.exists",
                         "LLVM target snapshot reports no CPU device", false);
            }
            return;
        }
        if (target->kind != "cuda" || target->device_type != kCUDA) {
            AddIssue(locator, "Target", "target",
                     "target kind and device type have no executable backend");
            return;
        }
#if !KXC_USE_CUDA
        AddIssue(locator, "Target", "backend.cuda",
                 "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON",
                 false);
#endif
        if (target->attrs.exists == 0) {
            AddIssue(locator, "Target", "cuda_device_exists",
                     "CUDA target snapshot reports no available device", false);
        }
        if (target->attrs.compute_version_major <= 0 ||
            target->attrs.compute_version_minor < 0 ||
            target->attrs.compute_version.empty() ||
            target->attrs.compute_version == "0.0") {
            AddIssue(locator, "Target", "cuda_compute_capability",
                     "CUDA target snapshot lacks a valid compute capability", false);
        }
        if (target->attrs.max_threads_per_block <= 0 ||
            target->attrs.warp_size <= 0 ||
            target->attrs.max_threads_per_multiprocessor <= 0 ||
            target->attrs.multi_processor_count <= 0 ||
            target->attrs.max_shared_memory_per_block < 0) {
            AddIssue(locator, "Target", "cuda_launch_snapshot",
                     "CUDA target snapshot lacks valid launch-limit fields", false);
        }
    }

    void ProbeExecutablePipeline() {
        const std::string locator = request_.graph_locator.empty()
                                        ? "graph"
                                        : request_.graph_locator;
        try {
            const CompileConfig config = CompileConfig::Create(
                request_.target, request_.opt_level);
            const internal::CompilerExecutionContract contract =
                internal::ResolveCompilerExecutionContract(config);
            if (!request_.pipeline_fingerprint.empty() &&
                request_.pipeline_fingerprint != contract.fingerprint) {
                AddIssue(locator, "Pipeline", "pipeline_identity",
                         "requested pipeline fingerprint does not match the normalized execution plan",
                         false);
                return;
            }
            result_.pipeline_fingerprint = contract.fingerprint;
            internal::ProbeCompilerExecution(request_.function, config,
                                             contract);
        } catch (const std::exception& error) {
            const std::string detail = error.what();
            std::string capability = "backend_executable";
            std::string node_kind = "Target";
            if (detail.find("optimize_relay") != std::string::npos ||
                detail.find("infer_type") != std::string::npos) {
                capability = "relay_pipeline";
                node_kind = "Function";
            } else if (detail.find("stage 'lower'") != std::string::npos ||
                       detail.find("lowering") != std::string::npos) {
                capability = "per_unit_lowering";
                node_kind = "Function";
            } else if (detail.find("optimize_tir") != std::string::npos ||
                       detail.find("bind_cuda_threads") != std::string::npos ||
                       detail.find("CUDA launch") != std::string::npos) {
                capability = "target_schedule";
                node_kind = "TIR";
            } else if (detail.find("build_signature") != std::string::npos) {
                capability = "kernel_abi";
                node_kind = "TIR";
            }
            AddIssue(locator, node_kind, capability, detail, false);
        }
    }

    void CheckType(const Type& type, const std::string& locator,
                   const std::string& node_kind, bool required) {
        if (!type.defined()) {
            if (required) {
                AddIssue(locator, node_kind, "checked_type",
                         "static-exact execution requires a complete type");
            }
            return;
        }
        if (const auto* tensor = type.As<TensorTypeNode>()) {
            if (tensor->dtype.empty()) {
                AddIssue(locator, node_kind, "tensor_dtype",
                         "tensor dtype must be explicit");
            }
            for (size_t i = 0; i < tensor->shape.size(); ++i) {
                if (tensor->shape[i] < 0) {
                    AddIssue(locator + "/dim[" + std::to_string(i) + "]",
                             node_kind, "static_exact_shape",
                             "symbolic or legacy -1 dimensions are not static-exact capability");
                }
            }
            return;
        }
        if (const auto* tuple = type.As<TupleTypeNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                if (!tuple->fields[i].As<TensorTypeNode>()) {
                    AddIssue(locator + "/field[" + std::to_string(i) + "]",
                             node_kind, "flat_tensor_tuple",
                             "only a flat tuple of tensor leaves is executable");
                } else {
                    CheckType(tuple->fields[i],
                              locator + "/field[" + std::to_string(i) + "]",
                              node_kind, true);
                }
            }
            return;
        }
        AddIssue(locator, node_kind, "tensor_value_type",
                 "only TensorType and flat TupleType values are executable");
    }

    void Visit(const Expr& expr, const std::string& locator) {
        if (!expr.defined()) {
            AddIssue(locator, "Expr", "defined_expression",
                     "expression is undefined");
            return;
        }
        if (!visited_.insert(expr.get()).second) return;
        if (const auto* var = expr.As<VarNode>()) {
            if (bound_vars_.count(expr.get()) == 0) {
                AddIssue(locator, "Var", "bound_variable",
                         "free or unbound variables are not executable");
            }
            CheckType(expr.checked_type().defined() ? expr.checked_type()
                                                    : var->type_annotation,
                      locator, "Var", request_.require_checked_types);
            return;
        }
        if (const auto* constant = expr.As<ConstantNode>()) {
            if (!constant->data.defined()) {
                AddIssue(locator, "Constant", "constant_payload",
                         "constant payload is undefined");
            }
            CheckType(expr.checked_type(), locator, "Constant",
                      request_.require_checked_types);
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            VisitCall(expr, call, locator);
            return;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                Visit(tuple->fields[i], locator + "/field[" +
                                            std::to_string(i) + "]");
            }
            CheckType(expr.checked_type(), locator, "Tuple",
                      request_.require_checked_types);
            return;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            Visit(get_item->tuple, locator + "/tuple");
            if (get_item->index < 0) {
                AddIssue(locator, "TupleGetItem", "tuple_index",
                         "tuple field index must be non-negative");
            }
            CheckType(expr.checked_type(), locator, "TupleGetItem",
                      request_.require_checked_types);
            return;
        }
        if (const auto* while_node = expr.As<WhileNode>()) {
            if (!request_.allow_while) {
                AddIssue(locator, "While", "control_flow.loop",
                         "While is representable Relay IR but disabled for this executable boundary");
                return;
            }
            if (while_node->max_trip_count < 0 || !while_node->loop_var.defined()) {
                AddIssue(locator, "While", "bounded_loop",
                         "While requires a defined binder and non-negative max_trip_count");
                return;
            }
            Visit(while_node->initial_state, locator + "/initial_state");
            const Type binder_type = while_node->loop_var.checked_type().defined()
                                         ? while_node->loop_var.checked_type()
                                         : while_node->loop_var->type_annotation;
            CheckType(binder_type, locator + "/loop_var", "Var", true);
            if (while_node->initial_state.checked_type().defined() && binder_type.defined() &&
                !TypeEqual(while_node->initial_state.checked_type(), binder_type)) {
                AddIssue(locator + "/loop_var", "Var", "typed_loop_binding",
                         "loop binder type does not match initial state");
            }
            const Device state_device = Placement(while_node->initial_state);
            if (Placement(Expr(ObjectRef(while_node->loop_var))) != state_device ||
                Placement(while_node->body) != state_device ||
                Placement(expr) != state_device) {
                AddIssue(locator, "While", "exact_loop_state_placement",
                         "While initial state, binder, body, and result must have one exact device placement");
            }
            if (Placement(while_node->condition) != Device::CPU()) {
                AddIssue(locator + "/condition", "While",
                         "cpu_loop_condition_placement",
                         "While condition must be placed on CPU:0");
            }
            const bool already_bound = bound_vars_.count(while_node->loop_var.get()) != 0;
            bound_vars_.insert(while_node->loop_var.get());
            Visit(while_node->condition, locator + "/condition");
            Visit(while_node->body, locator + "/body");
            if (!already_bound) bound_vars_.erase(while_node->loop_var.get());
            const auto* predicate = while_node->condition.checked_type().As<TensorTypeNode>();
            if (!predicate || predicate->dtype != "bool" || !predicate->shape.empty()) {
                AddIssue(locator + "/condition", "While", "scalar_bool_loop_predicate",
                         "While condition must be a scalar bool TensorType");
            }
            if (!TypeEqual(while_node->initial_state.checked_type(),
                           while_node->body.checked_type()) ||
                !TypeEqual(expr.checked_type(), while_node->initial_state.checked_type())) {
                AddIssue(locator, "While", "exact_loop_state_type",
                         "initial state, body, and result must exactly match");
            }
            return;
        }
        if (expr.As<IfNode>()) {
            AddIssue(locator, "If", "control_flow.if",
                     "If is representable Relay IR but not executable by the static plan");
            return;
        }
        if (const auto* let = expr.As<LetNode>()) {
            Visit(let->value, locator + "/value");
            if (!let->var.defined()) {
                AddIssue(locator + "/var", "Var", "lexical_let_binding",
                         "Let binder is undefined");
                return;
            }
            const Type binder_type = let->var.checked_type().defined()
                                         ? let->var.checked_type()
                                         : let->var->type_annotation;
            CheckType(binder_type, locator + "/var", "Var", true);
            if (let->value.checked_type().defined() && binder_type.defined() &&
                !TypeEqual(let->value.checked_type(), binder_type)) {
                AddIssue(locator + "/var", "Var", "typed_let_binding",
                         "Let binder type does not match its value");
            }
            const bool already_bound = bound_vars_.count(let->var.get()) != 0;
            bound_vars_.insert(let->var.get());
            Visit(let->body, locator + "/body");
            if (!already_bound) bound_vars_.erase(let->var.get());
            CheckType(expr.checked_type(), locator, "Let",
                      request_.require_checked_types);
            return;
        }
        if (expr.As<FunctionNode>()) {
            AddIssue(locator, "Function", "function_value",
                     "nested function values are not executable");
            return;
        }
        AddIssue(locator, std::string(expr.get()->GetTypeKey()), "relay_node_kind",
                 "Relay node kind is outside the executable dialect");
    }

    void VisitCall(const Expr& ref, const CallNode* call,
                   const std::string& locator) {
        const auto* op = call->op.As<relay::OpNode>();
        const std::string op_name = op ? op->name : "<non-op>";
        const std::string call_locator = locator + "/call(" + op_name + ")";
        if (!op || !op->has_spec) {
            AddIssue(call_locator, "Call", "registered_operator",
                     "Call must reference an operator with an explicit OperatorSpec");
        } else {
            VerifyOperator(call, op, call_locator);
        }
        for (size_t i = 0; i < call->args.size(); ++i) {
            Visit(call->args[i], call_locator + "/arg[" +
                                     std::to_string(i) + "]");
        }
        CheckType(ref.checked_type(), call_locator, "Call",
                  request_.require_checked_types);
    }

    void VerifyOperator(const CallNode* call, const relay::OpNode* op,
                        const std::string& locator) {
        try {
            relay::ValidateOperatorSpec(op->spec);
        } catch (const std::exception& error) {
            AddIssue(locator, "Call", "operator_schema", error.what());
            return;
        }
        const relay::OperatorSpec& spec = op->spec;
        if (spec.input_arity.num_inputs >= 0) {
            if (call->args.size() !=
                static_cast<size_t>(spec.input_arity.num_inputs)) {
                AddIssue(locator, "Call", "operator_arity",
                         "Call argument count does not match OperatorSpec");
            }
        } else if (spec.input_arity.min_inputs < 0 ||
                   spec.input_arity.max_inputs < spec.input_arity.min_inputs ||
                   call->args.size() <
                       static_cast<size_t>(spec.input_arity.min_inputs) ||
                   call->args.size() >
                       static_cast<size_t>(spec.input_arity.max_inputs)) {
            AddIssue(locator, "Call", "operator_arity",
                     "Call argument count is outside OperatorSpec range");
        }
        if (spec.lowering_kind != relay::OperatorLoweringKind::kSingleTE &&
            spec.lowering_kind != relay::OperatorLoweringKind::kMultiTE) {
            AddIssue(locator, "Call", "ordinary_compute_lowering",
                     "only single-TE or multi-TE operators enter the per-unit compiler");
            return;
        }
        const auto relation = op->attrs.find(spec.type_relation_key);
        if (relation == op->attrs.end() ||
            std::any_cast<relay::FInferType>(&relation->second) == nullptr) {
            AddIssue(locator, "Call", "type_relation_binding",
                     "OperatorSpec type relation has no matching implementation binding");
        }
        const auto lowering = op->attrs.find(spec.lowering_key);
        const bool single_ok =
            spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE &&
            lowering != op->attrs.end() &&
            std::any_cast<relay::FRelayToTE>(&lowering->second) != nullptr;
        const bool multi_ok =
            spec.lowering_kind == relay::OperatorLoweringKind::kMultiTE &&
            lowering != op->attrs.end() &&
            std::any_cast<relay::FRelayToTEMulti>(&lowering->second) != nullptr;
        if (!single_ok && !multi_ok) {
            AddIssue(locator, "Call", "lowering_binding",
                     "OperatorSpec lowering has no matching TE implementation binding");
        }
        if (call->attrs.defined()) {
            if (spec.attrs_type_key.empty()) {
                AddIssue(locator, "Call", "operator_attrs",
                         "Call supplies attrs outside its OperatorSpec");
            } else {
                const std::string actual(call->attrs.get()->GetTypeKey());
                const std::string expected_node = spec.attrs_type_key + "Node";
                if (actual != spec.attrs_type_key && actual != expected_node) {
                    AddIssue(locator, "Call", "operator_attrs",
                             "Call attrs runtime type does not match OperatorSpec");
                }
            }
        }
    }
};

}  // namespace

const char* ToString(CapabilityBoundary boundary) {
    switch (boundary) {
        case CapabilityBoundary::kCompilerEntry: return "compiler_entry";
        case CapabilityBoundary::kPostGraphPass: return "post_graph_pass";
        case CapabilityBoundary::kPrePartition: return "pre_partition";
    }
    return "unknown_boundary";
}

const char* ToString(CapabilityMode mode) {
    switch (mode) {
        case CapabilityMode::kStaticExact: return "static_exact";
        case CapabilityMode::kShapeSpecialization: return "shape_specialization";
        case CapabilityMode::kControlFlow: return "control_flow";
        case CapabilityMode::kRegion: return "region";
    }
    return "unknown_mode";
}

const char* ToString(CapabilityStatus status) {
    switch (status) {
        case CapabilityStatus::kUnsupported: return "unsupported";
        case CapabilityStatus::kEligibleButNotExecutable:
            return "eligible_but_not_executable";
        case CapabilityStatus::kExecutable: return "executable";
    }
    return "unknown_status";
}

std::string CapabilityResult::Diagnostic() const {
    if (supported) return "supported";
    std::ostringstream stream;
    stream << "executable capability rejected (status=" << ToString(status)
           << ", mode=" << ToString(requested_mode)
           << ", target=" << target_identity << ", pipeline="
           << (pipeline_fingerprint.empty() ? "<none>" : pipeline_fingerprint)
           << ")";
    for (const CapabilityIssue& issue : issues) {
        stream << "\n- " << issue.diagnostic_locator << " ["
               << issue.relay_node_kind << "] missing "
               << issue.missing_capability << ": " << issue.detail;
    }
    return stream.str();
}

CapabilityResult CapabilityVerifier::Verify(const CapabilityRequest& request) {
    return Verifier(request, true).Run();
}

void CapabilityVerifier::RequireEligible(const CapabilityRequest& request) {
    const CapabilityResult result = Verifier(request, false).Run();
    if (result.supported != (result.status == CapabilityStatus::kExecutable)) {
        throw std::logic_error("CapabilityVerifier produced inconsistent status");
    }
    if (result.status == CapabilityStatus::kUnsupported) {
        throw std::invalid_argument(std::string("CapabilityVerifier[") +
                                    ToString(request.boundary) + "]: " +
                                    result.Diagnostic());
    }
}

void CapabilityVerifier::Require(const CapabilityRequest& request) {
    const CapabilityResult result = Verify(request);
    if (result.supported != (result.status == CapabilityStatus::kExecutable)) {
        throw std::logic_error("CapabilityVerifier produced inconsistent status");
    }
    if (!result.supported) {
        throw std::invalid_argument(std::string("CapabilityVerifier[") +
                                    ToString(request.boundary) + "]: " +
                                    result.Diagnostic());
    }
}

}  // namespace kxc::api
