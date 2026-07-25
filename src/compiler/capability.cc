/*! \file src/compiler/capability.cc
 * \brief Implements the compiler's fail-closed executable dialect check.
 */

#include "kxc/compiler/capability.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "internal/executable_capability.h"
#include "internal/execution_contract.h"

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

namespace kxc::api {
namespace {

std::string PublicLocator(const std::string& root, const std::string& path) {
    if (path == "function") return root;
    constexpr const char kFunction[] = "function";
    if (path.compare(0, sizeof(kFunction) - 1, kFunction) != 0) return path;
    std::string result = root;
    for (size_t i = sizeof(kFunction) - 1; i < path.size();) {
        if (path.compare(i, 7, ".params") == 0) {
            result += "/param";
            i += 7;
        } else if (path[i] == '.') {
            result += '/';
            ++i;
        } else {
            result += path[i++];
        }
    }
    return result;
}

std::string PublicCapability(const std::string& capability) {
    if (capability == "if") return "control_flow.if";
    if (capability == "registered_ordinary_op_call") return "registered_operator";
    if (capability == "operator_input_arity") return "operator_arity";
    if (capability == "operator_implementation_binding") return "lowering_binding";
    if (capability == "static_tensor_or_tuple_type") return "tensor_value_type";
    if (capability == "defined_typed_relay") return "checked_type";
    if (capability == "well_typed_tuple_get_item") return "tuple_index";
    if (capability == "first_order_relay") return "function_value";
    return capability;
}

class Verifier final {
public:
    Verifier(const CapabilityRequest& request, bool prove_execution)
        : request_(request), prove_execution_(prove_execution) {
        result_.pipeline_fingerprint = request.pipeline_fingerprint;
        if (request.target.defined() && request.target.As<TargetNode>()) {
            result_.target_identity = request.target->kind + ":" +
                std::to_string(static_cast<int>(request.target->device_type)) +
                ":" + std::to_string(request.target->device_id);
        } else {
            result_.target_identity = "undefined";
        }
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
        return result_;
    }

private:
    const CapabilityRequest& request_;
    CapabilityResult result_;
    bool has_structural_issue_{false};
    bool prove_execution_{false};

    void AddIssue(std::string locator, std::string node_kind,
                  std::string capability, std::string detail,
                  bool structural = true) {
        has_structural_issue_ = has_structural_issue_ || structural;
        result_.issues.push_back(CapabilityIssue{
            std::move(locator), std::move(node_kind), std::move(capability),
            std::move(detail)});
    }

    void AddRelayIssues(const std::string& root) {
        Device device;
        if (request_.target.defined() && request_.target.As<TargetNode>()) {
            device = Device(request_.target->device_type, request_.target->device_id);
        }
        internal::ExecutableCapabilityOptions options =
            internal::StaticDataflowExecutableCapabilities(device);
        options.require_checked_types = request_.require_checked_types;
        for (const internal::ExecutableCapabilityIssue& issue :
             internal::CollectExecutableCapabilityIssues(request_.function, options)) {
            AddIssue(PublicLocator(root, issue.path), issue.node_kind,
                     PublicCapability(issue.capability), issue.detail);
        }
    }

    void VerifyRequest() {
        const std::string root = request_.graph_locator.empty() ? "graph" : request_.graph_locator;
        AddRelayIssues(root);
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
                     "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON", false);
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
                 "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON", false);
#endif
        if (target->attrs.exists == 0) {
            AddIssue(locator, "Target", "cuda_device_exists",
                     "CUDA target snapshot reports no available device", false);
        }
        if (target->attrs.compute_version_major <= 0 ||
            target->attrs.compute_version_minor < 0 || target->attrs.compute_version.empty() ||
            target->attrs.compute_version == "0.0") {
            AddIssue(locator, "Target", "cuda_compute_capability",
                     "CUDA target snapshot lacks a valid compute capability", false);
        }
        if (target->attrs.max_threads_per_block <= 0 || target->attrs.warp_size <= 0 ||
            target->attrs.max_threads_per_multiprocessor <= 0 ||
            target->attrs.multi_processor_count <= 0 ||
            target->attrs.max_shared_memory_per_block < 0) {
            AddIssue(locator, "Target", "cuda_launch_snapshot",
                     "CUDA target snapshot lacks valid launch-limit fields", false);
        }
    }

    void ProbeExecutablePipeline() {
        const std::string locator = request_.graph_locator.empty() ? "graph" : request_.graph_locator;
        try {
            const CompileConfig config = CompileConfig::Create(request_.target, request_.opt_level);
            const internal::CompilerExecutionContract contract =
                internal::ResolveCompilerExecutionContract(config);
            if (!request_.pipeline_fingerprint.empty() &&
                request_.pipeline_fingerprint != contract.fingerprint) {
                AddIssue(locator, "Pipeline", "pipeline_identity",
                         "requested pipeline fingerprint does not match the normalized execution plan", false);
                return;
            }
            result_.pipeline_fingerprint = contract.fingerprint;
            internal::ProbeCompilerExecution(request_.function, config, contract);
        } catch (const std::exception& error) {
            const std::string detail = error.what();
            std::string capability = "backend_executable";
            std::string node_kind = "Target";
            if (detail.find("prepare_relay") != std::string::npos ||
                detail.find("optimize_relay") != std::string::npos ||
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

const char* ToString(CapabilityStatus status) {
    switch (status) {
        case CapabilityStatus::kUnsupported: return "unsupported";
        case CapabilityStatus::kEligibleButNotExecutable: return "eligible_but_not_executable";
        case CapabilityStatus::kExecutable: return "executable";
    }
    return "unknown_status";
}

std::string CapabilityResult::Diagnostic() const {
    if (status == CapabilityStatus::kExecutable) return "executable";
    std::ostringstream stream;
    stream << "executable capability rejected (status=" << ToString(status)
           << ", target=" << target_identity
           << ", pipeline=" << (pipeline_fingerprint.empty() ? "<none>" : pipeline_fingerprint)
           << ")";
    for (const CapabilityIssue& issue : issues) {
        stream << "\n- " << issue.diagnostic_locator << " [" << issue.relay_node_kind
               << "] missing " << issue.missing_capability << ": " << issue.detail;
    }
    return stream.str();
}

CapabilityResult CapabilityVerifier::Verify(const CapabilityRequest& request) {
    return Verifier(request, true).Run();
}

void CapabilityVerifier::RequireEligible(const CapabilityRequest& request) {
    const CapabilityResult result = Verifier(request, false).Run();
    if (result.status == CapabilityStatus::kUnsupported) {
        throw std::invalid_argument(std::string("CapabilityVerifier[") +
                                    ToString(request.boundary) + "]: " + result.Diagnostic());
    }
}

}  // namespace kxc::api
