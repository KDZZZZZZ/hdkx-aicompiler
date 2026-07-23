/*! \file src/compiler/compiler.cc
 * \brief Implements the target-driven per-operator compiler pipeline.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal/compile_state.h"
#include "internal/execution_contract.h"
#include "internal/kernel_abi_builder.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "../runtime/internal/compiled_module_node.h"
#include "kxc/compiler/capability.h"
#include "kxc/compiler/pipeline.h"
#include "../runtime/internal/memory_plan.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/pass/print_ir.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/pass/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "../codegen/llvm/internal/codegen_llvm.h"
#include "../codegen/llvm/internal/llvm_jit.h"
#endif

#if KXC_USE_CUDA
#include "../codegen/cuda/internal/codegen_cuda.h"
#include "../codegen/cuda/internal/cuda_module.h"
#endif

namespace kxc::api {
namespace {

bool SameTargetSnapshot(const Target& left, const Target& right) {
    if (!left.defined() || !right.defined()) return false;
    return internal::CanonicalTargetSnapshot(left) ==
           internal::CanonicalTargetSnapshot(right);
}

std::shared_ptr<profiling::ProfileContext> MaybeCreateProfileContext(
    const CompileConfig& config) {
    if (profiling::CurrentContext()) return profiling::CurrentContext();
    if (!config->profile_options.enabled) return nullptr;
    return profiling::ProfileContext::Create(config->profile_options);
}

profiling::EventSpec MakeStageEvent(const char* stage,
                                    const CompileConfig& config) {
    profiling::EventSpec spec;
    spec.component = "compiler";
    spec.event_type = "compile_stage";
    spec.pass_name = stage;
    spec.fields = profiling::MakeFields({
        {"stage", stage},
        {"target_kind", config->target->kind},
        {"device_type",
         std::to_string(static_cast<int>(config->target->device_type))},
        {"device_id", std::to_string(config->target->device_id)},
        {"opt_level", std::to_string(config->opt_level)},
    });
    return spec;
}

std::string PrimitiveContext(const PrimitiveCompileState& primitive) {
    return "unit " + std::to_string(primitive.unit_id) + " ('" +
           std::string(primitive.symbol) + "', " +
           std::string(primitive.operator_identity) + ")";
}

uint64_t PlannedStorageBytes(const runtime::ExecutablePlan& plan) {
    std::unordered_map<int64_t, uint64_t> bytes_by_storage;
    for (const auto& value : plan.values()) {
        uint64_t elements = 1;
        bool dynamic = false;
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) {
                dynamic = true;
                break;
            }
            if (dimension != 0 &&
                elements > std::numeric_limits<uint64_t>::max() /
                               static_cast<uint64_t>(dimension)) {
                return std::numeric_limits<uint64_t>::max();
            }
            elements *= static_cast<uint64_t>(dimension);
        }
        if (dynamic) continue;
        const uint64_t element_bytes =
            static_cast<uint64_t>(value->dtype.bits / 8) *
            static_cast<uint64_t>(value->dtype.lanes);
        if (element_bytes != 0 &&
            elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
            return std::numeric_limits<uint64_t>::max();
        }
        bytes_by_storage[value->storage_id] =
            std::max(bytes_by_storage[value->storage_id],
                     elements * element_bytes);
    }
    uint64_t total = 0;
    for (const auto& item : bytes_by_storage) {
        if (total > std::numeric_limits<uint64_t>::max() - item.second) {
            return std::numeric_limits<uint64_t>::max();
        }
        total += item.second;
    }
    return total;
}

void AddResultFields(profiling::ScopedSpan* span, const CompileResult& result) {
    const CompileStage stage = result.stage();
    if (stage == CompileStage::kRelayOptimized) {
        const std::string text = relay::pass::ToText(result.optimized_relay());
        span->AddField("ir_hash", profiling::HashText(text));
        span->AddMetric("ir_bytes", static_cast<double>(text.size()));
        return;
    }
    if (static_cast<int>(stage) < static_cast<int>(CompileStage::kLowered)) {
        return;
    }
    const std::vector<PrimitiveCompileState> primitives = result.primitives();
    span->AddMetric("primitive_count", static_cast<double>(primitives.size()));
    std::unordered_set<int64_t> storage_ids;
    for (const auto& value : result.plan().values()) {
        storage_ids.insert(value->storage_id);
    }
    span->AddMetric("value_count",
                    static_cast<double>(result.plan().values().size()));
    span->AddMetric("storage_slot_count",
                    static_cast<double>(storage_ids.size()));
    span->AddMetric(
        "storage_reuse_count",
        static_cast<double>(result.plan().values().size() -
                            storage_ids.size()));
    span->AddMetric("planned_peak_storage_bytes",
                    static_cast<double>(PlannedStorageBytes(result.plan())));
    size_t cache_hits = 0;
    for (const PrimitiveCompileState& primitive : primitives) {
        const std::string prefix = "unit." + std::to_string(primitive.unit_id) + ".";
        std::ostringstream stream;
        tir::pass::DumpPrimFunc(primitive.tir, stream);
        const std::string text = stream.str();
        span->AddField(prefix + "symbol", std::string(primitive.symbol));
        span->AddField(prefix + "operator",
                       std::string(primitive.operator_identity));
        span->AddField(prefix + "ir_hash", profiling::HashText(text));
        span->AddMetric(prefix + "ir_bytes", static_cast<double>(text.size()));
        if (primitive.launch_metadata) {
            span->AddField(
                prefix + "backend",
                (*primitive.launch_metadata)->backend ==
                        codegen::CodeGenBackend::kLLVM
                    ? "llvm"
                    : "cuda");
            span->AddMetric(prefix + "cache_hit",
                            primitive.cache_hit ? 1.0 : 0.0);
            if (primitive.cache_hit) ++cache_hits;
        }
    }
    if (stage == CompileStage::kBackendCompiled) {
        span->AddMetric("cache_hits", static_cast<double>(cache_hits));
        span->AddMetric(
            "cache_hit_rate",
            primitives.empty()
                ? 0.0
                : static_cast<double>(cache_hits) /
                      static_cast<double>(primitives.size()));
    }
}

template <typename Fn>
CompileResult RunStage(const char* stage, const CompileConfig& config, Fn&& fn) {
    profiling::ScopedSpan span(profiling::CurrentContext(),
                               MakeStageEvent(stage, config));
    try {
        CompileResult result = fn();
        AddResultFields(&span, result);
        return result;
    } catch (const std::exception& error) {
        span.SetStatus("error");
        span.SetMessage(error.what());
        throw std::runtime_error(std::string("Compiler stage '") + stage +
                                 "' failed: " + error.what());
    }
}

template <typename Fn>
auto RunPrimitiveStage(const char* stage,
                       const PrimitiveCompileState& primitive, Fn&& fn)
    -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(stage) + " failed for " +
                                 PrimitiveContext(primitive) + ": " +
                                 error.what());
    }
}

String ReadKernelSymbol(const tir::PrimFunc& function) {
    const String key("global_symbol");
    if (!function->attrs.count(key)) {
        throw std::invalid_argument("optimized PrimFunc is missing global_symbol");
    }
    const auto* value = function->attrs.at(key).As<StringObj>();
    if (!value || value->data.empty()) {
        throw std::invalid_argument(
            "PrimFunc global_symbol must be a non-empty String");
    }
    const std::string& symbol = value->data;
    const auto is_head = [](unsigned char ch) {
        return std::isalpha(ch) != 0 || ch == '_';
    };
    const auto is_tail = [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_';
    };
    if (!is_head(static_cast<unsigned char>(symbol[0]))) {
        throw std::invalid_argument("PrimFunc global_symbol is not a C identifier");
    }
    for (size_t i = 1; i < symbol.size(); ++i) {
        if (!is_tail(static_cast<unsigned char>(symbol[i]))) {
            throw std::invalid_argument(
                "PrimFunc global_symbol is not a C identifier");
        }
    }
    return String(symbol);
}

Map<String, runtime::NDArray> PlaceConstants(
    const Map<String, runtime::NDArray>& source, const Target& target) {
    Map<String, runtime::NDArray> result;
    const Device target_device(target->device_type, target->device_id);
    for (const auto& item : source) {
        runtime::NDArray value = item.second;
#if KXC_USE_CUDA
        if (value.device() != target_device) value = value.CopyTo(target_device);
#else
        if (target_device.device_type() != kCUDA &&
            value.device() != target_device) {
            value = value.CopyTo(target_device);
        }
#endif
        result.Set(item.first, std::move(value));
    }
    return result;
}

NormalizedPipeline ResolveRelayPipeline(const CompileConfig& config) {
    PipelineRequest request;
    request.dialect = IRDialect::kRelay;
    request.requested_scope = PassScope::kGraph;
    request.target = config->target;
    request.opt_level = config->opt_level;
    request.named_pipeline = String("compiler");
    return PipelineResolver::Resolve(request);
}

NormalizedPipeline ResolveTIRPipeline(const CompileConfig& config) {
    PipelineRequest request;
    request.dialect = IRDialect::kTIR;
    request.requested_scope = PassScope::kPrimFunc;
    request.target = config->target;
    request.opt_level = config->opt_level;
    request.named_pipeline = String("compiler");
    return PipelineResolver::Resolve(request);
}

CompileResult ValidateInput(
    Function function, const CompileConfig& config,
    const internal::CompilerExecutionContract& contract) {
    config.Validate();
    CapabilityVerifier::RequireEligible(CapabilityRequest{
        function, config->target, "graph", contract.fingerprint,
        CapabilityBoundary::kCompilerEntry, CapabilityMode::kStaticExact,
        false, config->opt_level});
    return CompileResult::Validate(config->target, std::move(function));
}

CompileResult OptimizeRelay(
    const CompileResult& input, const CompileConfig& config,
    const internal::CompilerExecutionContract& contract) {
    Function optimized = PipelineExecutor::ExecuteRelay(
        contract.relay_pipeline, input.validated_relay(), input.target());
    CapabilityVerifier::RequireEligible(CapabilityRequest{
        optimized, input.target(), "graph", contract.fingerprint,
        CapabilityBoundary::kPostGraphPass, CapabilityMode::kStaticExact,
        true, config->opt_level});
    return input.AfterRelayOptimization(std::move(optimized));
}

CompileResult LowerPreparedOperators(
    const CompileResult& input, const internal::PreparedStaticGraph& prepared) {
    const Device device(input.target()->device_type, input.target()->device_id);
    if (prepared.device != device ||
        !SameTargetSnapshot(prepared.target, input.target())) {
        throw std::invalid_argument(
            "LowerPreparedOperators requires prepared Device/Target identity unchanged");
    }
    internal::LoweredGraph lowered =
        internal::LowerPreparedStaticGraph(prepared);
    std::vector<PrimitiveCompileState> primitives;
    primitives.reserve(lowered.primitives.size());
    for (const internal::LoweredPrimitive& source : lowered.primitives) {
        PrimitiveCompileState primitive;
        primitive.unit_id = source.unit_id;
        primitive.symbol = source.symbol;
        primitive.operator_identity = source.operator_identity;
        primitive.semantic_key = source.semantic_key;
        primitive.tir = source.lowered->prim_func;
        primitives.push_back(std::move(primitive));
    }
    return input.AfterLowering(
        std::move(primitives), runtime::internal::PlanMemory(lowered.plan),
        PlaceConstants(lowered.constants, input.target()));
}

CompileResult LowerOperators(
    const CompileResult& input,
    const internal::CompilerExecutionContract& contract) {
    const Device device(input.target()->device_type, input.target()->device_id);
    return LowerPreparedOperators(
        input, internal::PrepareStaticGraph(input.optimized_relay(), device,
                                             input.target(),
                                             String(contract.fingerprint)));
}

CompileResult OptimizeTIR(
    const CompileResult& input,
    const internal::CompilerExecutionContract& contract) {
    std::vector<tir::PrimFunc> optimized;
    for (const PrimitiveCompileState& primitive : input.primitives()) {
        optimized.push_back(RunPrimitiveStage(
            "optimize_tir", primitive, [&] {
                return PipelineExecutor::ExecuteTIR(
                    contract.tir_pipeline, primitive.tir, input.target());
            }));
    }
    return input.AfterTIROptimization(std::move(optimized));
}

const char* BackendVersion(const Target& target) {
    if (target->kind == "llvm" && target->device_type == kCPU) {
        return "llvm-orc-v1";
    }
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        return "cuda-nvrtc-driver-v1";
    }
    throw std::invalid_argument("Primitive cache target has no backend version");
}

void AppendPipelineIdentityField(std::string* canonical,
                                 const std::string& name,
                                 const std::string& value) {
    *canonical += std::to_string(name.size()) + ":" + name + "=" +
                  std::to_string(value.size()) + ":" + value + ";";
}

template <typename Node>
const Node* AsExactExprNode(const Expr& expr) noexcept {
    if (!expr.defined() || expr.get()->GetTypeId() != Node::_type_index ||
        typeid(*expr.get()) != typeid(Node)) {
        return nullptr;
    }
    return expr.As<Node>();
}

UnitSemanticKey BuildGraphSemanticKey(const Function& function) {
    if (!AsExactExprNode<FunctionNode>(function)) {
        throw std::invalid_argument(
            "graph semantic identity requires an exact Function node");
    }
    std::string canonical;
    AppendPipelineIdentityField(&canonical, "kind",
                                "graph-semantic-key-v2");
    std::unordered_map<const Object*, size_t> node_ids;
    std::function<void(const Expr&)> visit = [&](const Expr& expr) {
        if (!expr.defined()) {
            throw std::invalid_argument(
                "graph semantic identity has an undefined Relay Expr");
        }
        const auto existing = node_ids.find(expr.get());
        if (existing != node_ids.end()) {
            AppendPipelineIdentityField(&canonical, "node_ref",
                                        std::to_string(existing->second));
            return;
        }
        const size_t node_id = node_ids.size();
        node_ids.emplace(expr.get(), node_id);
        AppendPipelineIdentityField(&canonical, "node_id",
                                    std::to_string(node_id));
        if (const auto* relay_node = dynamic_cast<const RelayNode*>(expr.get())) {
            if (relay_node->virtual_device_.defined()) {
                AppendPipelineIdentityField(
                    &canonical, "virtual_device",
                    relay_node->virtual_device_.ToString());
            }
        }
        if (const auto* constant = AsExactExprNode<ConstantNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "constant");
            if (!constant->data.defined()) {
                throw std::invalid_argument(
                    "graph semantic identity has an undefined Constant");
            }
            const DLDataType dtype = constant->data.dtype();
            AppendPipelineIdentityField(
                &canonical, "constant_dtype_code",
                std::to_string(static_cast<int>(dtype.code)));
            AppendPipelineIdentityField(
                &canonical, "constant_dtype_bits",
                std::to_string(static_cast<int>(dtype.bits)));
            AppendPipelineIdentityField(
                &canonical, "constant_dtype_lanes",
                std::to_string(static_cast<int>(dtype.lanes)));
            for (int64_t dimension : constant->data.shape()) {
                AppendPipelineIdentityField(&canonical, "constant_dimension",
                                            std::to_string(dimension));
            }
            AppendPipelineIdentityField(
                &canonical, "constant_device",
                constant->data.device().ToString());
            std::string bytes(constant->data.NBytes(), '\0');
            if (!bytes.empty()) {
                constant->data.CopyToBytes(bytes.data(), bytes.size());
            }
            AppendPipelineIdentityField(&canonical, "constant_bytes", bytes);
            return;
        }
        if (const auto* variable = AsExactExprNode<VarNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "var");
            AppendPipelineIdentityField(
                &canonical, "type_annotation",
                TypeToString(variable->type_annotation));
            return;
        }
        if (const auto* op = AsExactExprNode<relay::OpNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "op");
            AppendPipelineIdentityField(&canonical, "op_name", op->name);
            AppendPipelineIdentityField(
                &canonical, "op_spec",
                op->has_spec ? relay::SerializeOperatorSpec(op->spec)
                             : "<unspecified>");
            return;
        }
        if (const auto* call = AsExactExprNode<CallNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "call");
            AppendPipelineIdentityField(
                &canonical, "call_attrs",
                call->attrs.defined()
                    ? relay::SerializeAttrs(relay::Attrs(call->attrs))
                    : "<none>");
            visit(call->op);
            for (const Expr& argument : call->args) visit(argument);
            return;
        }
        if (const auto* function_node = AsExactExprNode<FunctionNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "function");
            for (const Var& parameter : function_node->params) visit(parameter);
            visit(function_node->body);
            return;
        }
        if (const auto* branch = AsExactExprNode<IfNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "if");
            visit(branch->cond);
            visit(branch->true_branch);
            visit(branch->false_branch);
            return;
        }
        if (const auto* loop = AsExactExprNode<WhileNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "while");
            AppendPipelineIdentityField(&canonical, "max_trip_count",
                                        std::to_string(loop->max_trip_count));
            visit(loop->initial_state);
            visit(loop->loop_var);
            visit(loop->condition);
            visit(loop->body);
            return;
        }
        if (const auto* let = AsExactExprNode<LetNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "let");
            visit(let->var);
            visit(let->value);
            visit(let->body);
            return;
        }
        if (const auto* tuple = AsExactExprNode<TupleNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "tuple");
            for (const Expr& field : tuple->fields) visit(field);
            return;
        }
        if (const auto* item = AsExactExprNode<TupleGetItemNode>(expr)) {
            AppendPipelineIdentityField(&canonical, "node_kind", "tuple_get");
            AppendPipelineIdentityField(&canonical, "tuple_index",
                                        std::to_string(item->index));
            visit(item->tuple);
            return;
        }
        throw std::invalid_argument(
            "graph semantic identity does not support Relay node type '" +
            std::string(expr.get()->GetTypeKey()) + "'");
    };
    visit(function);
    return UnitSemanticKey(std::move(canonical));
}

bool SameDType(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits &&
           lhs.lanes == rhs.lanes;
}

bool SameShape(const Array<int64_t>& lhs, const Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

void ValidateArtifactABI(const codegen::KernelSignature& cached,
                         const codegen::KernelSignature& current,
                         const std::string& context) {
    const Array<codegen::KernelArgSpec> cached_args = cached.arguments();
    const Array<codegen::KernelArgSpec> current_args = current.arguments();
    if (cached_args.size() != current_args.size()) {
        throw std::logic_error(context + " cached artifact ABI arity changed");
    }
    for (size_t i = 0; i < cached_args.size(); ++i) {
        const auto& lhs = cached_args[i];
        const auto& rhs = current_args[i];
        if (lhs->role != rhs->role || !SameDType(lhs->dtype, rhs->dtype) ||
            lhs->device != rhs->device || lhs->alignment != rhs->alignment ||
            lhs->mutable_data != rhs->mutable_data ||
            !SameShape(lhs.shape(), rhs.shape())) {
            throw std::logic_error(context +
                                   " cached artifact ABI contract changed");
        }
    }
}

codegen::CompiledKernel RelocateCachedKernel(
    const internal::PrimitiveArtifactPin& pin,
    const codegen::KernelSignature& current_signature,
    const std::string& context) {
    const internal::CachedPrimitive& artifact = pin.artifact();
    ValidateArtifactABI(artifact.signature, current_signature, context);
    if (!artifact.kernel.IsReady() || !artifact.kernel->launcher) {
        throw std::logic_error(context + " cached artifact is not executable");
    }
    return codegen::CompiledKernel(current_signature,
                                   artifact.launch_metadata,
                                   artifact.kernel->launcher);
}

uint64_t AccountedTIRBytes(const tir::PrimFunc& function) {
    std::ostringstream out;
    tir::pass::DumpPrimFunc(function, out);
    return std::max<uint64_t>(1, static_cast<uint64_t>(out.str().size()));
}

class PrimitiveOwnerGuard final {
public:
    explicit PrimitiveOwnerGuard(
        const std::vector<ProductionCompileTransaction>* transactions)
        : transactions_(transactions) {}

    ~PrimitiveOwnerGuard() {
        if (dismissed_ || transactions_ == nullptr) return;
        const ProductionArtifactCacheAdapter adapter;
        for (const auto& transaction : *transactions_) {
            if (!transaction.owns_compile()) continue;
            try {
                (void)adapter.Fail(
                    transaction,
                    CompileFailure{
                        CompileFailureCategory::kCompile, 1000,
                        "compile owner abandoned before publishing a validated artifact"});
            } catch (...) {
            }
        }
    }

    void Dismiss() noexcept { dismissed_ = true; }

private:
    const std::vector<ProductionCompileTransaction>* transactions_{nullptr};
    bool dismissed_{false};
};

internal::PrimitiveArtifactPin RequireProductionPin(
    const CompileOutcome& outcome, const std::string& context) {
    if (outcome.state == CompileRequestState::kReady &&
        outcome.pin.defined()) {
        return internal::ProductionArtifactAccess::Pin(outcome.pin);
    }
    const std::string diagnostic =
        outcome.failure ? outcome.failure->diagnostic : "missing compile outcome";
    throw std::runtime_error(context + " production cache transaction failed: " +
                             diagnostic);
}

CompileResult BuildSignatures(const CompileResult& input) {
    std::vector<codegen::KernelSignature> signatures;
    for (const PrimitiveCompileState& primitive : input.primitives()) {
        signatures.push_back(RunPrimitiveStage(
            "build_signature", primitive, [&] {
                const String symbol = ReadKernelSymbol(primitive.tir);
                if (!(symbol == primitive.symbol)) {
                    throw std::invalid_argument(
                        "PrimFunc global_symbol drifted from unit symbol");
                }
                return codegen::BuildKernelSignature(
                    primitive.tir, input.constants(), input.target(), symbol);
            }));
    }
    return input.AfterSignatures(std::move(signatures));
}

CompileResult BuildBackends(
    const CompileResult& input, const CompileConfig& config,
    const internal::CompilerExecutionContract& contract) {
    const Target target = input.target();
    const Device device(target->device_type, target->device_id);
    const std::vector<PrimitiveCompileState> primitives = input.primitives();
    std::vector<std::optional<codegen::KernelLaunchMetadata>> metadata_slots(
        primitives.size());
    std::vector<std::optional<codegen::CompiledKernel>> kernel_slots(
        primitives.size());
    const std::string& pipeline_identity = contract.canonical_bytes;
    const ProductionArtifactCacheAdapter cache_adapter;
    std::vector<ProductionCompileTransaction> transactions;
    std::vector<internal::PrimitiveArtifactPin> pins(primitives.size());
    std::vector<size_t> misses;
    transactions.reserve(primitives.size());
    PrimitiveOwnerGuard owner_guard(&transactions);
    for (size_t i = 0; i < primitives.size(); ++i) {
        const PrimitiveCompileState& primitive = primitives[i];
        if (!primitive.signature) {
            throw std::logic_error(
                PrimitiveContext(primitive) + " has no signature");
        }
        const ArtifactKey artifact_key = internal::BuildPrimitiveArtifactKey(
            primitive.semantic_key, target, pipeline_identity,
            contract.schedule_version.c_str(),
            contract.backend_version.c_str());
        CompileRequest request;
        request.artifact_key = artifact_key;
        request.request_origin = "Compiler::Compile";
        request.cancellation.id = "compiler-sync-" + std::to_string(i);
        transactions.push_back(cache_adapter.Acquire(request));
        if (transactions.back().owns_compile()) {
            misses.push_back(i);
            continue;
        }
        const CompileRequestState state = transactions.back().ticket().state;
        if (state == CompileRequestState::kFailed ||
            state == CompileRequestState::kCancelled ||
            state == CompileRequestState::kRejected) {
            (void)RequireProductionPin(
                cache_adapter.Wait(transactions.back()),
                PrimitiveContext(primitive));
        }
    }

    if (!misses.empty() && target->kind == "llvm" &&
        target->device_type == kCPU) {
#if KXC_USE_LLVM
        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_context);
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (size_t index : misses) {
            const PrimitiveCompileState& primitive = primitives[index];
            functions.emplace_back(primitive.tir,
                                   std::string(primitive.symbol));
            signatures.push_back(*primitive.signature);
            metadata.emplace_back(
                device, codegen::CodeGenBackend::kLLVM);
        }
        codegen.AddFunctions(functions);
        codegen::LLVMJITEngine jit;
        std::vector<codegen::CompiledKernel> compiled = jit.CompileMany(
            codegen.TakeModule(), std::move(llvm_context), signatures,
            metadata, config->opt_level);
        for (size_t i = 0; i < misses.size(); ++i) {
            const size_t index = misses[i];
            metadata_slots[index] = metadata[i];
            kernel_slots[index] = compiled[i];
        }
#else
        throw std::runtime_error(
            "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON");
#endif
    } else if (!misses.empty() && target->kind == "cuda" &&
               target->device_type == kCUDA) {
#if KXC_USE_CUDA
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (size_t index : misses) {
            const PrimitiveCompileState& primitive = primitives[index];
            const tir::CudaLaunchConfig launch_config =
                tir::GetCudaLaunchConfig(primitive.tir);
            functions.emplace_back(primitive.tir,
                                   std::string(primitive.symbol));
            signatures.push_back(*primitive.signature);
            metadata.emplace_back(
                device, codegen::CodeGenBackend::kCUDA,
                codegen::Dim3{launch_config.grid_x, launch_config.grid_y,
                              launch_config.grid_z},
                codegen::Dim3{launch_config.block_x, launch_config.block_y,
                              launch_config.block_z},
                launch_config.dynamic_shared_memory_bytes);
        }
        if (target->attrs.compute_version_major <= 0 ||
            target->attrs.compute_version_minor < 0) {
            throw std::runtime_error(
                "CUDA Target has no usable compute capability");
        }
        codegen::CodeGenCUDA emitter;
        const std::string source = emitter.GenerateModule(functions);
        codegen::CUDACompileOptions options;
        options.architecture =
            "compute_" +
            std::to_string(target->attrs.compute_version_major) +
            std::to_string(target->attrs.compute_version_minor);
        options.source_name = "kxc_operator_module.cu";
        std::vector<codegen::CompiledKernel> compiled =
            codegen::CUDAModule::CompileMany(source, signatures, metadata,
                                             options);
        for (size_t i = 0; i < misses.size(); ++i) {
            const size_t index = misses[i];
            metadata_slots[index] = metadata[i];
            kernel_slots[index] = compiled[i];
        }
#else
        throw std::runtime_error(
            "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON");
#endif
    } else if (!misses.empty()) {
        throw std::runtime_error("Compiler Target has no matching backend");
    }

    for (size_t index : misses) {
        if (!metadata_slots[index] || !kernel_slots[index]) {
            throw std::logic_error(
                PrimitiveContext(primitives[index]) +
                " backend batch did not produce an executable");
        }
        pins[index] = RequireProductionPin(
            cache_adapter.Publish(
                transactions[index],
                internal::ProductionArtifactAccess::Make(
                    internal::CachedPrimitive{
                        *primitives[index].signature, *metadata_slots[index],
                        *kernel_slots[index],
                        AccountedTIRBytes(primitives[index].tir),
                        "Compiler::Compile",
                        "signature+backend-validated"})),
            PrimitiveContext(primitives[index]));
    }

    std::vector<codegen::KernelLaunchMetadata> metadata;
    std::vector<codegen::CompiledKernel> kernels;
    std::vector<bool> cache_hits;
    metadata.reserve(primitives.size());
    kernels.reserve(primitives.size());
    cache_hits.reserve(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        if (!pins[i].defined()) {
            pins[i] = RequireProductionPin(
                cache_adapter.Wait(transactions[i]),
                PrimitiveContext(primitives[i]));
        }
        const internal::CachedPrimitive& artifact = pins[i].artifact();
        metadata_slots[i] = artifact.launch_metadata;
        kernel_slots[i] = RelocateCachedKernel(
            pins[i], *primitives[i].signature, PrimitiveContext(primitives[i]));
        metadata.push_back(*metadata_slots[i]);
        kernels.push_back(*kernel_slots[i]);
        cache_hits.push_back(!transactions[i].owns_compile());
    }
    std::vector<ArtifactPin> public_pins;
    public_pins.reserve(pins.size());
    for (const internal::PrimitiveArtifactPin& pin : pins) {
        public_pins.push_back(internal::ToArtifactPin(pin));
    }
    owner_guard.Dismiss();
    return input.AfterBackends(std::move(metadata), std::move(kernels),
                               std::move(cache_hits),
                               std::move(public_pins));
}

CompiledModule AssembleModule(
    const CompileResult& result,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    result.ValidateState();
    if (result.stage() != CompileStage::kBackendCompiled) {
        throw std::logic_error("AssembleModule requires backend_compiled state");
    }
    std::vector<internal::CompiledModuleEntry> entries;
    for (const PrimitiveCompileState& primitive : result.primitives()) {
        entries.push_back(internal::CompiledModuleEntry{
            primitive.tir, *primitive.signature, *primitive.launch_metadata,
            *primitive.kernel});
    }
    return internal::BuildCompiledModule(
        result.target(), std::move(entries), result.constants(),
        std::move(profile_context));
}

runtime::PlanVariant AssemblePlanVariant(
    const CompiledModule& module, const runtime::ExecutablePlan& plan,
    const std::vector<ArtifactPin>& pins) {
    const Array<runtime::KernelCall> calls = plan.calls();
    if (pins.size() != calls.size()) {
        throw std::logic_error(
            "selected artifact pins do not match ordered plan calls");
    }
    Array<runtime::ArtifactSelection> selections;
    for (size_t index = 0; index < pins.size(); ++index) {
        if (!pins[index].defined()) {
            throw std::logic_error(
                "selected artifact manifest cannot retain an undefined pin");
        }
        selections.push_back(runtime::ArtifactSelection{
            static_cast<int64_t>(index),
            String(pins[index]
                       .handle()
                       .record()
                       .artifact_key.canonical_bytes()),
            0});
    }
    // Compiler is the authority that associates this declaration with pins;
    // Runtime only observes it and keeps the type-erased lease alive.
    const auto retention_lease =
        std::make_shared<const std::vector<ArtifactPin>>(pins);
    return runtime::MakePlanVariant(module, plan, selections,
                                    retention_lease);
}

CompiledGraph CompilePipeline(
    Function function, CompileConfig config,
    const internal::CompilerExecutionContract& contract) {
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);

    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);

    CompileResult result = RunStage(
        "validate", config,
        [&] { return ValidateInput(function, config, contract); });
    result = RunStage(
        "optimize_relay", config,
        [&] { return OptimizeRelay(result, config, contract); });
    result = RunStage("lower", config,
                      [&] { return LowerOperators(result, contract); });
    result = RunStage("optimize_tir", config,
                      [&] { return OptimizeTIR(result, contract); });
    result = RunStage("build_signature", config,
                      [&] { return BuildSignatures(result); });
    result = RunStage(
        "build_backend", config,
        [&] { return BuildBackends(result, config, contract); });

    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    runtime::ExecutablePlan plan = result.plan();
    std::vector<ArtifactPin> artifact_pins = result.artifact_pins();
    runtime::PlanVariant variant =
        AssemblePlanVariant(module, plan, artifact_pins);
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    std::vector<ArtifactPlanBinding> bindings;
    bindings.reserve(artifact_pins.size());
    const Array<runtime::KernelCall> calls = plan.calls();
    const std::vector<PrimitiveCompileState> primitives = result.primitives();
    if (calls.size() != artifact_pins.size() ||
        primitives.size() != artifact_pins.size()) {
        throw std::logic_error(
            "Compiler produced different plan-call and artifact-pin counts");
    }
    for (size_t index = 0; index < artifact_pins.size(); ++index) {
        bindings.push_back(ArtifactPlanBinding{
            index,
            LinkSymbol{std::string(calls[index]->symbol)},
            artifact_pins[index],
            profiling::HashText(primitives[index].signature->ToString()),
            profiling::HashText(
                primitives[index].launch_metadata->ToString())});
    }
    return CompiledGraph{std::move(module), std::move(plan),
                         std::move(artifact_pins), std::move(variant),
                         std::move(bindings), ArtifactKey()};
}

}  // namespace

Target internal::CloneTargetSnapshot(const Target& target) {
    const auto* source = target.As<TargetNode>();
    if (!source) {
        throw std::invalid_argument(
            "CloneTargetSnapshot requires a defined TargetNode");
    }
    auto* copy = new TargetNode();
    copy->kind = source->kind;
    copy->device_type = source->device_type;
    copy->device_id = source->device_id;
    copy->attrs = source->attrs;
    return Target(ObjectRef(copy));
}

std::string internal::CanonicalTargetSnapshot(const Target& target) {
    const auto* node = target.As<TargetNode>();
    if (!node) {
        throw std::invalid_argument(
            "CanonicalTargetSnapshot requires a defined TargetNode");
    }
    std::string canonical;
    AppendPipelineIdentityField(&canonical, "kind", "target-snapshot-v1");
    AppendPipelineIdentityField(&canonical, "target_kind", node->kind);
    AppendPipelineIdentityField(
        &canonical, "device_type",
        std::to_string(static_cast<int>(node->device_type)));
    AppendPipelineIdentityField(&canonical, "device_id",
                                std::to_string(node->device_id));
    const DeviceAttributes& attrs = node->attrs;
    const auto append_integer = [&canonical](const char* name, int64_t value) {
        AppendPipelineIdentityField(&canonical, name, std::to_string(value));
    };
    append_integer("exists", attrs.exists);
    append_integer("max_threads_per_block", attrs.max_threads_per_block);
    append_integer("warp_size", attrs.warp_size);
    append_integer("max_shared_memory_per_block",
                   attrs.max_shared_memory_per_block);
    AppendPipelineIdentityField(&canonical, "compute_version",
                                attrs.compute_version);
    AppendPipelineIdentityField(&canonical, "device_name", attrs.device_name);
    append_integer("max_clock_rate_khz", attrs.max_clock_rate_khz);
    append_integer("max_registers_per_block", attrs.max_registers_per_block);
    append_integer("api_version", attrs.api_version);
    append_integer("driver_version", attrs.driver_version);
    append_integer("l2_cache_size_bytes", attrs.l2_cache_size_bytes);
    append_integer("total_global_memory", attrs.total_global_memory);
    // available_global_memory is a volatile observation, not a codegen
    // capability. It is deliberately excluded from reusable identity.
    append_integer("max_shared_memory_per_multiprocessor",
                   attrs.max_shared_memory_per_multiprocessor);
    append_integer("max_registers_per_multiprocessor",
                   attrs.max_registers_per_multiprocessor);
    append_integer("max_threads_per_multiprocessor",
                   attrs.max_threads_per_multiprocessor);
    append_integer("compute_version_major", attrs.compute_version_major);
    append_integer("compute_version_minor", attrs.compute_version_minor);
    append_integer("multi_processor_count", attrs.multi_processor_count);
    AppendPipelineIdentityField(&canonical, "arch", attrs.arch);
    return canonical;
}

internal::CompilerExecutionContract
internal::ResolveCompilerExecutionContract(const CompileConfig& config) {
    config.Validate();
    CompilerExecutionContract contract;
    contract.relay_pipeline = ResolveRelayPipeline(config);
    contract.tir_pipeline = ResolveTIRPipeline(config);
    contract.schedule_version = "per-unit-schedule-v2";
    contract.backend_version = BackendVersion(config->target);
    AppendPipelineIdentityField(&contract.canonical_bytes, "kind",
                                "compiler-execution-plan-v3");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "relay",
        std::string(contract.relay_pipeline.canonical_bytes));
    AppendPipelineIdentityField(&contract.canonical_bytes, "lowering",
                                "per-unit-boundary-lowering-v1");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "tir",
        std::string(contract.tir_pipeline.canonical_bytes));
    AppendPipelineIdentityField(&contract.canonical_bytes, "kernel_abi",
                                "kernel-abi-v1");
    AppendPipelineIdentityField(&contract.canonical_bytes, "schedule",
                                contract.schedule_version);
    AppendPipelineIdentityField(&contract.canonical_bytes, "backend",
                                contract.backend_version);
    contract.fingerprint = profiling::HashText(contract.canonical_bytes);
    return contract;
}

void internal::ProbeCompilerExecution(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
    (void)CompilePipeline(std::move(function), std::move(config), contract);
}

internal::PreparedCompilerGraph internal::PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
    config.Validate();
    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("shape_exact") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    size_t capability_boundary_checks = 0;
    size_t relay_graph_pipelines = 0;
    CompileResult result = RunStage(
        "validate", config,
        [&] { return ValidateInput(std::move(function), config, contract); });
    ++capability_boundary_checks;
    result = RunStage("optimize_relay", config,
                      [&] { return OptimizeRelay(result, config, contract); });
    ++relay_graph_pipelines;
    ++capability_boundary_checks;
    const Device device(result.target()->device_type, result.target()->device_id);
    profiling::ScopedSpan prepare_span(
        profile_context, MakeStageEvent("prepare_graph", config), run_id);
    PreparedStaticGraph graph;
    try {
        graph = PrepareStaticGraph(result.optimized_relay(), device,
                                   result.target(),
                                   String(contract.fingerprint));
    } catch (const std::exception& error) {
        prepare_span.SetStatus("error");
        prepare_span.SetMessage(error.what());
        throw std::runtime_error(
            std::string("Compiler stage 'prepare_graph' failed: ") +
            error.what());
    }
    capability_boundary_checks += graph.capability_boundary_checks;
    const size_t value_graph_builds = graph.value_graph_builds;
    const size_t partitions = graph.partitions;
    return PreparedCompilerGraph{
        result, std::move(graph), result.target(), contract.canonical_bytes,
        std::move(profile_context), run_id, relay_graph_pipelines,
        capability_boundary_checks, value_graph_builds, partitions};
}

CompiledGraph internal::FinishCompilerGraph(
    const PreparedCompilerGraph& prepared, CompileConfig config,
    const CompilerExecutionContract& contract) {
    config.Validate();
    if (prepared.execution_contract_canonical != contract.canonical_bytes ||
        !SameTargetSnapshot(prepared.target, config->target) ||
        !SameTargetSnapshot(prepared.optimized.target(), config->target) ||
        !SameTargetSnapshot(prepared.graph.target, config->target) ||
        prepared.graph.device !=
            Device(config->target->device_type, config->target->device_id) ||
        std::string(prepared.graph.pipeline_fingerprint) !=
            contract.fingerprint) {
        throw std::invalid_argument(
            "FinishCompilerGraph requires the prepared target and execution contract unchanged");
    }
    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(prepared.optimized.optimized_relay()), config->target);
    PassContext::Scope pass_scope(pass_context);
    auto profile_context = prepared.profile_context;
    const std::string& run_id = prepared.profile_run_id;
    profiling::ActivationScope activation(profile_context, run_id);
    CompileResult result = RunStage("lower", config,
        [&] { return LowerPreparedOperators(prepared.optimized, prepared.graph); });
    result = RunStage("optimize_tir", config,
                      [&] { return OptimizeTIR(result, contract); });
    result = RunStage("build_signature", config,
                      [&] { return BuildSignatures(result); });
    result = RunStage("build_backend", config,
                      [&] { return BuildBackends(result, config, contract); });
    profiling::ScopedSpan assemble_span(profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    runtime::ExecutablePlan plan = result.plan();
    std::vector<ArtifactPin> artifact_pins = result.artifact_pins();
    runtime::PlanVariant variant =
        AssemblePlanVariant(module, plan, artifact_pins);
    std::vector<ArtifactPlanBinding> bindings;
    bindings.reserve(artifact_pins.size());
    const Array<runtime::KernelCall> calls = plan.calls();
    const std::vector<PrimitiveCompileState> primitives = result.primitives();
    if (calls.size() != artifact_pins.size() ||
        primitives.size() != artifact_pins.size()) {
        throw std::logic_error(
            "Compiler produced different plan-call and artifact-pin counts");
    }
    for (size_t index = 0; index < artifact_pins.size(); ++index) {
        bindings.push_back(ArtifactPlanBinding{
            index,
            LinkSymbol{std::string(calls[index]->symbol)},
            artifact_pins[index],
            profiling::HashText(primitives[index].signature->ToString()),
            profiling::HashText(
                primitives[index].launch_metadata->ToString())});
    }
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    return CompiledGraph{std::move(module), std::move(plan),
                         std::move(artifact_pins), std::move(variant),
                         std::move(bindings), ArtifactKey()};
}

Array<String> Compiler::RelayPassPolicy(int opt_level) {
    auto* policy_target_node = new TargetNode();
    policy_target_node->kind = "llvm";
    policy_target_node->device_type = kCPU;
    policy_target_node->device_id = 0;
    PipelineRequest request;
    request.dialect = IRDialect::kRelay;
    request.requested_scope = PassScope::kGraph;
    request.target = Target(ObjectRef(policy_target_node));
    request.opt_level = opt_level;
    request.named_pipeline = String("compiler");
    Array<String> compatibility;
    for (const String& pass :
         PipelineResolver::Resolve(request).ordered_passes) {
        const std::string name(pass);
        if (name != "infer_type" && name != "normalize_to_anf") {
            compatibility.push_back(pass);
        }
    }
    return compatibility;
}

Array<String> Compiler::TIRPassPolicy(int opt_level, const Target& target) {
    PipelineRequest request;
    request.dialect = IRDialect::kTIR;
    request.requested_scope = PassScope::kPrimFunc;
    request.target = target;
    request.opt_level = opt_level;
    request.named_pipeline = String("compiler");
    Array<String> compatibility;
    for (const String& pass :
         PipelineResolver::Resolve(request).ordered_passes) {
        if (std::string(pass) != "bind_cuda_threads") {
            compatibility.push_back(pass);
        }
    }
    return compatibility;
}

ArtifactKey Compiler::BuildGraphArtifactKey(
    const Function& function, const CompileConfig& config) {
    if (!function.defined()) {
        throw std::invalid_argument(
            "graph artifact identity requires a defined Function");
    }
    config.Validate();
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    return ArtifactKey(
        BuildGraphSemanticKey(function),
        internal::BuildTargetCapabilityFingerprint(config->target),
        contract.canonical_bytes, 1, contract.schedule_version,
        contract.backend_version);
}

CompiledGraph Compiler::Compile(Function function, CompileConfig config) {
    const ArtifactKey graph_artifact_key =
        BuildGraphArtifactKey(function, config);
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    CompiledGraph result =
        CompilePipeline(std::move(function), std::move(config), contract);
    result.graph_artifact_key = graph_artifact_key;
    return result;
}
}  // namespace kxc::api
