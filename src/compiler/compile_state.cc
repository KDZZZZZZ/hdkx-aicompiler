/*! \file src/compiler/compile_state.cc
 * \brief Implements validation and transitions for multi-primitive compile state.
 */

#include "internal/compile_state.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace kxc::api {

KXC_OBJECT_DEFINE_WITH_KEY(CompileResultNode, "kxc.api.CompileResultNode")

namespace {

bool HasReached(CompileStage current, CompileStage required) {
    return static_cast<int>(current) >= static_cast<int>(required);
}

void RequireStage(CompileStage actual, CompileStage expected,
                  const char* operation) {
    if (actual != expected) {
        throw std::logic_error(std::string(operation) + " requires stage " +
                               std::to_string(static_cast<int>(expected)) +
                               ", actual stage is " +
                               std::to_string(static_cast<int>(actual)));
    }
}

void ValidateFunction(const Function& function, const char* field) {
    if (!function.defined() || !function.As<FunctionNode>() ||
        !function->body.defined()) {
        throw std::invalid_argument(std::string(field) + " must have a body");
    }
}

void ValidatePrimFunc(const tir::PrimFunc& function,
                      const std::string& context) {
    if (!function.defined() || !function.As<tir::PrimFuncNode>() ||
        !function->body.defined()) {
        throw std::invalid_argument(context +
                                    " must contain a PrimFunc with a body");
    }
}

void ValidateTarget(const Target& target) {
    if (!target.defined() || !target.As<TargetNode>() ||
        target->device_type == kUnknown || target->device_id < 0 ||
        target->kind.empty()) {
        throw std::invalid_argument("CompileResult target identity is incomplete");
    }
    (void)Device(target->device_type, target->device_id);
}

Map<String, runtime::NDArray> CopyConstants(
    const Map<String, runtime::NDArray>& constants) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : constants) result.Set(item.first, item.second);
    return result;
}

void ValidateConstants(const Map<String, runtime::NDArray>& constants) {
    for (const auto& item : constants) {
        if (std::string(item.first).empty() || !item.second.defined() ||
            !item.second.As<runtime::NDArrayNode>()) {
            throw std::invalid_argument(
                "CompileResult constants require non-empty keys and NDArrays");
        }
    }
}

std::string PrimitiveContext(const PrimitiveCompileState& primitive) {
    return "unit " + std::to_string(primitive.unit_id) + " ('" +
           std::string(primitive.symbol) + "', " +
           std::string(primitive.operator_identity) + ")";
}

int64_t ReadIntAttr(const tir::PrimFunc& function, const char* key,
                    const std::string& context) {
    const String attr_key(key);
    if (!function->attrs.count(attr_key)) {
        throw std::invalid_argument(context + " is missing PrimFunc attr " + key);
    }
    const auto* value = function->attrs.at(attr_key).As<tir::IntImmNode>();
    if (!value) {
        throw std::invalid_argument(context + " has non-integer PrimFunc attr " + key);
    }
    return value->value;
}

String ReadStringAttr(const tir::PrimFunc& function, const char* key,
                      const std::string& context) {
    const String attr_key(key);
    if (!function->attrs.count(attr_key)) {
        throw std::invalid_argument(context + " is missing PrimFunc attr " + key);
    }
    const auto* value = function->attrs.at(attr_key).As<StringObj>();
    if (!value || value->data.empty()) {
        throw std::invalid_argument(context + " has invalid PrimFunc attr " + key);
    }
    return String(value->data);
}

void ValidateSignatureTarget(const codegen::KernelSignature& signature,
                             const Target& target,
                             const std::string& context) {
    signature.Validate();
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    if (arguments.empty()) {
        throw std::invalid_argument(context + " signature has no arguments");
    }
    const Device expected(target->device_type, target->device_id);
    for (const auto& argument : arguments) {
        if (argument->device != expected) {
            throw std::invalid_argument(context +
                                        " signature device does not match Target");
        }
    }
}

void ValidatePrimitiveIdentity(const PrimitiveCompileState& primitive,
                               size_t index) {
    if (primitive.unit_id != static_cast<int64_t>(index) ||
        std::string(primitive.symbol).empty() ||
        std::string(primitive.operator_identity).empty() ||
        std::string(primitive.structural_hash).empty()) {
        throw std::invalid_argument(
            "CompileResult primitive identities must be dense and non-empty");
    }
    const std::string context = PrimitiveContext(primitive);
    ValidatePrimFunc(primitive.tir, context);
    if (ReadIntAttr(primitive.tir, "kxc.unit_id", context) != primitive.unit_id ||
        !(ReadStringAttr(primitive.tir, "global_symbol", context) ==
          primitive.symbol) ||
        !(ReadStringAttr(primitive.tir, "kxc.operator_identity", context) ==
          primitive.operator_identity) ||
        !(ReadStringAttr(primitive.tir, "kxc.structural_hash", context) ==
          primitive.structural_hash)) {
        throw std::invalid_argument(context +
                                    " identity drifted from PrimFunc metadata");
    }
}

}  // namespace

CompileResult::CompileResult(CompileResultNode* node) : ObjectRef(node) {}

CompileResult CompileResult::Validate(Target target, Function relay) {
    ValidateTarget(target);
    ValidateFunction(relay, "validated relay");
    auto* node = new CompileResultNode();
    node->target_ = std::move(target);
    node->relay_ = std::move(relay);
    CompileResult result(node);
    result.ValidateState();
    return result;
}

CompileResult::CompileResult(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompileResultNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompileResultNode");
    }
    if (defined()) ValidateState();
}

CompileResult CompileResult::AfterRelayOptimization(
    Function optimized_relay) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kValidated,
                 "AfterRelayOptimization");
    ValidateFunction(optimized_relay, "optimized relay");
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kRelayOptimized;
    next->target_ = current->target_;
    next->relay_ = std::move(optimized_relay);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

CompileResult CompileResult::AfterLowering(
    std::vector<PrimitiveCompileState> lowered_primitives,
    runtime::ExecutablePlan plan,
    const Map<String, runtime::NDArray>& constants) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kRelayOptimized,
                 "AfterLowering");
    plan.Validate();
    ValidateConstants(constants);
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kLowered;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->primitives_ = std::move(lowered_primitives);
    next->plan_ = std::move(plan);
    next->constants_ = CopyConstants(constants);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

CompileResult CompileResult::AfterTIROptimization(
    std::vector<tir::PrimFunc> optimized_tir) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kLowered,
                 "AfterTIROptimization");
    if (optimized_tir.size() != current->primitives_.size()) {
        throw std::invalid_argument(
            "AfterTIROptimization primitive count changed");
    }
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kTIROptimized;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->primitives_ = current->primitives_;
    for (size_t i = 0; i < optimized_tir.size(); ++i) {
        next->primitives_[i].tir = std::move(optimized_tir[i]);
        next->primitives_[i].structural_hash = ReadStringAttr(
            next->primitives_[i].tir, "kxc.structural_hash",
            PrimitiveContext(next->primitives_[i]));
    }
    next->plan_ = current->plan_;
    next->constants_ = CopyConstants(current->constants_);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

CompileResult CompileResult::AfterSignatures(
    std::vector<codegen::KernelSignature> signatures,
    std::vector<bool> cache_hits) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kTIROptimized,
                 "AfterSignatures");
    if (signatures.size() != current->primitives_.size()) {
        throw std::invalid_argument("AfterSignatures primitive count changed");
    }
    if (cache_hits.empty()) cache_hits.resize(signatures.size(), false);
    if (cache_hits.size() != signatures.size()) {
        throw std::invalid_argument("AfterSignatures cache-hit count changed");
    }
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kSignatureBuilt;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->primitives_ = current->primitives_;
    for (size_t i = 0; i < signatures.size(); ++i) {
        next->primitives_[i].signature = std::move(signatures[i]);
        next->primitives_[i].cache_hit = cache_hits[i];
    }
    next->plan_ = current->plan_;
    next->constants_ = CopyConstants(current->constants_);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

CompileResult CompileResult::AfterBackends(
    std::vector<codegen::KernelLaunchMetadata> launch_metadata,
    std::vector<codegen::CompiledKernel> kernels) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kSignatureBuilt,
                 "AfterBackends");
    if (launch_metadata.size() != current->primitives_.size() ||
        kernels.size() != current->primitives_.size()) {
        throw std::invalid_argument("AfterBackends primitive count changed");
    }
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kBackendCompiled;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->primitives_ = current->primitives_;
    for (size_t i = 0; i < kernels.size(); ++i) {
        next->primitives_[i].launch_metadata = std::move(launch_metadata[i]);
        next->primitives_[i].kernel = std::move(kernels[i]);
    }
    next->plan_ = current->plan_;
    next->constants_ = CopyConstants(current->constants_);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

void CompileResult::ValidateState() const {
    const auto* node = operator->();
    ValidateTarget(node->target_);
    const int stage_value = static_cast<int>(node->stage_);
    if (stage_value < static_cast<int>(CompileStage::kValidated) ||
        stage_value > static_cast<int>(CompileStage::kBackendCompiled)) {
        throw std::invalid_argument("CompileResult contains an unknown stage");
    }
    if (!node->relay_) {
        throw std::invalid_argument("CompileResult stage requires Relay state");
    }
    ValidateFunction(*node->relay_, "CompileResult relay");

    const bool needs_primitives =
        HasReached(node->stage_, CompileStage::kLowered);
    if (needs_primitives != !node->primitives_.empty() ||
        needs_primitives != node->plan_.has_value()) {
        throw std::invalid_argument(
            "CompileResult primitive/plan presence does not match stage");
    }
    if (!needs_primitives && node->constants_.size() != 0) {
        throw std::invalid_argument(
            "CompileResult constants cannot exist before lowering");
    }
    ValidateConstants(node->constants_);
    if (!needs_primitives) return;

    node->plan_->Validate();
    const Array<runtime::KernelCall> calls = node->plan_->calls();
    if (calls.size() != node->primitives_.size()) {
        throw std::invalid_argument(
            "CompileResult plan and primitive counts do not match");
    }
    std::unordered_set<std::string> symbols;
    const bool needs_signature =
        HasReached(node->stage_, CompileStage::kSignatureBuilt);
    const bool needs_backend =
        HasReached(node->stage_, CompileStage::kBackendCompiled);
    const Device expected(node->target_->device_type, node->target_->device_id);
    for (size_t i = 0; i < node->primitives_.size(); ++i) {
        const PrimitiveCompileState& primitive = node->primitives_[i];
        ValidatePrimitiveIdentity(primitive, i);
        const std::string context = PrimitiveContext(primitive);
        if (!symbols.insert(std::string(primitive.symbol)).second ||
            !(calls[i]->symbol == primitive.symbol)) {
            throw std::invalid_argument(context +
                                        " plan symbol is missing or duplicated");
        }
        if (needs_signature != primitive.signature.has_value()) {
            throw std::invalid_argument(context +
                                        " signature presence does not match stage");
        }
        if (!needs_signature && primitive.cache_hit) {
            throw std::invalid_argument(
                context + " cache state exists before signature construction");
        }
        if (primitive.signature) {
            ValidateSignatureTarget(*primitive.signature, node->target_, context);
            if (!((*primitive.signature)->symbol == primitive.symbol)) {
                throw std::invalid_argument(context + " signature symbol drifted");
            }
        }
        if (needs_backend != primitive.launch_metadata.has_value() ||
            needs_backend != primitive.kernel.has_value()) {
            throw std::invalid_argument(context +
                                        " backend presence does not match stage");
        }
        if (needs_backend) {
            primitive.launch_metadata->Validate();
            if ((*primitive.launch_metadata)->device != expected ||
                !primitive.kernel->IsReady() ||
                primitive.kernel->signature().get() !=
                    primitive.signature->get() ||
                primitive.kernel->launch_metadata().get() !=
                    primitive.launch_metadata->get()) {
                throw std::invalid_argument(context +
                                            " backend objects are inconsistent");
            }
        }
    }
}

CompileStage CompileResult::stage() const { return operator->()->stage_; }

std::string CompileResult::stage_name() const {
    switch (stage()) {
        case CompileStage::kValidated: return "validated";
        case CompileStage::kRelayOptimized: return "relay_optimized";
        case CompileStage::kLowered: return "lowered";
        case CompileStage::kTIROptimized: return "tir_optimized";
        case CompileStage::kSignatureBuilt: return "signature_built";
        case CompileStage::kBackendCompiled: return "backend_compiled";
    }
    throw std::runtime_error("CompileResult has an unknown stage");
}

Target CompileResult::target() const { return operator->()->target_; }

Function CompileResult::validated_relay() const {
    if (stage() != CompileStage::kValidated) {
        throw std::logic_error("validated_relay is visible only at validated stage");
    }
    return *operator->()->relay_;
}

Function CompileResult::optimized_relay() const {
    if (!HasReached(stage(), CompileStage::kRelayOptimized)) {
        throw std::logic_error("optimized_relay requires relay_optimized stage");
    }
    return *operator->()->relay_;
}

std::vector<PrimitiveCompileState> CompileResult::primitives() const {
    if (!HasReached(stage(), CompileStage::kLowered)) {
        throw std::logic_error("primitives require lowered stage");
    }
    return operator->()->primitives_;
}

runtime::ExecutablePlan CompileResult::plan() const {
    if (!HasReached(stage(), CompileStage::kLowered)) {
        throw std::logic_error("plan requires lowered stage");
    }
    return *operator->()->plan_;
}

Map<String, runtime::NDArray> CompileResult::constants() const {
    if (!HasReached(stage(), CompileStage::kLowered)) {
        throw std::logic_error("constants require lowered stage");
    }
    return CopyConstants(operator->()->constants_);
}

const CompileResultNode* CompileResult::operator->() const {
    const auto* node = As<CompileResultNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompileResult");
    return node;
}

}  // namespace kxc::api
