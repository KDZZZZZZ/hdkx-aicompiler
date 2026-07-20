/*! \file src/api/compile_result.cc
 * \brief 实现 CompileResult 的阶段校验、不可变复制和单向状态转移。
 */

#include "api/compile_result.h"

#include <stdexcept>
#include <utility>

namespace kxc::api {
namespace {

// 判断状态是否已经到达指定阶段，供只读访问器统一控制字段可见性。
bool HasReached(CompileStage current, CompileStage required) {
    return static_cast<int>(current) >= static_cast<int>(required);
}

// 强制状态转移只能前进一步，避免跳过签名或重复执行同一 pipeline 阶段。
void RequireStage(CompileStage actual, CompileStage expected, const char* operation) {
    if (actual != expected) {
        throw std::logic_error(std::string(operation) + " requires stage " +
                               std::to_string(static_cast<int>(expected)) +
                               ", actual stage is " +
                               std::to_string(static_cast<int>(actual)));
    }
}

// 校验 Relay 句柄确实包含 FunctionNode，避免继承构造器带入其他 Relay 节点。
void ValidateFunction(const Function& function, const char* field) {
    if (!function.defined() || !function.As<FunctionNode>()) {
        throw std::invalid_argument(std::string(field) + " must contain FunctionNode");
    }
    if (!function->body.defined()) {
        throw std::invalid_argument(std::string(field) + " must have a body");
    }
}

// 校验 TIR 句柄的动态节点类型和最小可编译结构。
void ValidatePrimFunc(const tir::PrimFunc& function, const char* field) {
    if (!function.defined() || !function.As<tir::PrimFuncNode>()) {
        throw std::invalid_argument(std::string(field) + " must contain PrimFuncNode");
    }
    if (!function->body.defined()) {
        throw std::invalid_argument(std::string(field) + " must have a body");
    }
}

// 深拷贝 Map 容器节点，隔离项目 Map 的共享可变实现。
Map<String, runtime::NDArray> CopyConstants(
    const Map<String, runtime::NDArray>& constants) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : constants) result.Set(item.first, item.second);
    return result;
}

// 常量 key 和 payload 在 Lower 阶段一经写入就必须完整可解释。
void ValidateConstants(const Map<String, runtime::NDArray>& constants) {
    for (const auto& item : constants) {
        if (std::string(item.first).empty()) {
            throw std::invalid_argument("CompileResult constant key must not be empty");
        }
        if (!item.second.defined() || !item.second.As<runtime::NDArrayNode>()) {
            throw std::invalid_argument(
                "CompileResult constant value must contain NDArrayNode");
        }
    }
}

// Target 是后续签名与后端选择的唯一设备事实来源。
void ValidateTarget(const Target& target) {
    if (!target.defined() || !target.As<TargetNode>()) {
        throw std::invalid_argument("CompileResult target must contain TargetNode");
    }
    if (target->device_type == kUnknown || target->device_id < 0 ||
        target->kind.empty()) {
        throw std::invalid_argument("CompileResult target identity is incomplete");
    }
    // Device 构造器统一执行 device type/id 的项目级合法性检查。
    (void)Device(target->device_type, target->device_id);
}

// 签名中的物理设备必须与全流程唯一 Target 一致。
void ValidateSignatureTarget(const codegen::KernelSignature& signature,
                             const Target& target) {
    signature.Validate();
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    if (arguments.empty()) {
        throw std::invalid_argument("CompileResult signature has no arguments");
    }
    const Device expected(target->device_type, target->device_id);
    if (arguments[0]->device != expected) {
        throw std::invalid_argument("CompileResult signature device does not match Target");
    }
}

}  // namespace

// 内部只接管完整节点；所有公开创建路径仍会调用 ValidateState。
CompileResult::CompileResult(CompileResultNode* node) : ObjectRef(node) {}

// Validate 是状态机唯一入口，不允许用空 Target 或非 Function Relay 起步。
CompileResult CompileResult::Validate(Target target, Function relay) {
    ValidateTarget(target);
    ValidateFunction(relay, "validated relay");
    auto* node = new CompileResultNode();
    node->stage_ = CompileStage::kValidated;
    node->target_ = std::move(target);
    node->relay_ = std::move(relay);
    CompileResult result(node);
    result.ValidateState();
    return result;
}

// 对象系统恢复路径必须重新检查内容，不能只验证节点 RTTI。
CompileResult::CompileResult(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompileResultNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompileResultNode");
    }
    if (defined()) ValidateState();
}

// Relay 优化只替换唯一 Relay 字段，其余事实从前一快照继承。
CompileResult CompileResult::AfterRelayOptimization(Function optimized_relay) const {
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

// Lowering 首次引入 TIR 和常量，二者必须在同一个原子状态转移中发布。
CompileResult CompileResult::AfterLowering(
    tir::PrimFunc lowered_tir,
    const Map<String, runtime::NDArray>& constants) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kRelayOptimized,
                 "AfterLowering");
    ValidatePrimFunc(lowered_tir, "lowered TIR");
    ValidateConstants(constants);
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kLowered;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->tir_ = std::move(lowered_tir);
    next->constants_ = CopyConstants(constants);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

// TIR 优化覆盖唯一 TIR 字段，不同时保留可能漂移的 lowered 副本。
CompileResult CompileResult::AfterTIROptimization(tir::PrimFunc optimized_tir) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kLowered,
                 "AfterTIROptimization");
    ValidatePrimFunc(optimized_tir, "optimized TIR");
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kTIROptimized;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->tir_ = std::move(optimized_tir);
    next->constants_ = CopyConstants(current->constants_);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

// Signature 必须匹配唯一 Target，且只在 TIR 已稳定后才能绑定。
CompileResult CompileResult::AfterSignature(
    codegen::KernelSignature signature) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kTIROptimized,
                 "AfterSignature");
    ValidateSignatureTarget(signature, current->target_);
    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kSignatureBuilt;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->tir_ = current->tir_;
    next->constants_ = CopyConstants(current->constants_);
    next->signature_ = std::move(signature);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

// Backend 发布必须一次性带齐 metadata 与 kernel，并共享同一签名/metadata 节点。
CompileResult CompileResult::AfterBackend(
    codegen::KernelLaunchMetadata launch_metadata,
    codegen::CompiledKernel kernel) const {
    const auto* current = operator->();
    RequireStage(current->stage_, CompileStage::kSignatureBuilt,
                 "AfterBackend");
    if (!launch_metadata.defined()) {
        throw std::invalid_argument("AfterBackend requires launch metadata");
    }
    launch_metadata.Validate();
    if (launch_metadata->device !=
        Device(current->target_->device_type, current->target_->device_id)) {
        throw std::invalid_argument("Backend metadata device does not match Target");
    }
    if (!kernel.defined() || !kernel.IsReady()) {
        throw std::invalid_argument("AfterBackend requires a ready CompiledKernel");
    }
    // 对象身份约束保证 Result、metadata 和 kernel 不会各自维护可漂移副本。
    if (kernel.signature().get() != current->signature_->get() ||
        kernel.launch_metadata().get() != launch_metadata.get()) {
        throw std::invalid_argument(
            "CompiledKernel must share CompileResult signature and metadata objects");
    }

    auto* next = new CompileResultNode();
    next->stage_ = CompileStage::kBackendCompiled;
    next->target_ = current->target_;
    next->relay_ = current->relay_;
    next->tir_ = current->tir_;
    next->constants_ = CopyConstants(current->constants_);
    next->signature_ = current->signature_;
    next->launch_metadata_ = std::move(launch_metadata);
    next->kernel_ = std::move(kernel);
    CompileResult result(next);
    result.ValidateState();
    return result;
}

// 每次状态转移和 ObjectRef 恢复都执行全量校验，尽早发现半初始化节点。
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

    const bool needs_tir = HasReached(node->stage_, CompileStage::kLowered);
    if (needs_tir != node->tir_.has_value()) {
        throw std::invalid_argument("CompileResult TIR presence does not match stage");
    }
    if (node->tir_) ValidatePrimFunc(*node->tir_, "CompileResult TIR");
    if (!needs_tir && node->constants_.size() != 0) {
        throw std::invalid_argument(
            "CompileResult constants cannot exist before lowering");
    }
    ValidateConstants(node->constants_);

    const bool needs_signature =
        HasReached(node->stage_, CompileStage::kSignatureBuilt);
    if (needs_signature != node->signature_.has_value()) {
        throw std::invalid_argument("CompileResult signature presence does not match stage");
    }
    if (node->signature_) ValidateSignatureTarget(*node->signature_, node->target_);

    const bool needs_backend =
        HasReached(node->stage_, CompileStage::kBackendCompiled);
    if (needs_backend != node->launch_metadata_.has_value() ||
        needs_backend != node->kernel_.has_value()) {
        throw std::invalid_argument("CompileResult backend fields do not match stage");
    }
    if (needs_backend) {
        node->launch_metadata_->Validate();
        const Device expected(node->target_->device_type, node->target_->device_id);
        if ((*node->launch_metadata_)->device != expected ||
            !node->kernel_->IsReady() ||
            node->kernel_->signature().get() != node->signature_->get() ||
            node->kernel_->launch_metadata().get() !=
                node->launch_metadata_->get()) {
            throw std::invalid_argument("CompileResult backend objects are inconsistent");
        }
    }
}

// 返回阶段枚举，不允许调用方直接修改节点状态。
CompileStage CompileResult::stage() const { return operator->()->stage_; }

// 使用稳定英文标识，便于日志、profiling 和测试比较。
std::string CompileResult::stage_name() const {
    switch (stage()) {
        case CompileStage::kValidated: return "validated";
        case CompileStage::kRelayOptimized: return "relay_optimized";
        case CompileStage::kLowered: return "lowered";
        case CompileStage::kTIROptimized: return "tir_optimized";
        case CompileStage::kSignatureBuilt: return "signature_built";
        case CompileStage::kBackendCompiled: return "backend_compiled";
    }
    throw std::runtime_error("CompileResult contains an unknown stage");
}

// Target 在所有阶段都存在，并始终返回同一不可变对象句柄。
Target CompileResult::target() const { return operator->()->target_; }

// Validate 阶段从结果本身提供 pipeline 输入，Compiler 不需要保留第二份外部事实。
Function CompileResult::validated_relay() const {
    const auto* node = operator->();
    if (node->stage_ != CompileStage::kValidated) {
        throw std::logic_error(
            "validated Relay is only available at the Validate stage");
    }
    return *node->relay_;
}

// 入口函数不冒充优化结果，只有 Relay pipeline 完成后才允许读取。
Function CompileResult::optimized_relay() const {
    const auto* node = operator->();
    if (!HasReached(node->stage_, CompileStage::kRelayOptimized)) {
        throw std::logic_error("optimized Relay is not available yet");
    }
    return *node->relay_;
}

// Lowered TIR 在完成下一阶段后已被替换，禁止把旧事实继续向后传播。
tir::PrimFunc CompileResult::lowered_tir() const {
    const auto* node = operator->();
    if (node->stage_ != CompileStage::kLowered) {
        throw std::logic_error("lowered TIR is only available at the Lower stage");
    }
    return *node->tir_;
}

// TIR pipeline 完成后，后续所有阶段共享同一个优化结果。
tir::PrimFunc CompileResult::optimized_tir() const {
    const auto* node = operator->();
    if (!HasReached(node->stage_, CompileStage::kTIROptimized)) {
        throw std::logic_error("optimized TIR is not available yet");
    }
    return *node->tir_;
}

// Lower 之前没有常量契约；之后返回深拷贝以隔离 Map::Set。
Map<String, runtime::NDArray> CompileResult::constants() const {
    const auto* node = operator->();
    if (!HasReached(node->stage_, CompileStage::kLowered)) {
        throw std::logic_error("constants are not available before lowering");
    }
    return CopyConstants(node->constants_);
}

// Signature 阶段之后只返回同一个已验证签名节点。
codegen::KernelSignature CompileResult::signature() const {
    const auto* node = operator->();
    if (!node->signature_) {
        throw std::logic_error("kernel signature is not available yet");
    }
    return *node->signature_;
}

// 启动元数据只能来自最终 CompiledKernel 的同源对象。
codegen::KernelLaunchMetadata CompileResult::launch_metadata() const {
    const auto* node = operator->();
    if (!node->launch_metadata_) {
        throw std::logic_error("launch metadata is not available yet");
    }
    return *node->launch_metadata_;
}

// 只有 Backend 阶段可以向 CompiledModule 暴露可执行资源。
codegen::CompiledKernel CompileResult::kernel() const {
    const auto* node = operator->();
    if (!node->kernel_) {
        throw std::logic_error("compiled kernel is not available yet");
    }
    return *node->kernel_;
}

// 所有访问均通过动态类型检查，undefined 或伪造句柄明确失败。
const CompileResultNode* CompileResult::operator->() const {
    const auto* node = As<CompileResultNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompileResult");
    return node;
}

}  // namespace kxc::api
