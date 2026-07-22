/*! \file src/compiler/internal/compile_state.h
 * \brief 定义 Compiler 各阶段之间传递的强类型、单向 CompileResult。
 */

#pragma once

#include <optional>
#include <string>

#include "kxc/runtime/ndarray.h"
#include "kxc/target/target.h"
#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/relay/relay.h"
#include "kxc/tir/stmt.h"

namespace kxc::api {

/*! \brief Compiler 已完成的最后一个阶段；枚举顺序就是唯一合法推进顺序。 */
enum class CompileStage : int {
    /*! \brief Target 与输入 Relay Function 已完成入口校验。 */
    kValidated = 0,
    /*! \brief Relay pass pipeline 已产出唯一的优化后 Relay。 */
    kRelayOptimized = 1,
    /*! \brief Lowering 已产出 TIR 和按 key 持有的常量 payload。 */
    kLowered = 2,
    /*! \brief TIR pass pipeline 已产出唯一的优化后 TIR。 */
    kTIROptimized = 3,
    /*! \brief 优化后 TIR 已绑定后端无关 KernelSignature。 */
    kSignatureBuilt = 4,
    /*! \brief 后端已产出启动元数据和可执行 CompiledKernel。 */
    kBackendCompiled = 5,
};

/*! \brief 保存单次编译当前阶段的不可分叉状态。 */
class CompileResultNode final : public Object {
public:
    KXC_OBJECT_DECLARE

private:
    friend class CompileResult;

    /*! \brief 当前最后完成的阶段，后续转移必须严格增加一。 */
    CompileStage stage_{CompileStage::kValidated};
    /*! \brief 全流程唯一的目标能力快照。 */
    Target target_;
    /*! \brief 当前 Relay 事实；Relay 优化阶段会用优化结果替换入口函数。 */
    std::optional<Function> relay_;
    /*! \brief 当前 TIR 事实；TIR 优化阶段会用优化结果替换 lowered TIR。 */
    std::optional<tir::PrimFunc> tir_;
    /*! \brief lowering 保活的常量表，后续阶段只继承而不重建。 */
    Map<String, runtime::NDArray> constants_;
    /*! \brief Signature 阶段建立并由后端产物共享的调用契约。 */
    std::optional<codegen::KernelSignature> signature_;
    /*! \brief Backend 阶段建立并与 CompiledKernel 共享的启动契约。 */
    std::optional<codegen::KernelLaunchMetadata> launch_metadata_;
    /*! \brief Backend 阶段产出的强类型可执行资源。 */
    std::optional<codegen::CompiledKernel> kernel_;
};


/*! \brief 以不可变快照表达 Compiler 的分阶段结果并约束状态转移。 */
class CompileResult : public ObjectRef {
public:
    /*! \brief 校验 Target 和入口 Relay Function，并创建唯一合法起点。 */
    static CompileResult Validate(Target target, Function relay);

    /*! \brief 从对象系统恢复 CompileResult，并重新验证完整阶段不变量。 */
    explicit CompileResult(const ObjectRef& ref);

    /*! \brief 仅从 Validate 阶段推进，替换为 pass pipeline 的优化后 Relay。 */
    CompileResult AfterRelayOptimization(Function optimized_relay) const;
    /*! \brief 仅从 Relay 阶段推进，写入 lowered TIR 与常量表。 */
    CompileResult AfterLowering(
        tir::PrimFunc lowered_tir,
        const Map<String, runtime::NDArray>& constants) const;
    /*! \brief 仅从 Lower 阶段推进，以优化后 TIR 替换 lowered TIR。 */
    CompileResult AfterTIROptimization(tir::PrimFunc optimized_tir) const;
    /*! \brief 仅从 TIR 阶段推进，绑定已校验的后端无关签名。 */
    CompileResult AfterSignature(codegen::KernelSignature signature) const;
    /*! \brief 仅从 Signature 阶段推进，绑定同源 metadata 与可执行 kernel。 */
    CompileResult AfterBackend(
        codegen::KernelLaunchMetadata launch_metadata,
        codegen::CompiledKernel kernel) const;

    /*! \brief 重新检查当前快照及所有已完成阶段的字段一致性。 */
    void ValidateState() const;
    /*! \brief 返回当前最后完成的编译阶段。 */
    CompileStage stage() const;
    /*! \brief 返回用于日志和诊断的稳定阶段名称。 */
    std::string stage_name() const;
    /*! \brief 返回全流程共享的 Target。 */
    Target target() const;
    /*! \brief 返回 Validate 阶段的入口 Relay，供 Relay pipeline 读取唯一输入。 */
    Function validated_relay() const;
    /*! \brief 返回优化后 Relay；Relay 阶段完成前明确失败。 */
    Function optimized_relay() const;
    /*! \brief 返回 Lower 阶段的 TIR；完成 TIR 优化后不再保留旧事实。 */
    tir::PrimFunc lowered_tir() const;
    /*! \brief 返回优化后 TIR；TIR 阶段完成前明确失败。 */
    tir::PrimFunc optimized_tir() const;
    /*! \brief 返回常量表的独立 Map，调用方修改不会影响编译快照。 */
    Map<String, runtime::NDArray> constants() const;
    /*! \brief 返回签名契约；Signature 阶段完成前明确失败。 */
    codegen::KernelSignature signature() const;
    /*! \brief 返回启动元数据；Backend 阶段完成前明确失败。 */
    codegen::KernelLaunchMetadata launch_metadata() const;
    /*! \brief 返回可执行内核；Backend 阶段完成前明确失败。 */
    codegen::CompiledKernel kernel() const;
    /*! \brief 返回经过运行时类型检查的只读节点。 */
    const CompileResultNode* operator->() const;

private:
    /*! \brief 内部构造器只接收已经完整填充的新快照节点。 */
    explicit CompileResult(CompileResultNode* node);
};

}  // namespace kxc::api
