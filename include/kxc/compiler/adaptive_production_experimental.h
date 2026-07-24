/*! \file include/kxc/compiler/adaptive_production_experimental.h
 * \brief Default-off experimental adapter for a production compiler path.
 *
 * This adapter is not production-ready: it has no cancellation, deadline,
 * negative-cache/retry authority, or automatic/one-shot health authority.
 */

#pragma once

#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 0
#endif

#if KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/identity.h"
#include "kxc/runtime/session.h"

// =============================================================================
// 轨 03 W2 — experimental production path（default-OFF）
// -----------------------------------------------------------------------------
// 无取消/deadline/negative-cache/自动 health；完整 lifecycle 见 hot_swap v2。
// CMake：KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION=ON（本头整段 #if 包住）
//
// 典型用法：
//   auto compiler = std::make_shared<MyProductionPathCompilerAdapter>();
//   AdaptiveController ctl(compiler, options);
//   ProductionCompileRequest creq(graph, config, expected_compiled_graph);
//   creq.Validate();
//   auto published = ctl.CompileAndPublish(creq);   // 编译+校验+原子发布
//   ProductionExecutionRequest ereq(creq.dispatch_key(), creq.plan_abi());
//   auto held = ctl.Acquire(ereq);                   // 不编译，只取当前 generation
//   auto run = ctl.RunAsync(ereq, inputs, stream);   // 冻结 variant 后提交 session
//   // run.completion / run.artifact_lease / run.variant 均需保活到完成
//
// 仅准备、不发布：
//   auto cand = PrepareCandidate(creq, compiled_graph, receipt);
//   auto frozen = FreezePreparedCandidate(gen, key, abi, cand);
// =============================================================================
namespace kxc::api::adaptive::experimental::production_path {

inline constexpr uint32_t kExperimentalProductionPathContractVersion = 1;

// ---------------------------------------------------------------------------
// ProductionCompileRequest — 整图 static-exact 编译强请求
// 用法：构造后 Validate()；config 深冻结，图 IR 构造后不可突变/竞态写
// ---------------------------------------------------------------------------
class ProductionCompileRequest final {
public:
    // 从 graph/config + 类型化 baseline 派生 artifact/dispatch/plan_abi 等身份
    ProductionCompileRequest(Function graph, CompileConfig config,
                             const CompiledGraph& expected_contract);

    const Function& graph() const noexcept;
    // 返回冻结 config 的独立深拷贝。
    CompileConfig config() const;
    const GraphSemanticKey& graph_semantic_key() const noexcept;
    const ShapeProfileKey& shape_profile_key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;       // 路由键
    const PlanAbiFingerprint& plan_abi() const noexcept;    // Plan ABI 指纹
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts() const noexcept;
    // 已验证 baseline pin，定义可调用 ABI。
    const std::vector<ArtifactPin>& verified_artifact_pins() const noexcept;
    void Validate() const;

private:
    Function graph_;
    // 一次深克隆；后续编译只读该冻结快照。
    CompileConfig config_;
    GraphSemanticKey graph_semantic_key_;
    ShapeProfileKey shape_profile_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    std::vector<OrderedArtifactIdentity> ordered_artifacts_;
    std::vector<ArtifactPin> verified_artifact_pins_;
};

// 精确路由请求：只带 DispatchKey+PlanAbi，不能带图、不能触发编译。
class ProductionExecutionRequest final {
public:
    ProductionExecutionRequest(DispatchKey dispatch_key,
                               PlanAbiFingerprint plan_abi);

    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    void Validate() const;

private:
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
};

// 外部生产编译器注入点：不同 exact key 可并发 Compile；
// 同 key 由 controller singleflight。实现须线程安全。
class ProductionPathCompilerAdapter {
public:
    virtual ~ProductionPathCompilerAdapter() = default;
    virtual CompiledGraph Compile(const ProductionCompileRequest& request) = 0;
};

// 仅准备结果：结构校验后的不可变候选，尚未路由/发布。
// receipt 是选择证据，不是 Plan ABI 兼容证明；无 generation/route 副作用。
class FrozenPlanVariant;
class PreparedCandidate final {
public:
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;
    const PlanVariantKey& selection_plan_key() const noexcept;
    const std::vector<OrderedArtifactIdentity>& selected_artifacts() const noexcept;
    const std::string& validation_receipt() const noexcept;

private:
    PreparedCandidate(CompiledGraph graph,
                      std::shared_ptr<const runtime::RuntimeSession> session,
                      PlanVariantKey selection_plan_key,
                      std::vector<OrderedArtifactIdentity> selected_artifacts,
                      std::string validation_receipt);
    CompiledGraph graph_;
    std::shared_ptr<const runtime::RuntimeSession> session_;
    PlanVariantKey selection_plan_key_;
    std::vector<OrderedArtifactIdentity> selected_artifacts_;
    std::string validation_receipt_;
    friend std::shared_ptr<const PreparedCandidate> PrepareCandidate(
        const ProductionCompileRequest&, CompiledGraph, std::string);
};

// 仅结构/类型化准备：不发布、不 mint generation。
std::shared_ptr<const PreparedCandidate> PrepareCandidate(
    const ProductionCompileRequest& request, CompiledGraph graph,
    std::string validation_receipt);

// 物化为不可变 PlanVariant；不改路由表。
std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
    uint64_t generation, DispatchKey dispatch_key, PlanAbiFingerprint plan_abi,
    std::shared_ptr<const PreparedCandidate> candidate);

// 产物 pin 的强所有权 lease：generation 绑定，保证运行中不被回收。
class ArtifactLease final {
public:
    ArtifactLease() = default;

    bool valid() const noexcept;
    uint64_t generation() const noexcept;
    const PlanVariantKey& selection_plan_key() const noexcept;
    const std::vector<ArtifactPin>& pins() const noexcept;

private:
    ArtifactLease(uint64_t generation, PlanVariantKey selection_plan_key,
                  CompiledGraph graph);

    uint64_t generation_{0};
    PlanVariantKey selection_plan_key_;
    CompiledGraph graph_;
    friend class AdaptiveController;
    friend std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
        uint64_t, DispatchKey, PlanAbiFingerprint,
        std::shared_ptr<const PreparedCandidate>);
};

// 一整代不可变 CompiledModule+ExecutablePlan+session。
class FrozenPlanVariant final {
public:
    uint64_t generation() const noexcept;
    const PlanVariantKey& key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const ArtifactLease& artifact_lease() const noexcept;
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;

private:
    FrozenPlanVariant(uint64_t generation, PlanVariantKey key,
                      DispatchKey dispatch_key,
                      PlanAbiFingerprint plan_abi,
                      ArtifactLease artifact_lease,
                      CompiledGraph compiled_graph,
                      std::shared_ptr<const runtime::RuntimeSession> session);

    uint64_t generation_{0};
    PlanVariantKey key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    ArtifactLease artifact_lease_;
    CompiledGraph compiled_graph_;
    std::shared_ptr<const runtime::RuntimeSession> session_;
    friend class AdaptiveController;
    friend std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
        uint64_t, DispatchKey, PlanAbiFingerprint,
        std::shared_ptr<const PreparedCandidate>);
};

// 可信控制面管理隔离请求：不是 runtime health 证明，也不是 one-shot authority。
// 调用方须先完成认证/授权/防重放/证据来源，再调 ForTrustedControlPlane。
class AdministrativeQuarantineRequest final {
public:
    static AdministrativeQuarantineRequest ForTrustedControlPlane(
        std::shared_ptr<const FrozenPlanVariant> variant,
        std::string reason);

    const std::shared_ptr<const FrozenPlanVariant>& variant() const noexcept;
    const std::string& reason() const noexcept;

private:
    AdministrativeQuarantineRequest(
        std::shared_ptr<const FrozenPlanVariant> variant,
        std::string reason);

    std::shared_ptr<const FrozenPlanVariant> variant_;
    std::string reason_;
};

// 控制器可观测事件种类。
enum class AdaptiveControllerEventKind : uint8_t {
    kCompileStarted = 0,
    kCompileMerged = 1,
    kValidated = 2,
    kPublished = 3,
    kAcquired = 4,
    kRunSubmitted = 5,
    kRejected = 6,
    kQuarantined = 7,
    kRolledBack = 8,
};

// 结构化观测字段；digest 仅诊断，不作相等判定。
struct AdaptiveControllerEvent final {
    AdaptiveControllerEventKind kind{AdaptiveControllerEventKind::kCompileStarted};
    uint64_t generation{0};
    uint64_t predecessor_generation{0};
    size_t in_flight_compiles{0};
    std::string graph_semantic_key_digest;
    std::string selection_plan_key_digest;
    std::string dispatch_key_digest;
    std::string plan_abi_digest;
    std::string plan_variant_digest;
    std::string diagnostic;
};

// 同步 best-effort 观察者：锁外执行，异常吞掉。
// 回调窗口内该 controller 所有 API 跨线程 fail-fast（含无关调用方）。
using AdaptiveControllerObserver =
    std::function<void(const AdaptiveControllerEvent&)>;

struct AdaptiveControllerOptions final {
    size_t max_in_flight_compiles{64};
    size_t max_slots{64};
    // 仅限制 controller 可发现历史，不含外部 lease / completion / device 字节 / cache pin。
    size_t max_discoverable_generations{8};
    AdaptiveControllerObserver observer;
};

// 原子隔离/回滚路由交接结果。
struct AdaptiveHandoffResult final {
    bool changed{false};
    uint64_t generation{0};
    uint64_t predecessor_generation{0};
    std::string diagnostic;
};

struct AdaptiveControllerSnapshot final {
    uint64_t next_generation{1};
    uint64_t compile_requests{0};
    uint64_t merged_compiles{0};
    uint64_t published{0};
    uint64_t acquired{0};
    uint64_t run_requests{0};
    uint64_t rejected{0};
    uint64_t quarantined{0};
    // 仅 controller 历史条目，不是总 live variant / resident 字节。
    size_t discoverable_history_variants{0};
    size_t in_flight_compiles{0};
};

// 异步运行结果：显式暴露 completion 与两个保活 owner（lease + variant）。
struct AdaptiveRunAsyncResult final {
    Array<runtime::NDArray> outputs;
    AsyncOperation completion;
    ArtifactLease artifact_lease;
    std::shared_ptr<const FrozenPlanVariant> variant;
};

// 上层 whole-plan 控制器；RuntimeSession 只做数据面。
class AdaptiveController final {
public:
    explicit AdaptiveController(
        std::shared_ptr<ProductionPathCompilerAdapter> compiler = nullptr,
        AdaptiveControllerOptions options = {});
    ~AdaptiveController();

    AdaptiveController(const AdaptiveController&) = delete;
    AdaptiveController& operator=(const AdaptiveController&) = delete;

    // 编译 + 校验 + 原子发布一整图。
    std::shared_ptr<const FrozenPlanVariant> CompileAndPublish(
        const ProductionCompileRequest& request);

    // 不编译，只获取当前 exact generation。
    std::shared_ptr<const FrozenPlanVariant> Acquire(
        const ProductionExecutionRequest& request) const;

    // 入口冻结某一 exact variant，仅提交该 session。
    AdaptiveRunAsyncResult RunAsync(
        const ProductionExecutionRequest& request,
        const Array<runtime::NDArray>& inputs,
        const DeviceStream& stream) const;

    // 可信管理隔离 + 未来路由回滚。
    AdaptiveHandoffResult RollbackAdministrative(
        const AdministrativeQuarantineRequest& quarantine,
        uint64_t target_generation = 0);

    AdaptiveControllerSnapshot Snapshot() const;

private:
    class State;
    std::shared_ptr<State> state_;
};

}  // namespace kxc::api::adaptive::experimental::production_path

#endif  // KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
