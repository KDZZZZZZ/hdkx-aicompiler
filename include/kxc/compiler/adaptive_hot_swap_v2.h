/*! \file adaptive_hot_swap_v2.h
 * \brief Default-OFF v2 whole-plan adaptive lifecycle control plane.
 *
 * This API is process-local experimental control-plane evidence only.  The
 * default authority below is test authority, not authentication or attestation.
 */
#pragma once

#ifndef KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#define KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 0
#endif

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 1
#elif !KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#error "KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 requires preparation contracts"
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include "kxc/compiler/adaptive_production_experimental.h"

// =============================================================================
// 轨 03 W3 — AdaptiveHotSwapController v2（default-OFF，依赖 preparation contracts）
// -----------------------------------------------------------------------------
// 进程内 experimental 控制面；内建 Generation/Health authority 是测试权威，
// 不是认证/attestation。CMake：KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2=ON
//
// 典型用法：
//   CancellationSource cancel;
//   CompileRequest req{production_req, deadline, cancel.token()};
//   AdaptiveHotSwapController ctl(compiler, options);
//   CompileTicket ticket = ctl.Submit(req);           // 异步入队 + singleflight
//   CompileResult r = ticket.Wait();                  // 或 CompileAndPublish 同步
//   if (r.ready()) {
//     auto lease = r.lease;                           // GenerationLease 保活 pin
//     auto acquired = ctl.Acquire(exec_req);          // 仅路由获取
//     auto run = ctl.RunAsync(exec_req, inputs, stream);
//     ctl.EvaluateHealth(lease);                      // 可选 one-shot health
//   }
//   cancel.Cancel();  // 只结束本 waiter，不污染共享 flight
//
// 发布条件：DispatchKey + PlanAbiFingerprint 均匹配才可换 future routing
// =============================================================================
namespace kxc::api::adaptive::hot_swap::v2 {
namespace preparation = experimental::production_path;
using ProductionCompileRequest = preparation::ProductionCompileRequest;
using ProductionExecutionRequest = preparation::ProductionExecutionRequest;
using ProductionPathCompilerAdapter = preparation::ProductionPathCompilerAdapter;
using PreparedCandidate = preparation::PreparedCandidate;
inline constexpr uint32_t kAdaptiveHotSwapContractVersion = 3;
using Generation = uint64_t;  // 不回绕的代际号

// 协作式取消令牌：cancel 只结束该 waiter，不杀共享 flight。
class CancellationToken final {
public:
    CancellationToken() = default;
    bool cancelled() const noexcept;
    // 仅 controller 注册回调；Cancel 在释放 state mutex 后同步调用。
    // 返回 false 表示取消已先发生，注册失败。
    bool RegisterControllerCallback(std::function<void()> callback,
                                    uint64_t* registration) const;
    void UnregisterControllerCallback(uint64_t registration) const noexcept;
private:
    struct State;
    explicit CancellationToken(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
    friend class CancellationSource;
};
// 取消源：持有共享 State，可发出 Cancel。
class CancellationSource final {
public:
    CancellationSource();
    CancellationToken token() const noexcept;
    void Cancel() noexcept;
private:
    std::shared_ptr<CancellationToken::State> state_;
};

// 失败分类：permanent/unsupported 进 negative cache 且不过期；
// transient/timeout 可退避；cancel/backpressure 不进 negative cache。
enum class FailureCategory : uint8_t { kPermanent, kTransient, kCancelled, kTimeout, kUnsupported, kBackpressure };
struct Failure final {
    FailureCategory category{FailureCategory::kTransient};
    std::string diagnostic;
    std::chrono::milliseconds retry_after{0};
    bool retryable{true};
};
class CompileError final : public std::runtime_error {
public:
    CompileError(FailureCategory category, std::string diagnostic);
    FailureCategory category() const noexcept;
private:
    FailureCategory category_;
};
// 带 deadline/cancellation 的编译请求包装。
struct CompileRequest final {
    ProductionCompileRequest production;
    std::chrono::steady_clock::time_point deadline{std::chrono::steady_clock::time_point::max()};
    CancellationToken cancellation;
};

// 不可伪造的校验回执：仅 CandidateValidationAuthority 可签发。
class ValidationReceipt final {
public:
    const std::string& value() const noexcept;
private:
    explicit ValidationReceipt(std::string value);
    std::string value_;
    friend class CandidateValidationAuthority;
};
// 结构准备后的策略校验注入点；不同 flight 可并发，须线程安全。
class CandidateValidationAuthority {
public:
    virtual ~CandidateValidationAuthority() = default;
    virtual ValidationReceipt Validate(const ProductionCompileRequest& request,
                                       const CompiledGraph& graph) = 0;
protected:
    static ValidationReceipt IssueReceipt(std::string value);
};

class GenerationLease;
// 向 GenerationAuthority 申请 lease 的打包请求。
struct GenerationAuthorityRequest final {
    const DispatchKey& route;
    const PlanVariantKey& selection_plan;
    const PlanAbiFingerprint& plan_abi;
    const ValidationReceipt& validation_receipt;
    std::shared_ptr<const PreparedCandidate> candidate;
    uint64_t producer_reported_bytes{0};  // 生产者声明字节，非 device resident
};
// 签发不透明单调 lease：controller 不能私自 mint 内部字段。
// Issue 在 publication mutex 下串行；同 controller reentry fail-fast。
// 内建 authority 是进程内测试证据，不是认证。
class GenerationAuthority {
public:
    virtual ~GenerationAuthority() = default;
    virtual std::shared_ptr<const GenerationLease> Issue(
        const GenerationAuthorityRequest& request) = 0;
    virtual Generation NextGenerationForTesting() const noexcept = 0;
protected:
    static std::shared_ptr<const GenerationLease> MakeLease(
        Generation generation, const GenerationAuthorityRequest& request);
};

// 不透明 lease：绑定不可变 candidate/pin/receipt/route/ABI。
class GenerationLease final {
public:
    Generation generation() const noexcept;
    const std::shared_ptr<const PreparedCandidate>& candidate() const noexcept;
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const PlanVariantKey& selection_plan_key() const noexcept;
    const std::string& validation_receipt() const noexcept;
    uint64_t producer_reported_bytes() const noexcept;
private:
    GenerationLease(Generation generation,
                    std::shared_ptr<const PreparedCandidate> candidate,
                    DispatchKey route, PlanAbiFingerprint plan_abi,
                    uint64_t producer_reported_bytes);
    Generation generation_{0};
    std::shared_ptr<const PreparedCandidate> candidate_;
    DispatchKey route_;
    PlanAbiFingerprint plan_abi_;
    uint64_t producer_reported_bytes_{0};
    friend class GenerationAuthority;
};

// 编译结果：成功则 lease 非空。
struct CompileResult final {
    std::shared_ptr<const GenerationLease> lease;
    Failure failure;
    bool ready() const noexcept { return lease != nullptr; }
};
// 异步编译票：Wait/WaitFor 取结果；绑定本 ticket 的 deadline/cancel。
class CompileTicket final {
public:
    CompileTicket() = default;
    bool valid() const noexcept;
    CompileResult Wait() const;
    std::future_status WaitFor(std::chrono::milliseconds timeout) const;
private:
    CompileTicket(std::shared_future<CompileResult> result, std::chrono::steady_clock::time_point deadline,
                  CancellationToken cancellation);
    std::shared_future<CompileResult> result_;
    std::chrono::steady_clock::time_point deadline_;
    CancellationToken cancellation_;
    friend class AdaptiveHotSwapController;
};

enum class HealthDisposition : uint8_t { kHealthy, kQuarantine };
// health 决策：含 evidence_id / replay_token，供 one-shot 消费。
struct HealthDecision final {
    Generation generation{0};
    HealthDisposition disposition{HealthDisposition::kHealthy};
    std::string evidence_id;
    std::string replay_token;
};
// Evaluate 可在 v2 锁外并发；VerifyAndConsume 在 route/health 锁下串行、
// 仅当 lease 仍是 route head 时生效，且 noexcept；同 controller reentry fail-fast。
class HealthAuthority {
public:
    virtual ~HealthAuthority() = default;
    virtual HealthDecision Evaluate(const GenerationLease& lease) = 0;
    virtual bool VerifyAndConsume(const HealthDecision& decision, const GenerationLease& lease) noexcept = 0;
};

// 发布事务阶段（测试可注入 fail seam，且在改路由前检查）。
enum class PublicationStage : uint8_t {
    kByteBudget, kRoute, kQuarantine, kAuthority, kMapAllocation, kFinalCommit
};
enum class EventKind : uint8_t {
    kQueued, kMerged, kPublished, kCancelled, kRetryCached, kEvicted,
    kNegativeEvicted, kNegativeCacheSaturated, kHealthDecision, kQuarantined,
    kQuarantineSaturated, kRolledBack, kRejected, kRouteSaturated, kTombstoneSaturated
};
struct Event final {
    EventKind kind{EventKind::kQueued};
    Generation generation{0};
    Generation predecessor_generation{0};
    uint64_t producer_reported_bytes{0};
    std::chrono::milliseconds retry_after{0};
    std::string dispatch_key_digest;
    std::string plan_abi_digest;
    std::string diagnostic;
};
using Observer = std::function<void(const Event&)>;

struct Options final {
    Generation initial_generation{1};
    size_t worker_count{2}, max_queued_flights{64}, max_in_flight{64}, max_waiters_per_flight{1024};
    size_t max_discoverable_generations{8};
    uint64_t max_producer_reported_bytes{64ULL * 1024ULL * 1024ULL};
    size_t max_negative_cache_entries{64};
    uint64_t max_negative_diagnostic_bytes{64ULL * 1024ULL * 1024ULL};
    size_t max_quarantine_tombstones_per_route{64};
    size_t max_routes{64}, max_quarantine_tombstones{256};
    uint64_t max_route_metadata_bytes{256ULL * 1024ULL}, max_quarantine_tombstone_bytes{256ULL * 1024ULL};
    std::chrono::milliseconds transient_backoff{100}, timeout_backoff{100};
    Observer observer;
    std::shared_ptr<HealthAuthority> health_authority;
    std::shared_ptr<CandidateValidationAuthority> validation_authority;
    std::shared_ptr<GenerationAuthority> generation_authority;
    // 测试 seam：抛错/失败阶段在任何路由变更前检查。
    std::function<bool(PublicationStage)> fail_publication_stage;
};
// 取消在拿到 publication mutex 的 controller 回调时线性化：
// 先线性化的 cancel 抑制发布，否则 commit 胜出。
// deadline 是观察点不是定时器：admission/worker entry/final commit 检查；
// 最终观察后过期可能与已发布 generation 共存。
struct Snapshot final {
    Generation next_generation{1};
    size_t queued_flights{0}, active_flights{0}, discoverable_generations{0};
    uint64_t producer_reported_discoverable_bytes{0}, evictions{0}, merged_waiters{0}, retry_cached{0};
    size_t negative_cache_entries{0};
    uint64_t negative_cache_diagnostic_bytes{0}, negative_cache_evictions{0}, negative_cache_drops{0};
    bool negative_cache_compile_blocked{false};
    size_t quarantine_tombstones{0}, quarantine_compile_blocked_routes{0};
    uint64_t quarantine_saturations{0};
    size_t routes{0};
    uint64_t route_metadata_bytes{0}, quarantine_tombstone_bytes{0}, route_saturations{0}, tombstone_saturations{0};
    bool process_resident_bytes_known{false}, device_resident_bytes_known{false};
};
struct RunAsyncResult final {
    Array<runtime::NDArray> outputs;
    AsyncOperation completion;
    std::shared_ptr<const GenerationLease> lease;
};

// ---------------------------------------------------------------------------
// AdaptiveHotSwapController — queue / singleflight / lease / health / quarantine
// Submit：异步；CompileAndPublish：同步等到发布或失败
// Acquire/RunAsync：不触发编译；health 仅当 lease 仍是 route head 时可消费
// ---------------------------------------------------------------------------
class AdaptiveHotSwapController final {
public:
    explicit AdaptiveHotSwapController(std::shared_ptr<ProductionPathCompilerAdapter> compiler = nullptr, Options options = {});
    ~AdaptiveHotSwapController();
    AdaptiveHotSwapController(const AdaptiveHotSwapController&) = delete;
    AdaptiveHotSwapController& operator=(const AdaptiveHotSwapController&) = delete;
    // 异步入队；同 key singleflight，返回本 waiter 的 ticket
    CompileTicket Submit(CompileRequest request);
    // 同步编译+发布；成功返回 GenerationLease
    std::shared_ptr<const GenerationLease> CompileAndPublish(CompileRequest request);
    // 仅按 DispatchKey+PlanAbi 取当前路由 head
    std::shared_ptr<const GenerationLease> Acquire(const ProductionExecutionRequest& request) const;
    // 冻结 lease 对应 session 并提交；completion 与 lease 一并返回保活
    RunAsyncResult RunAsync(const ProductionExecutionRequest& request, const Array<runtime::NDArray>& inputs, const DeviceStream& stream) const;
    // Evaluate + VerifyAndConsume；过时/已消费返回 false
    bool EvaluateHealth(const std::shared_ptr<const GenerationLease>& lease);
    void ClearNegativeCacheForTesting();
    void ClearQuarantinesForTesting();
    Snapshot SnapshotForTesting() const;
private:
    class State; std::shared_ptr<State> state_;
};
}  // namespace kxc::api::adaptive::hot_swap::v2
#endif
