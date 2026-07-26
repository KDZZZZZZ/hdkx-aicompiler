/*! \file adaptive_hot_swap.cc */
#include "kxc/compiler/adaptive_hot_swap.h"
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../internal/execution_contract.h"
#include "../internal/primitive_cache.h"
#include "../internal/primitive_compiler.h"
#include "../../runtime/internal/compiled_module_node.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/printer/print_ir.h"
#include "support/hash.h"

namespace kxc::api::adaptive::hot_swap {
namespace {
std::string Part(const std::string& x) { return std::to_string(x.size()) + ":" + x + ";"; }
std::string FlightKey(const ProductionCompileRequest& request) {
    const CompileConfig config = request.config();
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    std::string key =
        Part(request.graph_semantic_key().canonical_bytes()) +
        Part(request.shape_profile_key().canonical_bytes()) +
        Part(request.dispatch_key().canonical_bytes()) +
        Part(request.plan_abi().canonical_bytes()) +
        Part(contract.canonical_bytes) +
        Part(std::to_string(config->opt_level)) +
        Part(internal::CanonicalTargetSnapshot(config->target));
    for (const std::int64_t unit_id : request.requested_unit_ids()) {
        key += Part(std::to_string(unit_id));
    }
    for (const OrderedArtifactIdentity& artifact : request.ordered_artifacts()) {
        key += Part(artifact.CanonicalBytes());
    }
    return key;
}
std::string RouteKey(const DispatchKey& d, const PlanAbiFingerprint& a) { return Part(d.canonical_bytes()) + Part(a.canonical_bytes()); }
Failure Fail(FailureCategory c, std::string s, std::chrono::milliseconds retry = {}, bool retryable = true) { return {c, std::move(s), retry, retryable}; }
CompileResult Failed(Failure f) { return {nullptr, std::move(f)}; }
Failure FromException(const std::exception& e, const Options& o) {
    if (const auto* typed = dynamic_cast<const CompileError*>(&e)) {
        const bool permanent = typed->category() == FailureCategory::kPermanent || typed->category() == FailureCategory::kUnsupported;
        return Fail(typed->category(), typed->what(), permanent ? std::chrono::milliseconds::max() : typed->category() == FailureCategory::kTimeout ? o.timeout_backoff : o.transient_backoff, !permanent);
    }
    const bool permanent = dynamic_cast<const std::invalid_argument*>(&e) || dynamic_cast<const std::overflow_error*>(&e);
    return Fail(permanent ? FailureCategory::kPermanent : FailureCategory::kTransient, e.what(), permanent ? std::chrono::milliseconds::max() : o.transient_backoff, !permanent);
}
uint64_t ProducerBytes(const PreparedCandidate& c) {
    uint64_t n = 0;
    for (const auto& pin : c.compiled_graph().artifact_pins()) {
        const uint64_t add = pin.record().byte_size;
        if (add > std::numeric_limits<uint64_t>::max() - n) throw std::overflow_error("adaptive hot-swap producer byte sum overflow");
        n += add;
    }
    return n;
}
profiling::EventSpec MakeStageEvent(const char* stage,
                                    const CompileConfig& config) {
    profiling::EventSpec event;
    event.component = "compiler";
    event.event_type = "compile_stage";
    event.pass_name = stage;
    event.fields = profiling::MakeFields({
        {"stage", stage},
        {"target_kind", config->target->kind},
        {"device_type", std::to_string(static_cast<int>(config->target->device_type))},
        {"device_id", std::to_string(config->target->device_id)},
        {"opt_level", std::to_string(config->opt_level)},
    });
    return event;
}
void AddPrimitiveBatchFields(
    profiling::ScopedSpan* span,
    const internal::PreparedCompilerGraph& prepared,
    const internal::CompiledPrimitiveBatch& batch) {
    span->AddMetric("primitive_count",
                    static_cast<double>(batch.primitives.size()));
    size_t cache_hits = 0;
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        const internal::PrimitiveUnit& unit = prepared.graph.partitioned.units.at(
            static_cast<std::size_t>(primitive.unit_id));
        const std::string prefix = "unit." +
            std::to_string(primitive.unit_id) + ".";
        std::ostringstream stream;
        tir::printer::DumpPrimFunc(primitive.diagnostic_tir, stream);
        const std::string text = stream.str();
        const internal::CachedPrimitive& artifact = primitive.pin.artifact();
        span->AddField(prefix + "symbol", std::string(unit.symbol));
        span->AddField(
            prefix + "operator", std::string(unit.call.spec.name) + "@v" +
                std::to_string(unit.call.spec.schema_version));
        span->AddField(prefix + "ir_hash", support::HashText(text));
        span->AddMetric(prefix + "ir_bytes", static_cast<double>(text.size()));
        span->AddField(prefix + "backend",
                       artifact.launch_metadata->backend ==
                               codegen::CodeGenBackend::kLLVM
                           ? "llvm"
                           : "cuda");
        span->AddMetric(prefix + "cache_hit", primitive.cache_hit ? 1.0 : 0.0);
        if (primitive.cache_hit) ++cache_hits;
    }
    span->AddMetric("cache_hits", static_cast<double>(cache_hits));
    span->AddMetric(
        "cache_hit_rate", batch.primitives.empty()
                              ? 0.0
                              : static_cast<double>(cache_hits) /
                                    static_cast<double>(batch.primitives.size()));
}
bool SameOrderedArtifactKeys(const CompiledGraph& left,
                             const CompiledGraph& right) {
    const auto& left_pins = left.artifact_pins();
    const auto& right_pins = right.artifact_pins();
    if (left_pins.size() != right_pins.size()) return false;
    for (size_t index = 0; index < left_pins.size(); ++index) {
        if (left_pins[index].record().artifact_key !=
            right_pins[index].record().artifact_key) {
            return false;
        }
    }
    return true;
}
CompiledGraph CompileReplacement(const ProductionCompileRequest& request) {
    const CompileConfig config = request.config();
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    const internal::PreparedCompilerGraph prepared =
        internal::PrepareCompilerGraph(request.graph(), config, contract);
    const CompiledGraph& baseline = request.baseline_graph();
    if (prepared.graph.partitioned.units.size() !=
        baseline.artifact_pins().size()) {
        throw std::invalid_argument(
            "adaptive prepared graph primitive count differs from baseline");
    }
    std::vector<internal::PrimitiveUnitId> requested_unit_ids;
    requested_unit_ids.reserve(request.requested_unit_ids().size());
    for (const std::int64_t unit_id : request.requested_unit_ids()) {
        requested_unit_ids.push_back(unit_id);
    }
    profiling::ActivationScope activation(
        prepared.profile_context, prepared.profile_run_id);
    internal::CompiledPrimitiveBatch batch;
    {
        const PassContext pass_context = PassContext::MergeTarget(
            relay::PassContextFromRelay(
                prepared.graph.partitioned.value_graph.function),
            config->target);
        PassContext::Scope pass_scope(pass_context);
        profiling::ScopedSpan primitive_span(
            prepared.profile_context,
            MakeStageEvent("compile_primitives", config),
            prepared.profile_run_id);
        try {
            batch = internal::CompilePrimitiveUnits(
                prepared.graph.partitioned.units,
                prepared.graph.partitioned.value_graph.values, config, contract,
                requested_unit_ids);
            AddPrimitiveBatchFields(&primitive_span, prepared, batch);
        } catch (const std::exception& error) {
            primitive_span.SetStatus("error");
            primitive_span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'compile_primitives' failed: ") +
                error.what());
        }
    }
    std::vector<internal::PrimitiveArtifactPin> pins;
    pins.reserve(baseline.artifact_pins().size());
    for (const ArtifactPin& pin : baseline.artifact_pins()) {
        pins.push_back(internal::ArtifactPinAccess::Unwrap(pin));
    }
    for (const internal::CompiledPrimitive& replacement : batch.primitives) {
        pins.at(static_cast<size_t>(replacement.unit_id)) = replacement.pin;
    }
    CompiledGraph graph;
    {
        profiling::ScopedSpan assemble_span(
            prepared.profile_context, MakeStageEvent("assemble", config),
            prepared.profile_run_id);
        graph = internal::AssembleCompiledGraph(
            prepared, pins,
            internal::BorrowCompiledModuleConstants(baseline.module()));
        AddPrimitiveBatchFields(&assemble_span, prepared, batch);
    }
    if (prepared.profile_context) prepared.profile_context->Flush();
    return graph;
}
class CallbackScope final { public: explicit CallbackScope(std::atomic<size_t>& n) : n_(n) { n_.fetch_add(1, std::memory_order_acq_rel); } ~CallbackScope() { n_.fetch_sub(1, std::memory_order_release); } private: std::atomic<size_t>& n_; };
thread_local const void* locked_authority_controller = nullptr;
class LockedAuthorityScope final {
public:
    explicit LockedAuthorityScope(const void* controller) noexcept
        : previous_(locked_authority_controller) {
        locked_authority_controller = controller;
    }
    ~LockedAuthorityScope() { locked_authority_controller = previous_; }
private:
    const void* previous_;
};
struct RunRetention final { std::shared_ptr<const GenerationLease> lease; };
class LocalValidation final : public CandidateValidationAuthority { public: ValidationReceipt Validate(const ProductionCompileRequest&, const CompiledGraph&) override { return IssueReceipt("process-local-structural-validation"); } };
class LocalGenerationAuthority final : public GenerationAuthority {
public:
    explicit LocalGenerationAuthority(Generation initial) : next_(initial) {}
    std::shared_ptr<const GenerationLease> Issue(const GenerationAuthorityRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_ == std::numeric_limits<Generation>::max()) throw std::overflow_error("adaptive hot-swap generation space is exhausted");
        const auto lease = MakeLease(next_, request);
        ++next_;  // Allocation/freeze failure above never consumes a generation.
        return lease;
    }
private: mutable std::mutex mutex_; Generation next_;
};
}  // namespace

struct CancellationToken::State final {
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    uint64_t next_registration{1};
    std::unordered_map<uint64_t, std::function<void()>> callbacks;
};
CancellationToken::CancellationToken(std::shared_ptr<State> state) : state_(std::move(state)) {}
bool CancellationToken::cancelled() const noexcept { return state_ && state_->cancelled.load(std::memory_order_acquire); }
bool CancellationToken::RegisterControllerCallback(std::function<void()> callback, uint64_t* registration) const {
    if (registration) *registration = 0;
    if (!state_) return true;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->cancelled.load(std::memory_order_acquire)) return false;
    const uint64_t id = state_->next_registration++;
    state_->callbacks.emplace(id, std::move(callback));
    if (registration) *registration = id;
    return true;
}
void CancellationToken::UnregisterControllerCallback(uint64_t registration) const noexcept {
    if (!state_ || !registration) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->callbacks.erase(registration);
}
CancellationSource::CancellationSource() : state_(std::make_shared<CancellationToken::State>()) {}
CancellationToken CancellationSource::token() const noexcept { return CancellationToken(state_); }
void CancellationSource::Cancel() noexcept {
    if (!state_) return;
    std::unordered_map<uint64_t, std::function<void()>> callbacks;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->cancelled.exchange(true, std::memory_order_acq_rel)) return;
        callbacks.swap(state_->callbacks);
    }
    for (auto& [_, callback] : callbacks) try { callback(); } catch (...) {}
}
CompileError::CompileError(FailureCategory c, std::string s) : std::runtime_error(std::move(s)), category_(c) {}
FailureCategory CompileError::category() const noexcept { return category_; }
ValidationReceipt::ValidationReceipt(std::string value) : value_(std::move(value)) { if (value_.empty()) throw std::invalid_argument("validation receipt is empty"); }
const std::string& ValidationReceipt::value() const noexcept { return value_; }
ValidationReceipt CandidateValidationAuthority::IssueReceipt(std::string value) { return ValidationReceipt(std::move(value)); }
GenerationLease::GenerationLease(Generation g, std::shared_ptr<const PreparedCandidate> c, DispatchKey r, PlanAbiFingerprint abi, uint64_t bytes)
    : generation_(g), candidate_(std::move(c)), route_(std::move(r)), plan_abi_(std::move(abi)), producer_reported_bytes_(bytes) {
    if (!generation_ || !candidate_ || !route_.defined() || !plan_abi_.defined()) throw std::invalid_argument("generation authority issued incomplete lease");
}
std::shared_ptr<const GenerationLease> GenerationAuthority::MakeLease(Generation g, const GenerationAuthorityRequest& r) {
    if (!g || !r.candidate || !r.route.defined() || !r.plan_abi.defined() ||
        !r.selection_plan.defined() || r.validation_receipt.value().empty() ||
        !r.candidate->compiled_graph().defined() || !r.candidate->session() ||
        r.selection_plan != r.candidate->selection_plan_key() ||
        r.candidate->validation_receipt() != r.validation_receipt.value()) {
        throw std::invalid_argument("generation authority request selection or receipt mismatch");
    }
    return std::shared_ptr<const GenerationLease>(new GenerationLease(
        g, r.candidate, r.route, r.plan_abi, r.producer_reported_bytes));
}
Generation GenerationLease::generation() const noexcept { return generation_; }
const std::shared_ptr<const PreparedCandidate>& GenerationLease::candidate() const noexcept { return candidate_; }
const CompiledGraph& GenerationLease::compiled_graph() const noexcept { return candidate_->compiled_graph(); }
const std::shared_ptr<const runtime::RuntimeSession>& GenerationLease::session() const noexcept { return candidate_->session(); }
const DispatchKey& GenerationLease::dispatch_key() const noexcept { return route_; }
const PlanAbiFingerprint& GenerationLease::plan_abi() const noexcept { return plan_abi_; }
const PlanVariantKey& GenerationLease::selection_plan_key() const noexcept { return candidate_->selection_plan_key(); }
const std::string& GenerationLease::validation_receipt() const noexcept { return candidate_->validation_receipt(); }
uint64_t GenerationLease::producer_reported_bytes() const noexcept { return producer_reported_bytes_; }
CompileTicket::CompileTicket(std::shared_future<CompileResult> r, std::chrono::steady_clock::time_point d, CancellationToken c) : result_(std::move(r)), deadline_(d), cancellation_(std::move(c)) {}
bool CompileTicket::valid() const noexcept { return result_.valid(); }
CompileResult CompileTicket::Wait() const {
    if (!valid()) return Failed(Fail(FailureCategory::kPermanent, "invalid compile ticket", {}, false));
    for (;;) {
        if (cancellation_.cancelled()) return Failed(Fail(FailureCategory::kCancelled, "waiter cancelled", {}, false));
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline_) return Failed(Fail(FailureCategory::kTimeout, "waiter deadline expired", {}, true));
        if (result_.wait_for(std::min(std::chrono::milliseconds(1), std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - now))) == std::future_status::ready) return cancellation_.cancelled() ? Failed(Fail(FailureCategory::kCancelled, "waiter cancelled", {}, false)) : result_.get();
    }
}
std::future_status CompileTicket::WaitFor(std::chrono::milliseconds timeout) const { if (!valid()) return std::future_status::timeout; const auto end = std::min(deadline_, std::chrono::steady_clock::now() + timeout); while (!cancellation_.cancelled() && std::chrono::steady_clock::now() < end) if (result_.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) return std::future_status::ready; return std::future_status::timeout; }

class AdaptiveHotSwapController::State final : public std::enable_shared_from_this<State> {
public:
    struct Waiter final {
        explicit Waiter(CancellationToken c, std::chrono::steady_clock::time_point d)
            : cancellation(std::move(c)), deadline(d) {}
        ~Waiter() { cancellation.UnregisterControllerCallback(registration); }
        CancellationToken cancellation;
        std::chrono::steady_clock::time_point deadline;
        uint64_t registration{0};
        bool active{true};  // Protected by State::mutex.
    };
    struct Flight final {
        Flight(ProductionCompileRequest r, std::string k, std::string route)
            : request(std::move(r)), key(std::move(k)), route_key(std::move(route)), future(promise.get_future().share()) {}
        ProductionCompileRequest request; std::string key, route_key; std::promise<CompileResult> promise; std::shared_future<CompileResult> future; std::vector<std::shared_ptr<Waiter>> waiters;
    };
    struct Route final { std::shared_ptr<const GenerationLease> current; std::vector<std::shared_ptr<const GenerationLease>> history; std::unordered_set<std::string> tombstones; bool compile_blocked{false}; };
    struct Negative final { Failure failure; std::chrono::steady_clock::time_point expires; std::list<std::string>::iterator order; };
    State(Options o) : options(std::move(o)) {
        if (!options.initial_generation || !options.worker_count ||
            !options.max_queued_flights || !options.max_in_flight ||
            !options.max_waiters_per_flight ||
            !options.max_discoverable_generations ||
            !options.max_producer_reported_bytes ||
            !options.max_negative_cache_entries ||
            !options.max_negative_diagnostic_bytes ||
            !options.max_quarantine_tombstones_per_route ||
            !options.max_routes || !options.max_quarantine_tombstones ||
            !options.max_route_metadata_bytes ||
            !options.max_quarantine_tombstone_bytes ||
            options.max_queued_flights >
                std::numeric_limits<size_t>::max() - options.max_in_flight) {
            throw std::invalid_argument("adaptive hot-swap bounds are invalid");
        }
        validation = options.validation_authority ? options.validation_authority : std::make_shared<LocalValidation>();
        generations = options.generation_authority ? options.generation_authority : std::make_shared<LocalGenerationAuthority>(options.initial_generation);
    }
    ~State() { Stop(); }
    void Start() { for (size_t i=0;i<options.worker_count;++i) workers.emplace_back([this]{ Worker(); }); }
    void Stop() { { std::lock_guard<std::mutex> lock(mutex); stopping=true; } wake.notify_all(); for (auto& worker:workers) if (worker.joinable()) { if (worker.get_id()==std::this_thread::get_id()) worker.detach(); else worker.join(); } }
    void RejectReentry() const { if (callbacks.load(std::memory_order_acquire) || locked_authority_controller == this) throw std::logic_error("adaptive hot-swap API called while callback is active"); }
    void Emit(Event e) const noexcept { if (!options.observer) return; try { CallbackScope s(callbacks); options.observer(e); } catch (...) {} }
    bool Live(const Flight& f) const { const auto now=std::chrono::steady_clock::now(); return std::any_of(f.waiters.begin(),f.waiters.end(),[&](const std::shared_ptr<Waiter>& w){ return w->active && now < w->deadline; }); }
    size_t LiveCount(const Flight& f) const { const auto now=std::chrono::steady_clock::now(); return static_cast<size_t>(std::count_if(f.waiters.begin(),f.waiters.end(),[&](const std::shared_ptr<Waiter>& w){ return w->active && now < w->deadline; })); }
    bool AddWaiter(const std::shared_ptr<Flight>& f, const CancellationToken& cancellation,
                   std::chrono::steady_clock::time_point deadline) {
        auto waiter = std::make_shared<Waiter>(cancellation, deadline);
        f->waiters.push_back(waiter);
        const std::weak_ptr<State> state_weak = this->shared_from_this();
        const std::weak_ptr<Flight> flight_weak = f;
        const std::weak_ptr<Waiter> waiter_weak = waiter;
        try {
            if (!cancellation.RegisterControllerCallback([state_weak, flight_weak, waiter_weak] {
                    if (const auto state = state_weak.lock()) state->CancelWaiter(flight_weak, waiter_weak);
                }, &waiter->registration)) {
                f->waiters.pop_back();
                return false;
            }
            return true;
        } catch (...) {
            f->waiters.pop_back();
            throw;
        }
    }
    void CancelWaiter(const std::weak_ptr<Flight>& flight_weak,
                      const std::weak_ptr<Waiter>& waiter_weak) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        const auto flight = flight_weak.lock();
        const auto waiter = waiter_weak.lock();
        if (!flight || !waiter) return;
        waiter->active = false;
        flight->waiters.erase(std::remove(flight->waiters.begin(), flight->waiters.end(), waiter), flight->waiters.end());
        wake.notify_all();
    }
    void Worker() { const auto keep=shared_from_this(); for (;;) { std::shared_ptr<Flight> f; { std::unique_lock<std::mutex> lock(mutex); wake.wait(lock,[&]{return stopping || !queue.empty();}); if (stopping && queue.empty()) return; f=queue.front(); queue.pop_front(); } Compile(f); wake.notify_all(); } }
    static bool Permanent(const Failure& f) { return f.category==FailureCategory::kPermanent || f.category==FailureCategory::kUnsupported; }
    void EraseNegative(const std::string& key) { auto it=negative.find(key); if(it==negative.end())return; negative_bytes-=it->second.failure.diagnostic.size(); negative_order.erase(it->second.order); negative.erase(it); }
    void CacheFailure(const std::string& key,const Failure& f,std::vector<Event>* events) {
        if(f.category==FailureCategory::kCancelled||f.category==FailureCategory::kBackpressure||f.retry_after<=std::chrono::milliseconds::zero()) return;
        while(negative.size()>=options.max_negative_cache_entries) { auto old=std::find_if(negative_order.begin(),negative_order.end(),[&](const std::string& k){return !Permanent(negative[k].failure);}); if(old==negative_order.end()) break; EraseNegative(*old); events->push_back({EventKind::kNegativeEvicted}); }
        if(negative.size()>=options.max_negative_cache_entries) { if(Permanent(f)) { negative_blocked=true; events->push_back({EventKind::kNegativeCacheSaturated}); } return; }
        Failure c=f; const uint64_t room=options.max_negative_diagnostic_bytes-negative_bytes; if(c.diagnostic.size()>room)c.diagnostic.resize(room); negative_order.push_back(key); auto order=std::prev(negative_order.end()); const auto expiry=f.retry_after==std::chrono::milliseconds::max()?std::chrono::steady_clock::time_point::max():std::chrono::steady_clock::now()+f.retry_after; negative.emplace(key,Negative{std::move(c),expiry,order}); negative_bytes+=negative[key].failure.diagnostic.size();
    }
    void Finish(const std::shared_ptr<Flight>& f,CompileResult result,bool cache) { std::vector<Event> events; { std::lock_guard<std::mutex> lock(mutex); flights.erase(f->key); if(cache) CacheFailure(f->key,result.failure,&events); } f->promise.set_value(result); Event e; e.kind=result.ready()?EventKind::kPublished:EventKind::kRejected; if(!result.ready() && result.failure.diagnostic.find("global route") != std::string::npos) e.kind=EventKind::kRouteSaturated; if(!result.ready() && result.failure.diagnostic.find("tombstone") != std::string::npos) e.kind=EventKind::kTombstoneSaturated; e.dispatch_key_digest=f->request.dispatch_key().digest(); e.plan_abi_digest=f->request.plan_abi().digest(); e.diagnostic=result.ready()?"":result.failure.diagnostic; if(result.ready())e.generation=result.lease->generation(); Emit(std::move(e)); for(auto& x:events)Emit(std::move(x)); }
    void Metrics(const std::unordered_map<std::string,Route>& map,size_t* route_count,uint64_t* route_bytes,size_t* tombstones,uint64_t* tombstone_bytes) const { *route_count=map.size(); *route_bytes=0; *tombstones=0; *tombstone_bytes=0; for(const auto& [key,route]:map){*route_bytes+=key.size();*tombstones+=route.tombstones.size();for(const auto& a:route.tombstones)*tombstone_bytes+=a.size();} }
    void Evict(std::unordered_map<std::string,Route>* rs,std::vector<std::shared_ptr<const GenerationLease>>* ds,uint64_t* bytes,std::vector<Event>* ev) const noexcept { while(!ds->empty()&&(ds->size()>options.max_discoverable_generations||*bytes>options.max_producer_reported_bytes)){auto l=ds->front();ds->erase(ds->begin());*bytes-=l->producer_reported_bytes();auto it=std::find_if(rs->begin(),rs->end(),[&](const auto& entry){return entry.second.current==l || std::find(entry.second.history.begin(),entry.second.history.end(),l)!=entry.second.history.end();});if(it!=rs->end()){auto& h=it->second.history;h.erase(std::remove(h.begin(),h.end(),l),h.end());if(it->second.current==l)it->second.current.reset();if(!it->second.current&&h.empty()&&it->second.tombstones.empty())rs->erase(it);} ev->push_back(Event{EventKind::kEvicted,l->generation()});} }
    void Compile(const std::shared_ptr<Flight>& f) {
        try {
            { std::lock_guard<std::mutex> lock(mutex); if(!Live(*f)) { FinishUnlockedCancelled(f); return; } }
            CompiledGraph graph=CompileReplacement(f->request);
            const ValidationReceipt receipt=validation->Validate(f->request,graph);
            const auto candidate=preparation::PrepareCandidate(f->request,std::move(graph),receipt.value());
            const uint64_t bytes=ProducerBytes(*candidate); if(bytes>options.max_producer_reported_bytes) throw CompileError(FailureCategory::kPermanent,"candidate exceeds adaptive hot-swap producer byte budget");
            std::shared_ptr<const GenerationLease> lease;
            std::shared_ptr<const GenerationLease> no_op_lease;
            Event published;
            std::vector<Event> evictions;
            { std::lock_guard<std::mutex> lock(mutex);
                if(!Live(*f)) { FinishUnlockedCancelled(f); return; }
                if(negative_blocked) throw CompileError(FailureCategory::kPermanent,"adaptive hot-swap permanent negative-cache capacity is fail-closed");
                auto staged=routes; auto staged_discoverable=discoverable; uint64_t staged_bytes=discoverable_bytes;
                auto route=staged.find(f->route_key); if(route==staged.end()){size_t rc;uint64_t rb;size_t tc;uint64_t tb;Metrics(staged,&rc,&rb,&tc,&tb);if(rc>=options.max_routes)throw CompileError(FailureCategory::kPermanent,"adaptive hot-swap global route capacity is fail-closed"); route=staged.emplace(f->route_key,Route{}).first;}
                if(route->second.compile_blocked)throw CompileError(FailureCategory::kPermanent,"route compilation is fail-closed after tombstone saturation");
                if(route->second.tombstones.count(candidate->selection_plan_key().canonical_bytes()))throw CompileError(FailureCategory::kPermanent,"candidate selection identity is quarantined");
                if(!Live(*f)) { FinishUnlockedCancelled(f); return; }
                if (route->second.current && SameOrderedArtifactKeys(
                        route->second.current->compiled_graph(),
                        candidate->compiled_graph())) {
                    no_op_lease = route->second.current;
                    flights.erase(f->key);
                } else {
                    if(bytes>std::numeric_limits<uint64_t>::max()-staged_bytes)throw std::overflow_error("adaptive hot-swap discoverable byte accounting overflow");
                    const Generation predecessor=route->second.current?route->second.current->generation():0;
                    size_t rc;uint64_t rb;size_t tc;uint64_t tb;Metrics(staged,&rc,&rb,&tc,&tb);if(rc>options.max_routes||rb>options.max_route_metadata_bytes||tc>options.max_quarantine_tombstones||tb>options.max_quarantine_tombstone_bytes)throw CompileError(FailureCategory::kPermanent,"adaptive hot-swap global route metadata capacity is fail-closed");
                    // Every fallible container/event operation completes before Issue.
                    route->second.history.reserve(route->second.history.size()+1);
                    staged_discoverable.reserve(staged_discoverable.size()+1);
                    evictions.reserve(staged_discoverable.size()+1);
                    published.kind=EventKind::kPublished;
                    published.predecessor_generation=predecessor;
                    published.producer_reported_bytes=bytes;
                    published.dispatch_key_digest=f->request.dispatch_key().digest();
                    published.plan_abi_digest=f->request.plan_abi().digest();
                    // This is the final deadline/cancellation observation and the
                    // publication linearization point. Cancel callbacks take this
                    // same mutex, so either they remove demand first or commit wins.
                    if(!Live(*f)) { FinishUnlockedCancelled(f); return; }
                    GenerationAuthorityRequest authority_request{f->request.dispatch_key(),candidate->selection_plan_key(),f->request.plan_abi(),receipt,candidate,bytes};
                    LockedAuthorityScope issuance(this);
                    lease=generations->Issue(authority_request);
                    if(!lease || lease->generation() <= last_committed_generation ||
                       lease->dispatch_key()!=f->request.dispatch_key() || lease->plan_abi()!=f->request.plan_abi() ||
                       lease->selection_plan_key()!=candidate->selection_plan_key() ||
                       lease->validation_receipt()!=receipt.value() || lease->producer_reported_bytes()!=bytes ||
                       lease->candidate()!=candidate ||
                       lease->compiled_graph().graph_semantic_key()!=candidate->compiled_graph().graph_semantic_key() ||
                       lease->session()!=candidate->session()) throw CompileError(FailureCategory::kPermanent,"generation authority issued mismatched lease");
                    // No-throw handoff only: reserved vector assignments/erase/swap.
                    route->second.current=lease;route->second.history.push_back(lease);staged_discoverable.push_back(lease);staged_bytes+=bytes;Evict(&staged,&staged_discoverable,&staged_bytes,&evictions);
                    routes.swap(staged);discoverable.swap(staged_discoverable);discoverable_bytes=staged_bytes;last_committed_generation=lease->generation();flights.erase(f->key);published.generation=lease->generation();
                }
            }
            if (no_op_lease) {
                f->promise.set_value({no_op_lease,{}});
                return;
            }
            f->promise.set_value({lease,{}});Emit(std::move(published));for(auto& e:evictions)Emit(std::move(e));
        } catch(const std::exception& e) { Finish(f,Failed(FromException(e,options)),true); } catch(...) { Finish(f,Failed(Fail(FailureCategory::kTransient,"non-standard compilation failure",options.transient_backoff)),true); }
    }
    void FinishUnlockedCancelled(const std::shared_ptr<Flight>& f) { flights.erase(f->key); f->promise.set_value(Failed(Fail(FailureCategory::kCancelled,"all flight waiters cancelled or expired",{},false))); }
    const Options options; std::shared_ptr<CandidateValidationAuthority> validation; std::shared_ptr<GenerationAuthority> generations;
    mutable std::mutex mutex,health_mutex; std::condition_variable wake; std::deque<std::shared_ptr<Flight>> queue; std::unordered_map<std::string,std::shared_ptr<Flight>> flights; std::unordered_map<std::string,Route> routes; std::vector<std::shared_ptr<const GenerationLease>> discoverable; std::vector<std::thread> workers; mutable std::atomic<size_t> callbacks{0}; bool stopping{false}; uint64_t discoverable_bytes{0},negative_bytes{0}; bool negative_blocked{false}; std::unordered_map<std::string,Negative> negative; std::list<std::string> negative_order; Generation last_committed_generation{0};
};

AdaptiveHotSwapController::AdaptiveHotSwapController(Options o):state_(std::make_shared<State>(std::move(o))){state_->Start();}
AdaptiveHotSwapController::~AdaptiveHotSwapController(){if(state_)state_->Stop();}
CompileTicket AdaptiveHotSwapController::Submit(CompileRequest request) {
    state_->RejectReentry();request.production.Validate();const std::string key=FlightKey(request.production),route=RouteKey(request.production.dispatch_key(),request.production.plan_abi());std::shared_future<CompileResult> future;Event event;{std::lock_guard<std::mutex> lock(state_->mutex);const auto now=std::chrono::steady_clock::now();auto immediate=[&](Failure f,EventKind k){std::promise<CompileResult> p;future=p.get_future().share();p.set_value(Failed(std::move(f)));event.kind=k;};if(request.cancellation.cancelled())immediate(Fail(FailureCategory::kCancelled,"waiter cancelled",{},false),EventKind::kCancelled);else if(now>=request.deadline)immediate(Fail(FailureCategory::kTimeout,"waiter deadline expired",{},true),EventKind::kCancelled);else if(auto it=state_->negative.find(key);it!=state_->negative.end()&&it->second.expires>now){immediate(it->second.failure,EventKind::kRetryCached);}else {if(state_->negative.count(key))state_->EraseNegative(key);if(state_->negative_blocked)immediate(Fail(FailureCategory::kPermanent,"adaptive hot-swap permanent negative-cache capacity is fail-closed",{},false),EventKind::kRejected);else if(auto it=state_->flights.find(key);it!=state_->flights.end()){if(it->second->waiters.size()>=state_->options.max_waiters_per_flight)immediate(Fail(FailureCategory::kBackpressure,"adaptive hot-swap waiter budget is full",{},true),EventKind::kRejected);else if(state_->AddWaiter(it->second,request.cancellation,request.deadline)){future=it->second->future;event.kind=EventKind::kMerged;}else immediate(Fail(FailureCategory::kCancelled,"waiter cancelled",{},false),EventKind::kCancelled);}else if(state_->queue.size()>=state_->options.max_queued_flights||state_->flights.size()>=state_->options.max_in_flight)immediate(Fail(FailureCategory::kBackpressure,"adaptive hot-swap queue or in-flight budget is full",{},true),EventKind::kRejected);else{auto f=std::make_shared<State::Flight>(request.production,key,route);if(state_->AddWaiter(f,request.cancellation,request.deadline)){future=f->future;state_->flights.emplace(key,f);state_->queue.push_back(std::move(f));event.kind=EventKind::kQueued;state_->wake.notify_one();}else immediate(Fail(FailureCategory::kCancelled,"waiter cancelled",{},false),EventKind::kCancelled);}}}event.dispatch_key_digest=request.production.dispatch_key().digest();event.plan_abi_digest=request.production.plan_abi().digest();state_->Emit(std::move(event));return CompileTicket(std::move(future),request.deadline,std::move(request.cancellation));
}
std::shared_ptr<const GenerationLease> AdaptiveHotSwapController::CompileAndPublish(CompileRequest r){auto result=Submit(std::move(r)).Wait();if(result.ready())return result.lease;throw CompileError(result.failure.category,result.failure.diagnostic);}
std::shared_ptr<const GenerationLease> AdaptiveHotSwapController::Acquire(const ProductionExecutionRequest& r) const {state_->RejectReentry();r.Validate();std::shared_ptr<const GenerationLease> out;{std::lock_guard<std::mutex> lock(state_->mutex);auto it=state_->routes.find(RouteKey(r.dispatch_key(),r.plan_abi()));if(it!=state_->routes.end()&&it->second.current&&it->second.current->plan_abi()==r.plan_abi()&&!it->second.tombstones.count(it->second.current->selection_plan_key().canonical_bytes()))out=it->second.current;}if(!out)throw std::out_of_range("no exact published adaptive hot-swap generation");return out;}
RunAsyncResult AdaptiveHotSwapController::RunAsync(const ProductionExecutionRequest& r,const Array<runtime::NDArray>& inputs,const DeviceStream& stream) const {auto lease=Acquire(r);auto result=lease->session()->RunAsync(inputs,stream);result.completion.RetainDependencies({},std::make_shared<RunRetention>(RunRetention{lease}));return {std::move(result.outputs),std::move(result.completion),std::move(lease)};}
bool AdaptiveHotSwapController::EvaluateHealth(
    const std::shared_ptr<const GenerationLease>& lease) {
    state_->RejectReentry();
    if (!lease || !state_->options.health_authority) return false;

    HealthDecision decision;
    try {
        // Evaluation can run concurrently and does not hold routing state.
        decision = state_->options.health_authority->Evaluate(*lease);
    } catch (...) {
        return false;
    }

    std::shared_ptr<const GenerationLease> predecessor;
    bool accepted = false;
    bool quarantine = false;
    bool saturated = false;
    {
        std::lock_guard<std::mutex> serial(state_->health_mutex);
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->routes.find(
            RouteKey(lease->dispatch_key(), lease->plan_abi()));
        if (found != state_->routes.end() && found->second.current == lease) {
            // Verification consumes one-shot evidence under the route lock so
            // publication cannot make the verified lease stale. Same-controller
            // re-entry fails fast rather than deadlocking on this mutex.
            LockedAuthorityScope verification(state_.get());
            accepted = state_->options.health_authority->VerifyAndConsume(
                decision, *lease);
            quarantine = accepted &&
                decision.generation == lease->generation() &&
                decision.disposition == HealthDisposition::kQuarantine;
            if (quarantine) {
                for (auto it = found->second.history.rbegin();
                     it != found->second.history.rend(); ++it) {
                    if ((*it)->generation() < lease->generation() &&
                        found->second.tombstones.count(
                            (*it)->selection_plan_key().canonical_bytes()) == 0) {
                        predecessor = *it;
                        break;
                    }
                }

                auto staged = state_->routes;
                auto& route = staged.find(RouteKey(
                    lease->dispatch_key(), lease->plan_abi()))->second;
                const std::string artifact =
                    lease->selection_plan_key().canonical_bytes();
                size_t route_count;
                uint64_t route_bytes;
                size_t tombstone_count;
                uint64_t tombstone_bytes;
                state_->Metrics(staged, &route_count, &route_bytes,
                                &tombstone_count, &tombstone_bytes);
                if (route.tombstones.size() >=
                        state_->options.max_quarantine_tombstones_per_route ||
                    tombstone_count >= state_->options.max_quarantine_tombstones ||
                    artifact.size() >
                        state_->options.max_quarantine_tombstone_bytes ||
                    tombstone_bytes >
                        state_->options.max_quarantine_tombstone_bytes -
                            artifact.size()) {
                    route.compile_blocked = true;
                    saturated = true;
                } else {
                    route.tombstones.insert(artifact);
                }
                // Quarantine is durable even without a rollback target.
                route.current = predecessor;
                state_->routes.swap(staged);
            }
        }
    }

    Event health;
    health.kind = EventKind::kHealthDecision;
    health.generation = lease->generation();
    health.diagnostic = accepted ? decision.evidence_id : "health decision rejected";
    state_->Emit(std::move(health));
    if (!accepted || !quarantine) return accepted;

    Event quarantined;
    quarantined.kind = saturated ? EventKind::kQuarantineSaturated
                                 : EventKind::kQuarantined;
    quarantined.generation = lease->generation();
    quarantined.predecessor_generation =
        predecessor ? predecessor->generation() : 0;
    quarantined.diagnostic = decision.evidence_id;
    state_->Emit(std::move(quarantined));
    if (predecessor) {
        Event rollback;
        rollback.kind = EventKind::kRolledBack;
        rollback.generation = predecessor->generation();
        rollback.predecessor_generation = lease->generation();
        rollback.diagnostic = decision.evidence_id;
        state_->Emit(std::move(rollback));
    }
    return true;
}
}  // namespace kxc::api::adaptive::hot_swap
#endif
