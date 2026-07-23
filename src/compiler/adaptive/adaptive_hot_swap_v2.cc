/*! \file adaptive_hot_swap_v2.cc */
#include "kxc/compiler/adaptive_hot_swap_v2.h"
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc::api::adaptive::hot_swap::v2 {
namespace {
std::string Part(const std::string& x) { return std::to_string(x.size()) + ":" + x + ";"; }
std::string FlightKey(const ProductionCompileRequest& r) { return Part(r.artifact_key().canonical_bytes()) + Part(r.dispatch_key().canonical_bytes()) + Part(r.plan_abi().canonical_bytes()); }
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
    for (const auto& pin : c.compiled_graph().artifact_pins) {
        const uint64_t add = pin.handle().record().byte_size;
        if (add > std::numeric_limits<uint64_t>::max() - n) throw std::overflow_error("adaptive v2 producer byte sum overflow");
        n += add;
    }
    return n;
}
class CallbackScope final { public: explicit CallbackScope(std::atomic<size_t>& n) : n_(n) { n_.fetch_add(1, std::memory_order_acq_rel); } ~CallbackScope() { n_.fetch_sub(1, std::memory_order_release); } private: std::atomic<size_t>& n_; };
struct RunRetention final { std::shared_ptr<const GenerationLease> lease; };
class LocalValidation final : public CandidateValidationAuthority { public: ValidationReceipt Validate(const ProductionCompileRequest&, const CompiledGraph&) override { return IssueReceipt("process-local-structural-validation"); } };
class LocalGenerationAuthority final : public GenerationAuthority {
public:
    explicit LocalGenerationAuthority(Generation initial) : next_(initial) {}
    std::shared_ptr<const GenerationLease> Issue(const GenerationAuthorityRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_ == std::numeric_limits<Generation>::max()) throw std::overflow_error("adaptive v2 generation space is exhausted");
        return MakeLease(next_++, request);
    }
    Generation NextGenerationForTesting() const noexcept override { std::lock_guard<std::mutex> lock(mutex_); return next_; }
private: mutable std::mutex mutex_; Generation next_;
};
}  // namespace

struct CancellationToken::State final { std::atomic<bool> cancelled{false}; };
CancellationToken::CancellationToken(std::shared_ptr<State> state) : state_(std::move(state)) {}
bool CancellationToken::cancelled() const noexcept { return state_ && state_->cancelled.load(std::memory_order_acquire); }
CancellationSource::CancellationSource() : state_(std::make_shared<CancellationToken::State>()) {}
CancellationToken CancellationSource::token() const noexcept { return CancellationToken(state_); }
void CancellationSource::Cancel() noexcept { if (state_) state_->cancelled.store(true, std::memory_order_release); }
CompileError::CompileError(FailureCategory c, std::string s) : std::runtime_error(std::move(s)), category_(c) {}
FailureCategory CompileError::category() const noexcept { return category_; }
ValidationReceipt::ValidationReceipt(std::string value) : value_(std::move(value)) { if (value_.empty()) throw std::invalid_argument("validation receipt is empty"); }
const std::string& ValidationReceipt::value() const noexcept { return value_; }
ValidationReceipt CandidateValidationAuthority::IssueReceipt(std::string value) { return ValidationReceipt(std::move(value)); }
GenerationLease::GenerationLease(Generation g, std::shared_ptr<const FrozenPlanVariant> v, DispatchKey r, ArtifactKey a, PlanAbiFingerprint abi, std::string receipt, uint64_t bytes)
    : generation_(g), variant_(std::move(v)), route_(std::move(r)), selection_artifact_(std::move(a)), plan_abi_(std::move(abi)), validation_receipt_(std::move(receipt)), producer_reported_bytes_(bytes) {
    if (!generation_ || !variant_ || !route_.defined() || !selection_artifact_.defined() || !plan_abi_.defined() || validation_receipt_.empty()) throw std::invalid_argument("generation authority issued incomplete lease");
}
std::shared_ptr<const GenerationLease> GenerationAuthority::MakeLease(Generation g, const GenerationAuthorityRequest& r) {
    if (!r.candidate || r.selection_artifact != r.candidate->selection_artifact_key()) {
        throw std::invalid_argument("generation authority request selection mismatch");
    }
    const auto variant = legacy::FreezePreparedCandidate(g, r.route, r.plan_abi, r.candidate);
    return std::shared_ptr<const GenerationLease>(new GenerationLease(g, variant, r.route, r.selection_artifact, r.plan_abi, r.validation_receipt.value(), r.producer_reported_bytes));
}
Generation GenerationLease::generation() const noexcept { return generation_; }
const std::shared_ptr<const FrozenPlanVariant>& GenerationLease::variant() const noexcept { return variant_; }
const DispatchKey& GenerationLease::dispatch_key() const noexcept { return route_; }
const PlanAbiFingerprint& GenerationLease::plan_abi() const noexcept { return plan_abi_; }
const ArtifactKey& GenerationLease::selection_artifact_key() const noexcept { return selection_artifact_; }
const std::string& GenerationLease::validation_receipt() const noexcept { return validation_receipt_; }
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
    struct Waiter final { CancellationToken cancellation; std::chrono::steady_clock::time_point deadline; };
    struct Flight final {
        Flight(ProductionCompileRequest r, std::string k, std::string route, Waiter waiter) : request(std::move(r)), key(std::move(k)), route_key(std::move(route)), future(promise.get_future().share()) { waiters.push_back(std::move(waiter)); }
        ProductionCompileRequest request; std::string key, route_key; std::promise<CompileResult> promise; std::shared_future<CompileResult> future; std::vector<Waiter> waiters;
    };
    struct Route final { std::shared_ptr<const GenerationLease> current; std::deque<std::shared_ptr<const GenerationLease>> history; std::unordered_set<std::string> tombstones; bool compile_blocked{false}; };
    struct Negative final { Failure failure; std::chrono::steady_clock::time_point expires; std::list<std::string>::iterator order; };
    State(std::shared_ptr<ProductionPathCompilerAdapter> c, Options o) : compiler(std::move(c)), options(std::move(o)) {
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
            throw std::invalid_argument("adaptive v2 bounds are invalid");
        }
        validation = options.validation_authority ? options.validation_authority : std::make_shared<LocalValidation>();
        generations = options.generation_authority ? options.generation_authority : std::make_shared<LocalGenerationAuthority>(options.initial_generation);
    }
    ~State() { Stop(); }
    void Start() { for (size_t i=0;i<options.worker_count;++i) workers.emplace_back([this]{ Worker(); }); }
    void Stop() { { std::lock_guard<std::mutex> lock(mutex); stopping=true; } wake.notify_all(); for (auto& worker:workers) if (worker.joinable()) { if (worker.get_id()==std::this_thread::get_id()) worker.detach(); else worker.join(); } }
    void RejectReentry() const { if (callbacks.load(std::memory_order_acquire)) throw std::logic_error("adaptive v2 API called while callback is active"); }
    void Emit(Event e) const noexcept { if (!options.observer) return; try { CallbackScope s(callbacks); options.observer(e); } catch (...) {} }
    bool Live(const Flight& f) const { const auto now=std::chrono::steady_clock::now(); return std::any_of(f.waiters.begin(),f.waiters.end(),[&](const Waiter& w){ return !w.cancellation.cancelled() && now < w.deadline; }); }
    size_t LiveCount(const Flight& f) const { const auto now=std::chrono::steady_clock::now(); return static_cast<size_t>(std::count_if(f.waiters.begin(),f.waiters.end(),[&](const Waiter& w){ return !w.cancellation.cancelled() && now < w.deadline; })); }
    void Worker() { const auto keep=shared_from_this(); for (;;) { std::shared_ptr<Flight> f; { std::unique_lock<std::mutex> lock(mutex); wake.wait(lock,[&]{return stopping || !queue.empty();}); if (stopping && queue.empty()) return; f=queue.front(); queue.pop_front(); ++active; } Compile(f); { std::lock_guard<std::mutex> lock(mutex); --active; } wake.notify_all(); } }
    static bool Permanent(const Failure& f) { return f.category==FailureCategory::kPermanent || f.category==FailureCategory::kUnsupported; }
    void EraseNegative(const std::string& key) { auto it=negative.find(key); if(it==negative.end())return; negative_bytes-=it->second.failure.diagnostic.size(); negative_order.erase(it->second.order); negative.erase(it); }
    void CacheFailure(const std::string& key,const Failure& f,std::vector<Event>* events) {
        if(f.category==FailureCategory::kCancelled||f.category==FailureCategory::kBackpressure||f.retry_after<=std::chrono::milliseconds::zero()) return;
        while(negative.size()>=options.max_negative_cache_entries) { auto old=std::find_if(negative_order.begin(),negative_order.end(),[&](const std::string& k){return !Permanent(negative[k].failure);}); if(old==negative_order.end()) break; EraseNegative(*old); ++negative_evictions; events->push_back({EventKind::kNegativeEvicted}); }
        if(negative.size()>=options.max_negative_cache_entries) { ++negative_drops; if(Permanent(f)) { negative_blocked=true; events->push_back({EventKind::kNegativeCacheSaturated}); } return; }
        Failure c=f; const uint64_t room=options.max_negative_diagnostic_bytes-negative_bytes; if(c.diagnostic.size()>room)c.diagnostic.resize(room); negative_order.push_back(key); auto order=std::prev(negative_order.end()); const auto expiry=f.retry_after==std::chrono::milliseconds::max()?std::chrono::steady_clock::time_point::max():std::chrono::steady_clock::now()+f.retry_after; negative.emplace(key,Negative{std::move(c),expiry,order}); negative_bytes+=negative[key].failure.diagnostic.size();
    }
    void Finish(const std::shared_ptr<Flight>& f,CompileResult result,bool cache) { std::vector<Event> events; { std::lock_guard<std::mutex> lock(mutex); flights.erase(f->key); if(cache) CacheFailure(f->key,result.failure,&events); } f->promise.set_value(result); Event e; e.kind=result.ready()?EventKind::kPublished:EventKind::kRejected; if(!result.ready() && result.failure.diagnostic.find("global route") != std::string::npos) e.kind=EventKind::kRouteSaturated; if(!result.ready() && result.failure.diagnostic.find("tombstone") != std::string::npos) e.kind=EventKind::kTombstoneSaturated; e.dispatch_key_digest=f->request.dispatch_key().digest(); e.plan_abi_digest=f->request.plan_abi().digest(); e.diagnostic=result.ready()?"":result.failure.diagnostic; if(result.ready())e.generation=result.lease->generation(); Emit(std::move(e)); for(auto& x:events)Emit(std::move(x)); }
    void Inject(PublicationStage stage) const { if(options.fail_publication_stage && options.fail_publication_stage(stage)) throw std::runtime_error("injected adaptive v2 publication failure"); }
    void Metrics(const std::unordered_map<std::string,Route>& map,size_t* route_count,uint64_t* route_bytes,size_t* tombstones,uint64_t* tombstone_bytes) const { *route_count=map.size(); *route_bytes=0; *tombstones=0; *tombstone_bytes=0; for(const auto& [key,route]:map){*route_bytes+=key.size();*tombstones+=route.tombstones.size();for(const auto& a:route.tombstones)*tombstone_bytes+=a.size();} }
    void Evict(std::unordered_map<std::string,Route>* rs,std::deque<std::shared_ptr<const GenerationLease>>* ds,uint64_t* bytes,std::vector<Event>* ev) const { while(!ds->empty()&&(ds->size()>options.max_discoverable_generations||*bytes>options.max_producer_reported_bytes)){auto l=ds->front();ds->pop_front();*bytes-=l->producer_reported_bytes();auto it=rs->find(RouteKey(l->dispatch_key(),l->plan_abi()));if(it!=rs->end()){auto& h=it->second.history;h.erase(std::remove(h.begin(),h.end(),l),h.end());if(it->second.current==l)it->second.current.reset();if(!it->second.current&&h.empty()&&it->second.tombstones.empty())rs->erase(it);} Event e;e.kind=EventKind::kEvicted;e.generation=l->generation();ev->push_back(std::move(e));} }
    void Compile(const std::shared_ptr<Flight>& f) {
        try {
            { std::lock_guard<std::mutex> lock(mutex); if(!Live(*f)) { FinishUnlockedCancelled(f); return; } }
            CompiledGraph graph=compiler?compiler->Compile(f->request):Compiler::Compile(f->request.graph(),f->request.config());
            const ValidationReceipt receipt=validation->Validate(f->request,graph);
            const auto candidate=legacy::PrepareCandidate(f->request,std::move(graph),receipt.value());
            const uint64_t bytes=ProducerBytes(*candidate); Inject(PublicationStage::kByteBudget); if(bytes>options.max_producer_reported_bytes) throw CompileError(FailureCategory::kPermanent,"candidate exceeds adaptive v2 producer byte budget");
            std::shared_ptr<const GenerationLease> lease; Event published; std::vector<Event> evictions;
            { std::lock_guard<std::mutex> lock(mutex);
                if(!Live(*f)) { FinishUnlockedCancelled(f); return; }
                if(negative_blocked) throw CompileError(FailureCategory::kPermanent,"adaptive v2 permanent negative-cache capacity is fail-closed");
                auto staged=routes; auto staged_discoverable=discoverable; uint64_t staged_bytes=discoverable_bytes;
                Inject(PublicationStage::kRoute); auto route=staged.find(f->route_key); if(route==staged.end()){size_t rc;uint64_t rb;size_t tc;uint64_t tb;Metrics(staged,&rc,&rb,&tc,&tb);if(rc>=options.max_routes){++route_saturations;throw CompileError(FailureCategory::kPermanent,"adaptive v2 global route capacity is fail-closed");} route=staged.emplace(f->route_key,Route{}).first;}
                if(route->second.compile_blocked)throw CompileError(FailureCategory::kPermanent,"route compilation is fail-closed after tombstone saturation");
                Inject(PublicationStage::kQuarantine); if(route->second.tombstones.count(candidate->selection_artifact_key().canonical_bytes()))throw CompileError(FailureCategory::kPermanent,"candidate selection identity is quarantined");
                if(bytes>std::numeric_limits<uint64_t>::max()-staged_bytes)throw std::overflow_error("adaptive v2 discoverable byte accounting overflow");
                Inject(PublicationStage::kAuthority); Inject(PublicationStage::kMapAllocation);
                const Generation predecessor=route->second.current?route->second.current->generation():0;
                // All deterministic rejection seams run before the opaque
                // authority issues its monotonic lease. The authority owns the
                // only generation-minting step.
                size_t rc;uint64_t rb;size_t tc;uint64_t tb;Metrics(staged,&rc,&rb,&tc,&tb);if(rc>options.max_routes||rb>options.max_route_metadata_bytes||tc>options.max_quarantine_tombstones||tb>options.max_quarantine_tombstone_bytes){++route_saturations;throw CompileError(FailureCategory::kPermanent,"adaptive v2 global route metadata capacity is fail-closed");}
                Inject(PublicationStage::kFinalCommit); GenerationAuthorityRequest authority_request{f->request.dispatch_key(),candidate->selection_artifact_key(),f->request.plan_abi(),receipt,candidate,bytes}; lease=generations->Issue(authority_request);
                if(!lease||lease->dispatch_key()!=f->request.dispatch_key()||lease->plan_abi()!=f->request.plan_abi()||lease->selection_artifact_key()!=candidate->selection_artifact_key())throw CompileError(FailureCategory::kPermanent,"generation authority issued mismatched lease");
                route->second.current=lease;route->second.history.push_back(lease);staged_discoverable.push_back(lease);staged_bytes+=bytes;Evict(&staged,&staged_discoverable,&staged_bytes,&evictions);
                routes.swap(staged);discoverable.swap(staged_discoverable);discoverable_bytes=staged_bytes;this->evictions+=evictions.size();flights.erase(f->key);published.kind=EventKind::kPublished;published.generation=lease->generation();published.predecessor_generation=predecessor;published.producer_reported_bytes=bytes;published.dispatch_key_digest=lease->dispatch_key().digest();published.plan_abi_digest=lease->plan_abi().digest();
            }
            f->promise.set_value({lease,{}});Emit(std::move(published));for(auto& e:evictions)Emit(std::move(e));
        } catch(const std::exception& e) { Finish(f,Failed(FromException(e,options)),true); } catch(...) { Finish(f,Failed(Fail(FailureCategory::kTransient,"non-standard compilation failure",options.transient_backoff)),true); }
    }
    void FinishUnlockedCancelled(const std::shared_ptr<Flight>& f) { flights.erase(f->key); f->promise.set_value(Failed(Fail(FailureCategory::kCancelled,"all flight waiters cancelled or expired",{},false))); }
    std::shared_ptr<ProductionPathCompilerAdapter> compiler; const Options options; std::shared_ptr<CandidateValidationAuthority> validation; std::shared_ptr<GenerationAuthority> generations;
    mutable std::mutex mutex,health_mutex; std::condition_variable wake; std::deque<std::shared_ptr<Flight>> queue; std::unordered_map<std::string,std::shared_ptr<Flight>> flights; std::unordered_map<std::string,Route> routes; std::deque<std::shared_ptr<const GenerationLease>> discoverable; std::vector<std::thread> workers; mutable std::atomic<size_t> callbacks{0}; size_t active{0}; bool stopping{false}; uint64_t discoverable_bytes{0},evictions{0},merged{0},retries{0},negative_bytes{0},negative_evictions{0},negative_drops{0}; bool negative_blocked{false}; std::unordered_map<std::string,Negative> negative; std::list<std::string> negative_order; uint64_t quarantine_saturations{0},route_saturations{0},tombstone_saturations{0};
};

AdaptiveHotSwapController::AdaptiveHotSwapController(std::shared_ptr<ProductionPathCompilerAdapter> c,Options o):state_(std::make_shared<State>(std::move(c),std::move(o))){state_->Start();}
AdaptiveHotSwapController::~AdaptiveHotSwapController(){if(state_)state_->Stop();}
CompileTicket AdaptiveHotSwapController::Submit(CompileRequest request) {
    state_->RejectReentry();request.production.Validate();const std::string key=FlightKey(request.production),route=RouteKey(request.production.dispatch_key(),request.production.plan_abi());std::shared_future<CompileResult> future;Event event;{std::lock_guard<std::mutex> lock(state_->mutex);const auto now=std::chrono::steady_clock::now();auto immediate=[&](Failure f,EventKind k){std::promise<CompileResult> p;future=p.get_future().share();p.set_value(Failed(std::move(f)));event.kind=k;};if(request.cancellation.cancelled())immediate(Fail(FailureCategory::kCancelled,"waiter cancelled",{},false),EventKind::kCancelled);else if(now>=request.deadline)immediate(Fail(FailureCategory::kTimeout,"waiter deadline expired",{},true),EventKind::kCancelled);else if(auto it=state_->negative.find(key);it!=state_->negative.end()&&it->second.expires>now){immediate(it->second.failure,EventKind::kRetryCached);++state_->retries;}else {if(state_->negative.count(key))state_->EraseNegative(key);if(state_->negative_blocked)immediate(Fail(FailureCategory::kPermanent,"adaptive v2 permanent negative-cache capacity is fail-closed",{},false),EventKind::kRejected);else if(auto it=state_->flights.find(key);it!=state_->flights.end()){if(it->second->waiters.size()>=state_->options.max_waiters_per_flight)immediate(Fail(FailureCategory::kBackpressure,"adaptive v2 waiter budget is full",{},true),EventKind::kRejected);else{it->second->waiters.push_back({request.cancellation,request.deadline});future=it->second->future;++state_->merged;event.kind=EventKind::kMerged;}}else if(state_->queue.size()>=state_->options.max_queued_flights||state_->flights.size()>=state_->options.max_in_flight)immediate(Fail(FailureCategory::kBackpressure,"adaptive v2 queue or in-flight budget is full",{},true),EventKind::kRejected);else{auto f=std::make_shared<State::Flight>(request.production,key,route,State::Waiter{request.cancellation,request.deadline});future=f->future;state_->flights.emplace(key,f);state_->queue.push_back(std::move(f));event.kind=EventKind::kQueued;state_->wake.notify_one();}}}event.dispatch_key_digest=request.production.dispatch_key().digest();event.plan_abi_digest=request.production.plan_abi().digest();state_->Emit(std::move(event));return CompileTicket(std::move(future),request.deadline,std::move(request.cancellation));
}
std::shared_ptr<const GenerationLease> AdaptiveHotSwapController::CompileAndPublish(CompileRequest r){auto result=Submit(std::move(r)).Wait();if(result.ready())return result.lease;throw CompileError(result.failure.category,result.failure.diagnostic);}
std::shared_ptr<const GenerationLease> AdaptiveHotSwapController::Acquire(const ProductionExecutionRequest& r) const {state_->RejectReentry();r.Validate();std::shared_ptr<const GenerationLease> out;{std::lock_guard<std::mutex> lock(state_->mutex);auto it=state_->routes.find(RouteKey(r.dispatch_key(),r.plan_abi()));if(it!=state_->routes.end()&&it->second.current&&it->second.current->plan_abi()==r.plan_abi()&&!it->second.tombstones.count(it->second.current->selection_artifact_key().canonical_bytes()))out=it->second.current;}if(!out)throw std::out_of_range("no exact published adaptive v2 generation");return out;}
RunAsyncResult AdaptiveHotSwapController::RunAsync(const ProductionExecutionRequest& r,const Array<runtime::NDArray>& inputs,const DeviceStream& stream) const {auto lease=Acquire(r);auto result=lease->variant()->session()->RunAsync(inputs,stream);result.completion.RetainDependencies({},std::make_shared<RunRetention>(RunRetention{lease}));return {std::move(result.outputs),std::move(result.completion),std::move(lease)};}
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
            // publication cannot make the verified lease stale. Health
            // authorities must not re-enter this controller from Verify.
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
                            (*it)->selection_artifact_key().canonical_bytes()) == 0) {
                        predecessor = *it;
                        break;
                    }
                }

                auto staged = state_->routes;
                auto& route = staged.find(RouteKey(
                    lease->dispatch_key(), lease->plan_abi()))->second;
                const std::string artifact =
                    lease->selection_artifact_key().canonical_bytes();
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
                    ++state_->quarantine_saturations;
                    ++state_->tombstone_saturations;
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
Snapshot AdaptiveHotSwapController::SnapshotForTesting() const {state_->RejectReentry();std::lock_guard<std::mutex> lock(state_->mutex);size_t rc;uint64_t rb;size_t tc;uint64_t tb;state_->Metrics(state_->routes,&rc,&rb,&tc,&tb);size_t blocked=0;for(const auto&[_,r]:state_->routes)if(r.compile_blocked)++blocked;return {state_->generations->NextGenerationForTesting(),state_->queue.size(),state_->active,state_->discoverable.size(),state_->discoverable_bytes,state_->evictions,state_->merged,state_->retries,state_->negative.size(),state_->negative_bytes,state_->negative_evictions,state_->negative_drops,state_->negative_blocked,tc,blocked,state_->quarantine_saturations,rc,rb,tb,state_->route_saturations,state_->tombstone_saturations,false,false};}
void AdaptiveHotSwapController::ClearNegativeCacheForTesting(){state_->RejectReentry();std::lock_guard<std::mutex> lock(state_->mutex);state_->negative.clear();state_->negative_order.clear();state_->negative_bytes=0;state_->negative_blocked=false;}
void AdaptiveHotSwapController::ClearQuarantinesForTesting(){state_->RejectReentry();std::lock_guard<std::mutex> lock(state_->mutex);for(auto it=state_->routes.begin();it!=state_->routes.end();){it->second.tombstones.clear();it->second.compile_blocked=false;if(!it->second.current&&it->second.history.empty())it=state_->routes.erase(it);else ++it;}}
}  // namespace kxc::api::adaptive::hot_swap::v2
#endif
