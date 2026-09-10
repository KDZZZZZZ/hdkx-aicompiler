// CPU/LLVM execution and completed Profile Bundle evidence for adaptive routing.
#include <algorithm>
#include <filesystem>
#include <condition_variable>
#include <fstream>
#include <future>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>
#include "kxc/compiler/adaptive_hot_swap.h"
#include "kxc/relay/op.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "../src/compiler/internal/compiled_graph_access.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/compiled_module_node.h"
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
#include "support/request_batching_graph.h"
#endif
#if KXC_USE_LLVM
#include <llvm/Support/JSON.h>
#include <llvm/Support/Error.h>

namespace {
using namespace kxc;
namespace hs = api::adaptive::hot_swap;
namespace ci = api::internal;
using Request = hs::ProductionCompileRequest;
using Execution = hs::ProductionExecutionRequest;
void Check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void Rejects(F f) { try { f(); } catch (const std::exception&) { return; } throw std::runtime_error("expected rejection"); }
runtime::NDArray Tensor(const std::vector<float>& data) {
    auto out = runtime::NDArray::Empty({static_cast<int64_t>(data.size())},runtime::DataTypeFromString("float32"),Device::CPU());
    out.CopyFromBytes(data.data(),out.NBytes()); return out;
}
std::vector<float> Read(const runtime::NDArray& value) {
    std::vector<float> out(value.NBytes()/4); value.CopyToBytes(out.data(),value.NBytes()); return out;
}
Function Graph(int64_t length=2) {
    Var a("a",TensorType({length},"float32")), b("b",TensorType({length},"float32"));
    return Function({a,b},Call(relay::Op::Get("mul"),{Call(relay::Op::Get("add"),{a,b}),b}));
}
api::CompileConfig Config(int level, const std::string& name, bool enabled = true) {
    profiling::ProfileOptions options; options.enabled=enabled; options.record_pass_ir=false;
    options.ir_capture_mode=profiling::IRCaptureMode::kDisabled;
    options.bundle_dir=(std::filesystem::current_path()/"out"/"adaptive_runtime"/name).string();
    return api::CompileConfig::Create(BuildTarget(Device::CPU()),level,options);
}
std::string String(const llvm::json::Object& object, const char* key) {
    const auto value=object.getString(key); Check(bool(value),std::string("missing JSON string: ")+key); return value->str();
}
std::vector<llvm::json::Value> Events(const hs::GenerationLease& lease) {
    const auto context=lease.compiled_graph().module().As<api::CompiledModuleNode>()->profile_context_;
    Check(bool(context),"health evidence requires profiling"); context->Flush();
    std::ifstream input(std::filesystem::path(context->bundle_dir())/"events.jsonl");
    Check(input.good(),"missing profile bundle"); std::string line;
    std::vector<llvm::json::Value> result;
    while(std::getline(input,line)) {
        auto parsed=llvm::json::parse(line);
        if(!parsed) throw std::runtime_error(llvm::toString(parsed.takeError()));
        result.push_back(std::move(*parsed));
    }
    return result;
}
struct Proof { std::string id; int64_t duration; };
Proof CompletedRun(const hs::GenerationLease& lease, const std::string& stage="", const std::string& receipt="") {
    const auto events=Events(lease);
    std::string run, trace; int64_t duration=-1; size_t expected=lease.compiled_graph().plan().calls().size();
    for(const auto& value:events) {
        const auto* event=value.getAsObject(); Check(event,"event must be an object");
        if(String(*event,"event_type")!="runtime_session_run") continue;
        const auto* fields=event->getObject("fields"); Check(fields,"missing run fields");
        if(String(*fields,"generation")!=std::to_string(lease.generation())) continue;
        Check(String(*fields,"adaptive_contract")==std::to_string(hs::kAdaptiveHotSwapContractVersion) && String(*fields,"dispatch_key")==lease.dispatch_key().digest() &&
              String(*fields,"plan_abi")==lease.plan_abi().digest() && String(*fields,"plan_variant")==lease.selection_plan_key().digest() &&
              String(*fields,"validation_receipt")==lease.validation_receipt(),"run is not associated with its lease");
        Check(String(*event,"status")=="ok" && String(*event,"phase")=="complete","run is not completed successfully");
        if(!stage.empty()) Check(String(*fields,"stage")==stage && String(*fields,"export_receipt")==receipt,"run lost its model association");
        const auto* metrics=event->getObject("metrics"); Check(metrics,"missing run metrics");
        Check(metrics->getNumber("submit_count")==double(expected) && metrics->getNumber("kernel_count")==double(expected),"incomplete kernel submission evidence");
        const auto measured=event->getInteger("duration_ns"); Check(measured && *measured>=0,"invalid runtime duration");
        duration=*measured; run=String(*event,"run_id"); trace=String(*event,"trace_id");
    }
    Check(!run.empty(),"no completed run for this generation");
    std::set<std::string> completed, submitted;
    for(const auto& value:events) {
        const auto& event=*value.getAsObject();
        const auto type=String(event,"event_type");
        if(String(event,"run_id")!=run || (type!="kernel_exec" && type!="kernel_submit")) continue;
        const bool is_submit=type=="kernel_submit";
        Check(String(event,"status")=="ok" && String(event,"phase")== (is_submit?"submit":"complete"),"kernel event is incomplete");
        const auto* fields=event.getObject("fields"); Check(fields,"missing kernel fields");
        Check(String(*fields,"generation")==std::to_string(lease.generation()) && String(*fields,"plan_abi")==lease.plan_abi().digest() &&
              String(*fields,"adaptive_contract")==std::to_string(hs::kAdaptiveHotSwapContractVersion) && String(*fields,"dispatch_key")==lease.dispatch_key().digest() &&
              String(*fields,"plan_variant")==lease.selection_plan_key().digest() && String(*fields,"validation_receipt")==lease.validation_receipt(),"kernel association mismatch");
        if(!stage.empty()) Check(String(*fields,"stage")==stage && String(*fields,"export_receipt")==receipt,"kernel lost its model association");
        Check((is_submit?submitted:completed).insert(String(event,"kernel_symbol")).second,"duplicated kernel evidence");
    }
    Check(completed.size()==expected && submitted==completed,"missing kernel submission or completion evidence");
    return {trace+":"+run,duration};
}
bool SameStats(const ci::PrimitiveCacheStats& a,const ci::PrimitiveCacheStats& b) {
    return a.hits==b.hits && a.misses==b.misses && a.entries==b.entries && a.accounted_bytes==b.accounted_bytes &&
        a.evictions==b.evictions && a.in_flight==b.in_flight && a.merged_waiters==b.merged_waiters && a.failures==b.failures &&
        a.rejections==b.rejections && a.active_pins==b.active_pins;
}
// Fault injection at the existing issuance boundary: a receipt for a different
// candidate must never produce a discoverable/executable generation.
class WrongReceipt final : public hs::GenerationAuthority, private hs::CandidateValidationAuthority {
public:
    std::shared_ptr<const hs::GenerationLease> Issue(const hs::GenerationAuthorityRequest& request) override {
        const auto wrong=IssueReceipt("different-candidate-receipt");
        return MakeLease(1,{request.route,request.selection_plan,request.plan_abi,wrong,request.candidate,request.producer_reported_bytes});
    }
private:
    hs::ValidationReceipt Validate(const Request&,const api::CompiledGraph&) override { return IssueReceipt("unused"); }
};
// Explicit fixture policy consumes the real bundle, using LLVM's existing JSON
// parser. It is not a default policy or an automatic tuning decision.
class BundleHealth final : public hs::HealthAuthority {
public:
    int64_t threshold=std::numeric_limits<int64_t>::max();
    bool throw_evaluation=false, wrong_generation=false, empty_evidence=false;
    size_t verifications=0;
    hs::HealthDecision Evaluate(const hs::GenerationLease& lease) override {
        if(throw_evaluation) throw std::runtime_error("fixture health failure");
        const auto proof=CompletedRun(lease);
        return {lease.generation()+(wrong_generation?1:0),proof.duration>threshold?hs::HealthDisposition::kQuarantine:hs::HealthDisposition::kHealthy,
                empty_evidence?"":proof.id,proof.id};
    }
    bool VerifyAndConsume(const hs::HealthDecision& decision, const hs::GenerationLease& lease) noexcept override {
        ++verifications;
        try {
            const auto proof=CompletedRun(lease);
            return decision.generation==lease.generation() && decision.evidence_id==proof.id && decision.replay_token==proof.id && consumed.insert(proof.id).second;
        } catch(...) { return false; }
    }
private:
    std::set<std::string> consumed;
};
struct LaunchGate {
    std::mutex mutex; std::condition_variable wake;
    bool entered=false, released=false;
    std::atomic<bool> armed{true};
    void Release() { std::lock_guard<std::mutex> lock(mutex); released=true; wake.notify_all(); }
};
class GatedLLVM final : public codegen::KernelLauncher {
public:
    GatedLLVM(codegen::CompiledKernel real, std::shared_ptr<LaunchGate> gate):real_(std::move(real)),gate_(std::move(gate)) {}
    bool IsReady() const noexcept override { return real_.IsReady(); }
    AsyncOperation Launch(const Array<runtime::NDArray>& inputs,const DeviceStream& stream,const ObjectRef&) const override {
        auto result=real_.Launch(inputs,stream); result.Wait();
        if(gate_->armed.exchange(false)) {
            std::unique_lock<std::mutex> lock(gate_->mutex);
            gate_->entered=true; gate_->wake.notify_all();
            gate_->wake.wait(lock,[&] { return gate_->released; });
        }
        return result;
    }
private:
    codegen::CompiledKernel real_;
    std::shared_ptr<LaunchGate> gate_;
};
api::CompiledGraph WithLaunchGate(const api::CompiledGraph& graph,const std::shared_ptr<LaunchGate>& gate) {
    const auto* module=graph.module().As<api::CompiledModuleNode>();
    auto pins=graph.artifact_pins();
    const auto first=ci::ArtifactPinAccess::Unwrap(pins[0]);
    // Instrumentation changes the launcher, not the original unit's semantics.
    const auto key=ci::BuildPrimitiveArtifactKey(first.key().unit_semantic_key(),
        module->target_,"m6-test-launch-gate-v1","m6-test-launch-gate-v1","m6-test-launch-gate-v1");
    auto cache=ci::AcquirePrimitiveCache(key);
    Check(cache.access()==ci::PrimitiveCacheAccess::kOwner,"gate fixture key was reused");
    const auto& real=first.artifact();
    const auto controlled=ci::PublishPrimitiveCacheLease(cache,{real.signature,real.launch_metadata,
        codegen::CompiledKernel(real.signature,real.launch_metadata,std::make_shared<GatedLLVM>(real.kernel,gate)),
        real.accounted_bytes,"test-only gate around real LLVM","validated numeric delegate"});
    pins[0]=ci::ArtifactPinAccess::Wrap(controlled);
    std::vector<ci::CompiledModuleEntry> entries;
    for(size_t i=0;i<pins.size();++i) {
        const auto call=graph.plan().calls()[i];
        const auto& old=module->entries_.at(std::string(call->symbol));
        const auto artifact=ci::ArtifactPinAccess::Unwrap(pins[i]);
        entries.push_back({old.signature,old.launch_metadata,
            codegen::CompiledKernel(old.signature,old.launch_metadata,artifact.artifact().kernel->launcher),old.invocation_contract});
    }
    return ci::CompiledGraphAccess::Create(ci::BuildCompiledModule(module->target_,std::move(entries),
        ci::BorrowCompiledModuleConstants(graph.module())),graph.plan(),std::move(pins),graph.graph_semantic_key());
}
void TestRunningGenerationLifetime() {
    const auto function=Graph();
    const auto config=Config(1,"lifetime",false);
    const auto gate=std::make_shared<LaunchGate>();
    const auto graph=WithLaunchGate(api::Compiler::Compile(function,config),gate);
    hs::Options options; options.max_discoverable_generations=1;
    auto controller=std::make_unique<hs::AdaptiveHotSwapController>(options);
    auto first=controller->CompileAndPublish({Request(function,config,graph,{1})});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    std::weak_ptr<const hs::GenerationLease> weak=first;
    auto pending=std::async(std::launch::async,[&] {
        return controller->RunAsync(execution,{Tensor({1,2}),Tensor({3,4})},DeviceStream::Default(Device::CPU()));
    });
    // Release before the future/controller destruct even when an assertion fails.
    std::shared_ptr<void> release(nullptr,[&](void*) { gate->Release(); });
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        Check(gate->wake.wait_for(lock,std::chrono::seconds(5),[&] { return gate->entered; }),"old LLVM run did not reach launch gate");
    }
    const auto second=controller->CompileAndPublish({Request(function,Config(2,"lifetime-new",false),first->compiled_graph(),{1})});
    Check(controller->Acquire(execution)==second && second->generation()>first->generation(),"publication did not pass the old run");
    first.reset();
    Check(!weak.expired(),"eviction released an executing generation");
    const auto fresh=controller->RunAsync(execution,{Tensor({1,2}),Tensor({3,4})},DeviceStream::Default(Device::CPU()));
    Check(Read(fresh.outputs[0])==std::vector<float>({12,24}),"new generation failed while old run was paused");
    release.reset();
    auto old=pending.get();
    Check(old.lease->generation()<second->generation() && Read(old.outputs[0])==std::vector<float>({12,24}),"paused run switched generation");
    old.lease.reset(); old.outputs={};
    Check(!weak.expired(),"completion failed to retain the evicted generation");
    old.completion={};
    Check(weak.expired(),"old generation remained owned after its final completion handle");
    std::cout << "[PASS] real LLVM execution crosses publication with a controlled host gate; evicted lease lives until completion handle release\n";
}
Function StateGraph() {
    const Var past("past",TensorType({1,6,2},"float32")), token("token",TensorType({1,1,2},"float32"));
    const auto present=Call(relay::Op::Get("concatenate"),{past,token},relay::ConcatenateAttrs::Create(1));
    return Function({past,token},Tuple({Call(relay::Op::Get("nn_relu"),{present}),present}));
}
void TestStatefulReplacement() {
    const auto function=StateGraph();
    const auto config=Config(0,"state-seed");
    const auto graph=api::Compiler::Compile(function,config);
    const int64_t id=graph.plan().input_value_ids()[0];
    const auto plan=graph.plan().BindStateOutputs({{id,graph.plan().output_value_ids()[1],1,6,1}});
    const auto stateful=ci::CompiledGraphAccess::Create(graph.module(),plan,graph.artifact_pins(),graph.graph_semantic_key());
    const int64_t replaced=static_cast<int64_t>(plan.calls().size())-1;
    auto health=std::make_shared<BundleHealth>();
    hs::Options options; options.health_authority=health;
    hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({Request(function,Config(0,"state-first"),stateful,{replaced})});
    Check(!first->session(),"stateful generation allocated shared request state");
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto a=controller.CreateStatefulSession(execution), b=controller.CreateStatefulSession(execution);
    const auto alias=a;
    const auto seed=Tensor({1,2}).CreateView({1,1,2},{},0);
    a.InitializeState(id,seed,1);
    b.InitializeState(id,Tensor({-3,-4}).CreateView({1,1,2},{},0),1);
    const auto address=a.StateValue(id).storage().data();
    Check(address!=b.StateValue(id).storage().data(),"requests share KV storage");
    runtime::RuntimeSession reference(stateful.module(),plan);
    reference.InitializeState(id,seed,1);
    const auto stream=DeviceStream::Default(Device::CPU());
    const auto run=[&](const std::shared_ptr<const hs::GenerationLease>& expected, float value) {
        const auto input=Tensor({value,value+1}).CreateView({1,1,2},{},0);
        const auto stats=ci::GetPrimitiveCacheStats();
        const auto result=controller.RunAsync(a,{input},stream,{{"stage","decode"},{"export_receipt","cpp:stateful-concat-relu-v1"}});
        const auto truth=reference.Run({input});
        Check(result.completion.IsReady() && result.lease==expected && result.outputs.size()==truth.size(),"stateful run used the wrong generation or completion");
        for(size_t i=0;i<truth.size();++i) Check(Read(result.outputs[i])==Read(truth[i]),"stateful replacement changed outputs");
        Check(Read(a.StateValue(id))==Read(reference.StateValue(id)) && a.StateValue(id).storage().data()==address &&
              alias.StateExtent(id)==reference.StateExtent(id),"stateful replacement lost state, address or committed length");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"stateful execution compiled or accessed the cache");
        (void)CompletedRun(*result.lease,"decode","cpp:stateful-concat-relu-v1");
    };
    run(first,10);
    const auto before_publication=Read(a.StateValue(id));
    const auto second=controller.CompileAndPublish({Request(function,Config(1,"state-second"),first->compiled_graph(),{replaced})});
    Check(!second->session() && second->generation()>first->generation() && second->plan_abi()==first->plan_abi() &&
          second->compiled_graph().artifact_pins()[replaced].record().artifact_key!=first->compiled_graph().artifact_pins()[replaced].record().artifact_key &&
          Read(a.StateValue(id))==before_publication && a.StateExtent(id)==2,"publication changed state or failed to replace code");
    run(second,20);
    const auto a_before_b=Read(a.StateValue(id));
    const auto b_result=controller.RunAsync(b,{Tensor({-5,-6}).CreateView({1,1,2},{},0)},stream,
        {{"stage","decode"},{"export_receipt","cpp:stateful-concat-relu-v1"}});
    Check(b_result.lease==second && b.StateExtent(id)==2 && a.StateExtent(id)==3 && Read(a.StateValue(id))==a_before_b,
          "second request changed first request state");
    Check(Read(b.StateValue(id))[0]==-3 && Read(b.StateValue(id))[2]==-5,"second request lost its own prefix or append");
    health->threshold=0;
    Check(controller.EvaluateHealth(second) && controller.Acquire(execution)==first,"stateful health rollback failed");
    run(first,30);
    run(first,40);
    const auto state_before=Read(a.StateValue(id));
    const auto stats=ci::GetPrimitiveCacheStats();
    const auto event_count=Events(*first).size();
    const auto incompatible_plan=graph.plan().BindStateOutputs(
        {{id,graph.plan().output_value_ids()[1],1,6,1}},42.0);
    const auto incompatible=ci::CompiledGraphAccess::Create(graph.module(),incompatible_plan,
        graph.artifact_pins(),graph.graph_semantic_key());
    Rejects([&] {
        (void)hs::preparation::PrepareCandidate(Request(function,config,stateful,{replaced}),
            incompatible,"incompatible-state-fill");
    });
    Rejects([&] { (void)controller.RunAsync(execution,{seed},stream); });
    for(const char* key:{"adaptive_contract","generation","dispatch_key","plan_abi","plan_variant","validation_receipt"}) {
        Rejects([&] { (void)controller.RunAsync(a,{seed},stream,{{key,"forged"}}); });
    }
    Check(Events(*first).size()==event_count,"rejected state owner or metadata reached execution");
    Rejects([&] { (void)controller.RunAsync(a,{Tensor({1,2})},stream); });
    Rejects([&] { (void)controller.RunAsync(a,{seed},DeviceStream()); });
    Check(Read(a.StateValue(id))==state_before && a.StateExtent(id)==5 && SameStats(stats,ci::GetPrimitiveCacheStats()),
          "preflight changed state or compiled code");
    (void)controller.RunAsync(a,{seed},stream);
    const auto full=Read(a.StateValue(id));
    Rejects([&] { (void)controller.RunAsync(a,{seed},stream); });
    Check(a.StateExtent(id)==6 && Read(a.StateValue(id))==full,"capacity rejection changed state");
    const auto tombstone=controller.Submit({Request(function,Config(1,"state-tombstone"),first->compiled_graph(),{replaced})}).Wait();
    Check(!tombstone.ready() && controller.Acquire(execution)==first,"quarantined stateful selection was republished");
    std::cout << "[PASS] stateful LLVM generation 1 -> 2 -> rollback 1, stable KV/extent, isolated requests, no runtime compilation, preflight and capacity rejection\n";
}

void TestStatefulPublicationLifetime() {
    const auto function=StateGraph();
    const auto gate=std::make_shared<LaunchGate>();
    const auto graph=WithLaunchGate(api::Compiler::Compile(function,Config(0,"state-lifetime-seed",false)),gate);
    const auto id=graph.plan().input_value_ids()[0];
    const auto plan=graph.plan().BindStateOutputs({{id,graph.plan().output_value_ids()[1],1,6,1}});
    const auto stateful=ci::CompiledGraphAccess::Create(graph.module(),plan,graph.artifact_pins(),graph.graph_semantic_key());
    const int64_t replaced=static_cast<int64_t>(plan.calls().size())-1;
    hs::Options options; options.max_discoverable_generations=1;
    hs::AdaptiveHotSwapController controller(options);
    auto first=controller.CompileAndPublish({Request(function,Config(0,"state-lifetime-first",false),stateful,{replaced})});
    std::weak_ptr<const hs::GenerationLease> weak=first;
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto session=controller.CreateStatefulSession(execution);
    session.InitializeState(id,Tensor({1,2}).CreateView({1,1,2},{},0),1);
    const auto address=session.StateValue(id).storage().data();
    const auto stream=DeviceStream::Default(Device::CPU());
    auto pending=std::async(std::launch::async,[&] {
        return controller.RunAsync(session,{Tensor({3,4}).CreateView({1,1,2},{},0)},stream);
    });
    std::shared_ptr<void> release(nullptr,[&](void*) { gate->Release(); });
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        Check(gate->wake.wait_for(lock,std::chrono::seconds(5),[&] { return gate->entered; }),"stateful LLVM launch did not reach gate");
    }
    Check(session.StateExtent(id)==1,"unfinished run committed its extent");
    const auto second=controller.CompileAndPublish({Request(function,Config(1,"state-lifetime-second",false),first->compiled_graph(),{replaced})});
    first.reset();
    Check(!weak.expired() && controller.Acquire(execution)==second,"publication lost the executing stateful lease");
    const auto other=controller.CreateStatefulSession(execution);
    const auto fresh=controller.RunAsync(other,{Tensor({9,10}).CreateView({1,1,2},{},0)},stream);
    Check(fresh.lease==second && other.StateExtent(id)==1 && session.StateExtent(id)==1,
          "independent request did not run while old generation was paused");
    release.reset();
    auto old=pending.get();
    Check(old.lease->generation()<second->generation() && session.StateExtent(id)==2 && session.StateValue(id).storage().data()==address,
          "paused stateful run changed generation or state owner");
    old.lease.reset(); old.outputs={};
    Check(!weak.expired(),"stateful completion failed to retain evicted lease");
    old.completion={};
    // RuntimeSession also holds its last completion until the next state run.
    const auto next=controller.RunAsync(session,{Tensor({5,6}).CreateView({1,1,2},{},0)},stream);
    Check(weak.expired(),"next stateful run failed to release the prior completion lease");
    const auto state=Read(session.StateValue(id));
    Check(next.lease==second && session.StateExtent(id)==3 && session.StateValue(id).storage().data()==address &&
          std::vector<float>(state.begin(),state.begin()+6)==std::vector<float>({1,2,3,4,5,6}),"next generation lost the previous run's committed prefix");
    std::cout << "[PASS] stateful run spans publication, independent request progresses, completion retains evicted lease, next step keeps the same KV\n";
}
class FaultLLVM final : public codegen::KernelLauncher {
public:
    FaultLLVM(codegen::CompiledKernel real, std::shared_ptr<size_t> submits, bool fail)
        : real_(std::move(real)),submits_(std::move(submits)),fail_(fail) {}
    bool IsReady() const noexcept override { return true; }
    AsyncOperation Launch(const Array<runtime::NDArray>& inputs,const DeviceStream& stream,const ObjectRef&) const override {
        ++*submits_;
        if(fail_) throw std::runtime_error("injected second kernel failure");
        return real_.Launch(inputs,stream);
    }
private:
    codegen::CompiledKernel real_;
    std::shared_ptr<size_t> submits_;
    bool fail_;
};
void TestStatefulReplacementFailure() {
    const auto graph=api::Compiler::Compile(StateGraph(),Config(0,"state-failure",false));
    const auto id=graph.plan().input_value_ids()[0];
    const auto plan=graph.plan().BindStateOutputs({{id,graph.plan().output_value_ids()[1],1,6,1}});
    runtime::RuntimeSession session(graph.module(),plan);
    const auto* original=graph.module().As<api::CompiledModuleNode>();
    const auto submits=std::make_shared<size_t>(0);
    std::vector<ci::CompiledModuleEntry> entries;
    for(size_t i=0;i<plan.calls().size();++i) {
        const auto& entry=original->entries_.at(std::string(plan.calls()[i]->symbol));
        entries.push_back({entry.signature,entry.launch_metadata,
            codegen::CompiledKernel(entry.signature,entry.launch_metadata,
                std::make_shared<FaultLLVM>(entry.executable,submits,i==1)),entry.invocation_contract});
    }
    const auto faulty=ci::BuildCompiledModule(original->target_,std::move(entries),ci::BorrowCompiledModuleConstants(graph.module()));
    const auto input=Tensor({1,2}).CreateView({1,1,2},{},0);
    const auto before=Read(session.StateValue(id));
    const auto stream=DeviceStream::Default(Device::CPU());
    Rejects([&] { (void)session.RunAsyncWithModule(faulty,{input},stream); });
    Check(*submits==2 && session.StateExtent(id)==0 && Read(session.StateValue(id))==before,
          "failed replacement committed state or did not execute its first real kernel");
    bool poisoned=false;
    try { (void)session.RunAsyncWithModule(graph.module(),{input},stream); }
    catch(const std::exception& error) { poisoned=std::string(error.what()).find("failed previously")!=std::string::npos; }
    Check(poisoned && *submits==2,"selecting the original module silently healed a poisoned session");
    Rejects([&] { session.InitializeState(id,input,1); });
    std::cout << "[PASS] replacement failure after real submission poisons the session; code rollback does not reset failed state\n";
}

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
void TestBoundedReplacement() {
    namespace rs = api::experimental::restricted_symbolic_shape::v1;
    using Adapter = rs::RestrictedSymbolicShapeAdapter;
    const Var past("past",TensorType({1,2,2},"float32")), token("token",TensorType({1,1,2},"float32"));
    const auto present=Call(relay::Op::Get("concatenate"),{past,token},relay::ConcatenateAttrs::Create(1));
    const Function function({past,token},Tuple({Call(relay::Op::Get("nn_relu"),{present}),present}));
    const auto authority=[&](int level,const std::string& name,int64_t upper=5) {
        return Adapter::MintBoundedCompileRequest(Adapter::Prepare(function,Config(level,name),
            {{0,0,"B",1,3,1},{0,1,"P",0,upper,1},{1,0,"B",1,3,1}}));
    };
    const auto initial=authority(0,"bounded-seed");
    const auto graph=api::Compiler::CompileBounded(initial);
    const int64_t id=graph.plan().input_value_ids()[0];
    const std::vector<runtime::StateOutputBinding> bindings{{id,graph.plan().output_value_ids()[1],1,-1,1}};
    const auto plan=graph.plan().BindBoundedStateOutputs(bindings,{{2,6,2}},-999);
    const auto stateful=ci::CompiledGraphAccess::Create(graph.module(),plan,graph.artifact_pins(),graph.graph_semantic_key());
    auto health=std::make_shared<BundleHealth>(); hs::Options options; options.health_authority=health;
    hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({Request(authority(0,"bounded-first"),stateful,{1})});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    Check(!first->session(),"bounded candidate allocated shared state");
    auto session=controller.CreateStatefulSession(execution), other=controller.CreateStatefulSession(execution);
    const auto seed=Tensor({-1,2,3,-4}).CreateView({2,1,2},{},0);
    session.InitializeState(id,seed,1); other.InitializeState(id,seed,1);
    runtime::RuntimeSession reference(graph.module(),plan); reference.InitializeState(id,seed,1);
    const auto address=session.StateValue(id).storage().data();
    const auto stream=DeviceStream::Default(Device::CPU());
    const auto run=[&](const std::shared_ptr<const hs::GenerationLease>& lease,float n,bool check_evidence=true) {
        const auto input=Tensor({n,-n,n+1,-n-1}).CreateView({2,1,2},{},0);
        const auto stats=ci::GetPrimitiveCacheStats();
        const auto actual=controller.RunAsync(session,{input},stream,
            {{"stage","bounded_decode"},{"export_receipt","cpp:bounded-concat-relu-v1"}});
        const auto expected=reference.Run({input});
        Check(actual.lease==lease && actual.completion.IsReady() && actual.outputs.size()==expected.size(),"bounded run lost its lease or completion");
        for(size_t i=0;i<expected.size();++i) Check(Read(actual.outputs[i])==Read(expected[i]),"bounded replacement changed output strides or data");
        Check(Read(session.StateValue(id))==Read(reference.StateValue(id)) &&
            session.StateExtent(id)==reference.StateExtent(id) && session.StateValue(id).storage().data()==address,
            "bounded replacement lost capacity storage, prefix, sentinel or committed extent");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"bounded Run compiled or accessed cache");
        if(check_evidence) (void)CompletedRun(*lease,"bounded_decode","cpp:bounded-concat-relu-v1");
    };
    run(first,10);
    const auto second=controller.CompileAndPublish({Request(authority(1,"bounded-second"),first->compiled_graph(),{1})});
    Check(second->plan_abi()==first->plan_abi() && second->dispatch_key()==first->dispatch_key() &&
        second->compiled_graph().artifact_pins()[0].record().artifact_key==first->compiled_graph().artifact_pins()[0].record().artifact_key &&
        second->compiled_graph().artifact_pins()[1].record().artifact_key!=first->compiled_graph().artifact_pins()[1].record().artifact_key,
        "bounded replacement changed an unselected artifact or failed to change selected code");
    run(second,20);
    const auto before_other=Read(session.StateValue(id));
    const auto isolated=controller.RunAsync(other,{seed},stream);
    Check(isolated.lease==second && other.StateExtent(id)==2 && session.StateExtent(id)==3 &&
          Read(session.StateValue(id))==before_other && other.StateValue(id).storage().data()!=address,
          "bounded requests share state");
    health->threshold=0;
    Check(controller.EvaluateHealth(second) && controller.Acquire(execution)==first,"bounded health rollback failed");
    run(first,30); run(first,40);
    const auto stats=ci::GetPrimitiveCacheStats();
    Rejects([&] { (void)Request(function,Config(0,"bounded-bare-reject"),stateful,{1}); });
    Rejects([&] { (void)Request(authority(0,"bounded-upper-reject",4),stateful,{1}); });
    const auto mismatch=ci::CompiledGraphAccess::Create(graph.module(),
        graph.plan().BindBoundedStateOutputs(bindings,{{2,5,2}},-999),graph.artifact_pins(),graph.graph_semantic_key());
    Rejects([&] { (void)hs::preparation::PrepareCandidate(Request(initial,stateful,{1}),mismatch,"wrong-capacity"); });
    Check(api::BuildBoundedShapeProfileKey(graph.graph_semantic_key(),graph.plan())==
        api::BuildBoundedShapeProfileKey(stateful.graph_semantic_key(),stateful.plan()),"KV ownership changed the logical input profile");
    for(int change=0;change<4;++change) {
        auto guards=graph.plan().graph_input_guards();
        if(change==0) guards[1].lower=1;
        if(change==1) guards[1].upper=4;
        if(change==2) guards[0].divisible_by=2;
        if(change==3) guards[2].equal_to.reset();
        const auto altered=runtime::ExecutablePlan(graph.plan().values(),graph.plan().calls(),
            graph.plan().input_value_ids(),graph.plan().constant_value_ids(),graph.plan().output_value_ids(),
            {},runtime::ExecutablePlanMode::kDynamicFreshOutputV1,guards);
        const auto profile=api::BuildBoundedShapeProfileKey(graph.graph_semantic_key(),altered);
        const auto dispatch=api::BuildBoundedDispatchKey(graph.graph_semantic_key(),profile);
        Check(dispatch!=first->dispatch_key(),"bounded route omitted a bound, divisibility or equality constraint");
        Rejects([&] { (void)controller.Acquire(Execution(dispatch,first->plan_abi())); });
    }
    Rejects([&] { (void)api::BuildStaticExactShapeProfileKey(graph.graph_semantic_key(),graph.plan()); });
    Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"invalid bounded preparation touched the primitive cache");
    const auto execution_events=[&] {
        size_t count=0;
        for(const auto& value:Events(*first)) {
            const auto type=String(*value.getAsObject(),"event_type");
            count+=type=="alloc" || type=="copy" || type=="kernel_submit" || type=="kernel_exec";
        }
        return count;
    };
    const auto checkpoint=execution_events();
    const auto state_before=Read(session.StateValue(id));
    for(const auto& bad : std::vector<runtime::NDArray>{Tensor({1,2}).CreateView({1,1,2},{},0),Tensor({1,2,3,4}).CreateView({2,2,1},{},0)})
        Rejects([&] { (void)controller.RunAsync(session,{bad},stream); });
    Check(execution_events()==checkpoint && session.StateExtent(id)==5 && Read(session.StateValue(id))==state_before &&
          SameStats(stats,ci::GetPrimitiveCacheStats()),
          "invalid bounded input launched or changed state");
    run(first,50,false);
    const auto full=execution_events();
    Rejects([&] { (void)controller.RunAsync(session,{seed},stream); });
    Check(execution_events()==full && session.StateExtent(id)==6,"full bounded state failed before-launch capacity rejection");

    const auto fresh=controller.CompileAndPublish({Request(authority(0,"bounded-fresh"),graph,{1})});
    const Execution fresh_execution(fresh->dispatch_key(),fresh->plan_abi());
    for(const auto& [batch,length] : std::vector<std::pair<int64_t,int64_t>>{{1,0},{2,2},{3,5}}) {
        std::vector<float> pv(batch*length*2,-3),tv(batch*2,7);
        const auto a=Tensor(pv).CreateView({batch,length,2},{},0), b=Tensor(tv).CreateView({batch,1,2},{},0);
        const auto before=ci::GetPrimitiveCacheStats();
        const auto actual=controller.RunAsync(fresh_execution,{a,b},stream);
        const auto expected=runtime::RuntimeSession(graph.module(),graph.plan()).Run({a,b});
        Check(Read(actual.outputs[0])==Read(expected[0]) && Read(actual.outputs[1])==Read(expected[1]) &&
            SameStats(before,ci::GetPrimitiveCacheStats()),"bounded stateless routing changed values or compiled at Run");
    }
    std::cout << "[PASS] bounded hot swap: selected unit only, B=2 stable KV, P=1->6, generation rollback, independent requests; "
                 "fresh B/P=1/0,2/2,3/5; authority/capacity/input rejection, no runtime compilation\n";
}

void TestBoundedWiringRejection() {
    namespace rs = api::experimental::restricted_symbolic_shape::v1;
    using Adapter = rs::RestrictedSymbolicShapeAdapter;
    const auto authority=Adapter::MintBoundedCompileRequest(Adapter::Prepare(Graph(),Config(0,"bounded-wiring"),
        {{0,0,"N",1,4,1},{1,0,"N",1,4,1}}));
    const auto graph=api::Compiler::CompileBounded(authority);
    const auto base=graph.plan(); auto calls=base.calls();
    Check(calls.size()==2,"wiring fixture requires add then multiply");
    auto inputs=calls[1].input_value_ids(); inputs[1]=base.input_value_ids()[0];
    calls[1]=runtime::KernelCall(calls[1]->symbol,inputs,calls[1].output_value_ids());
    const auto changed=runtime::ExecutablePlan(base.values(),calls,base.input_value_ids(),
        base.constant_value_ids(),base.output_value_ids(),{},base.mode(),base.graph_input_guards());
    // Physical signatures still match, but (a+b)*a is not (a+b)*b.
    const auto wrong=ci::CompiledGraphAccess::Create(graph.module(),changed,graph.artifact_pins(),graph.graph_semantic_key());
    const auto a=Tensor({1,2}), b=Tensor({3,4});
    Check(Read(runtime::RuntimeSession(graph.module(),base).Run({a,b})[0])!=
        Read(runtime::RuntimeSession(wrong.module(),wrong.plan()).Run({a,b})[0]),"wiring negative is not semantically different");
    const auto stats=ci::GetPrimitiveCacheStats();
    bool rejected=false;
    try { (void)Request(authority,wrong,{1}); }
    catch(const std::invalid_argument& e) { rejected=std::string(e.what()).find("call wiring")!=std::string::npos; }
    Check(rejected && SameStats(stats,ci::GetPrimitiveCacheStats()),"rewired bounded baseline was admitted or touched cache");
    std::cout << "[PASS] bounded authority rejects physically compatible but semantically different call wiring before compilation\n";
}

runtime::NDArray BatchTensor(const Array<int64_t>& shape, float first=0) {
    auto value=runtime::NDArray::Empty(shape,runtime::DataTypeFromString("float32"),Device::CPU());
    std::vector<float> data(value.NBytes()/sizeof(float));
    for(size_t i=0;i<data.size();++i) data[i]=first+static_cast<float>(i);
    if(!data.empty()) value.CopyFromBytes(data.data(),value.NBytes());
    return value;
}
runtime::ExecutablePlan BatchPlan(const api::CompiledGraph& graph, int64_t axis=1, int64_t append=1) {
    const Array<int64_t> shape=axis==1?Array<int64_t>{3,6,4}:Array<int64_t>{3,2,7,4};
    return graph.plan().BindBoundedStateOutputs(
        {{graph.plan().input_value_ids()[0],graph.plan().output_value_ids()[1],axis,-1,append}},
        {shape},-999).BindRequestBatching(2);
}
void TestRequestBatchReplacement() {
    const auto mint=[](int level,const char* name) { return test_support::RequestBatchingGraph(Config(level,name),2,2); };
    const auto baseline=api::Compiler::CompileBounded(mint(0,"batch-seed"));
    const auto plan=BatchPlan(baseline,2,2);
    const auto graph=ci::CompiledGraphAccess::Create(baseline.module(),plan,baseline.artifact_pins(),baseline.graph_semantic_key());
    auto health=std::make_shared<BundleHealth>(); hs::Options options; options.health_authority=health;
    hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({Request(mint(0,"batch-first"),graph,{1})});
    Check(first->selection_plan_key().canonical_bytes().find("bounded-request-slots-gather-prefix-scatter-append-v1")!=std::string::npos,
          "batch selection omitted the existing memory contract");
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto session=controller.CreateStatefulSession(execution), isolated=controller.CreateStatefulSession(execution);
    runtime::RuntimeSession reference(baseline.module(),plan);
    const auto state_id=plan.state_value_ids()[0];
    const auto admit=[&](int extent,float value) {
        Array<runtime::NDArray> seed;
        if(extent) seed.push_back(BatchTensor({1,2,extent,4},value));
        const auto id=session.AdmitRequest(seed,extent);
        Check(reference.AdmitRequest(seed,extent)==id,"reference request order differs"); return id;
    };
    const auto a=admit(1,-2), b=admit(0,0), c=admit(1,20);
    const auto independent=isolated.AdmitRequest();
    const auto enqueue=[&](uint64_t id,float value) {
        const Array<runtime::NDArray> inputs{BatchTensor({1,2,2,4},value),BatchTensor({1,2},value)};
        session.EnqueueRequest(id,inputs); reference.EnqueueRequest(id,inputs);
    };
    const auto stream=DeviceStream::Default(Device::CPU());
    const auto run=[&](const std::shared_ptr<const hs::GenerationLease>& lease,const std::vector<uint64_t>& ids) {
        const auto before=ci::GetPrimitiveCacheStats();
        const auto actual=controller.RunNextBatch(session,stream,{{"stage","request_decode"},{"export_receipt","cpp:request-axis2-v1"}});
        const auto expected=reference.RunNextBatch();
        Check(actual.lease==lease && actual.results.size()==ids.size() && expected.size()==ids.size(),"batch selected wrong generation or requests");
        for(size_t row=0;row<ids.size();++row) {
            Check(actual.results[row].request_id==ids[row] && expected[row].request_id==ids[row],"batch changed FIFO/equal-extent grouping");
            for(size_t out=0;out<expected[row].outputs.size();++out)
                Check(Read(actual.results[row].outputs[out])==Read(expected[row].outputs[out]),"replacement batch output differs");
            Check(session.RequestExtent(ids[row])==reference.RequestExtent(ids[row]) &&
                Read(session.CopyRequestState(ids[row],state_id))==Read(reference.CopyRequestState(ids[row],state_id)),"replacement lost request KV");
        }
        Check(SameStats(before,ci::GetPrimitiveCacheStats()),"request execution touched compiler cache");
        (void)CompletedRun(*lease,"request_decode","cpp:request-axis2-v1");
        return actual;
    };
    enqueue(a,-4); enqueue(b,10); enqueue(c,30);
    const auto retained=run(first,{a,c}); const auto retained_values=Read(retained.results[0].outputs[0]);
    Check(session.RequestExtent(b)==0 && isolated.RequestExtent(independent)==0,"incompatible or isolated request changed");
    const auto second=controller.CompileAndPublish({Request(mint(1,"batch-second"),first->compiled_graph(),{1})});
    Check(first->plan_abi()==second->plan_abi() && first->dispatch_key()==second->dispatch_key(),"request replacement changed route/ABI");
    for(size_t i=0;i<3;++i) Check((first->compiled_graph().artifact_pins()[i].record().artifact_key==
        second->compiled_graph().artifact_pins()[i].record().artifact_key)==(i!=1),"replacement did not preserve unselected request kernels");
    (void)run(second,{b});
    enqueue(a,40); session.ReleaseRequest(a); reference.ReleaseRequest(a);
    const auto d=admit(0,0); Check(d!=a,"replacement reused a departed request id");
    Rejects([&] { session.EnqueueRequest(a,{}); });
    enqueue(d,50); (void)run(second,{d});
    health->threshold=0;
    Check(controller.EvaluateHealth(second) && !controller.EvaluateHealth(second) && controller.Acquire(execution)==first,
          "request evidence did not quarantine and roll back exactly once");
    enqueue(b,60); enqueue(c,70); enqueue(d,80);
    (void)run(first,{b,d}); (void)run(first,{c});
    Check(controller.RunNextBatch(session,stream).results.empty() && Read(retained.results[0].outputs[0])==retained_values,
          "departure/reuse left work or invalidated old outputs");
    enqueue(c,90); (void)run(first,{c});
    const auto checkpoint=Events(*first).size(); const auto stats=ci::GetPrimitiveCacheStats();
    Rejects([&] { session.EnqueueRequest(c,{BatchTensor({1,2,2,4}),BatchTensor({1,2})}); });
    Rejects([&] { (void)controller.RunNextBatch(session,DeviceStream{}); });
    Rejects([&] { (void)controller.RunNextBatch(session,stream,{{"generation","forged"}}); });
    Rejects([&] { (void)controller.RunAsync(session,{},stream); });
    Rejects([&] { (void)session.StateValue(state_id); });
    const auto different=graph.BindRequestBatching(1);
    Rejects([&] { (void)hs::preparation::PrepareCandidate(Request(mint(0,"batch-max-reject"),graph,{1}),different,"wrong-max"); });
    Check(Events(*first).size()==checkpoint && SameStats(stats,ci::GetPrimitiveCacheStats()),"request preflight rejection performed work");

    runtime::RuntimeSession raw(first->compiled_graph().module(),plan);
    const auto id=raw.AdmitRequest(); raw.EnqueueRequest(id,{BatchTensor({1,2,2,4}),BatchTensor({1,2})});
    const auto wrong=api::Compiler::CompileBounded(test_support::RequestBatchingGraph(Config(0,"batch-wrong-module"),2,1));
    const auto before=Events(*first).size(); const auto cache=ci::GetPrimitiveCacheStats();
    Rejects([&] { (void)raw.RunNextBatchWithModule(wrong.module(),stream); });
    Check(raw.RequestExtent(id)==0 && Events(*first).size()==before && SameStats(cache,ci::GetPrimitiveCacheStats()),
          "incompatible replacement packed inputs, launched or changed queue state");
    Check(raw.RunNextBatchWithModule(baseline.module(),stream).size()==1 && raw.RequestExtent(id)==2,
          "replacement preflight discarded the queued request");
    const auto* module=baseline.module().As<api::CompiledModuleNode>();
    const auto submits=std::make_shared<size_t>(0);
    std::vector<ci::CompiledModuleEntry> entries;
    for(size_t i=0;i<plan.calls().size();++i) {
        const auto& entry=module->entries_.at(std::string(plan.calls()[i]->symbol));
        entries.push_back({entry.signature,entry.launch_metadata,
            codegen::CompiledKernel(entry.signature,entry.launch_metadata,
                std::make_shared<FaultLLVM>(entry.executable,submits,i==1)),entry.invocation_contract});
    }
    const auto faulty=ci::BuildCompiledModule(module->target_,std::move(entries),ci::BorrowCompiledModuleConstants(baseline.module()));
    raw.EnqueueRequest(id,{BatchTensor({1,2,2,4}),BatchTensor({1,2})});
    Rejects([&] { (void)raw.RunNextBatchWithModule(faulty,stream); });
    Check(*submits==2,"faulting replacement did not submit the first real request kernel");
    Rejects([&] { (void)raw.RunNextBatchWithModule(baseline.module(),stream); });
    Rejects([&] { (void)raw.RequestExtent(id); });
    Check(*submits==2,"code rollback healed a poisoned request session");
    std::cout << "[PASS] request hot swap: axis-2 two-token append; queued FIFO/equal-P batches survive 1->2->rollback 1; "
                 "selected kernel only, departure/reuse, isolated slots, retained outputs, capacity/ABI/module rejection, "
                 "post-submit replacement failure stays poisoned, no runtime compilation\n";
}

void TestRequestBatchPublicationLifetime() {
    const auto gate=std::make_shared<LaunchGate>();
    const auto mint=[](int level) { return test_support::RequestBatchingGraph(Config(level,"batch-lifetime",false)); };
    const auto baseline=WithLaunchGate(api::Compiler::CompileBounded(mint(0)),gate);
    const auto plan=BatchPlan(baseline);
    const auto graph=ci::CompiledGraphAccess::Create(baseline.module(),plan,baseline.artifact_pins(),baseline.graph_semantic_key());
    hs::Options options; options.max_discoverable_generations=1;
    hs::AdaptiveHotSwapController controller(options);
    auto first=controller.CompileAndPublish({Request(mint(0),graph,{2})});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto session=controller.CreateStatefulSession(execution), other=controller.CreateStatefulSession(execution);
    const auto a=session.AdmitRequest(), b=session.AdmitRequest(), c=other.AdmitRequest();
    const Array<runtime::NDArray> inputs{BatchTensor({1,1,4},-2),BatchTensor({1,2},-1)};
    session.EnqueueRequest(a,inputs); session.EnqueueRequest(b,inputs); other.EnqueueRequest(c,inputs);
    const auto stream=DeviceStream::Default(Device::CPU());
    std::weak_ptr<const hs::GenerationLease> weak=first;
    auto pending=std::async(std::launch::async,[&] { return controller.RunNextBatch(session,stream); });
    std::shared_ptr<void> release(nullptr,[&](void*) { gate->Release(); });
    {
        std::unique_lock<std::mutex> lock(gate->mutex);
        Check(gate->wake.wait_for(lock,std::chrono::seconds(5),[&] { return gate->entered; }),"old request batch did not enter the real kernel gate");
    }
    const auto second=controller.CompileAndPublish({Request(mint(1),first->compiled_graph(),{0})});
    first.reset(); Check(!weak.expired(),"executing batch lost its evicted generation");
    Rejects([&] { session.ReleaseRequest(a); });
    Rejects([&] { (void)session.AdmitRequest(); });
    Rejects([&] { (void)controller.RunNextBatch(session,stream); });
    const auto fresh=controller.RunNextBatch(other,stream);
    Check(fresh.lease==second && fresh.results.size()==1 && other.RequestExtent(c)==1,"independent new-generation batch could not progress");
    release.reset(); auto old=pending.get();
    Check(old.lease->generation()<second->generation() && old.results.size()==2 &&
        Read(old.results[0].outputs[0])==Read(fresh.results[0].outputs[0]) &&
        session.RequestExtent(a)==1 && session.RequestExtent(b)==1,"in-flight batch switched code or lost request state");
    old.lease.reset(); Check(weak.expired(),"completed result retained an unreferenced generation");
    session.EnqueueRequest(a,inputs);
    Check(controller.RunNextBatch(session,stream).lease==second && session.RequestExtent(a)==2,"next batch did not reuse the same state with new code");
    std::cout << "[PASS] real LLVM request batch spans publication; concurrent same-session calls reject, independent session progresses, "
                 "evicted lease survives execution, next batch keeps queued KV\n";
}
#endif

void TestObserverIsolation() {
    const auto function=Graph();
    const auto config=Config(1,"observer",false);
    const auto graph=api::Compiler::Compile(function,config);
    const Request request(function,config,graph,{0,1});
    const Execution execution(request.dispatch_key(),request.plan_abi());
    std::mutex mutex; std::condition_variable wake;
    bool entered=false, release=false, reentry_rejected=false;
    hs::AdaptiveHotSwapController* controller_ptr=nullptr;
    hs::Options options;
    options.observer=[&](const hs::Event& event) {
        if(event.kind!=hs::EventKind::kPublished) return;
        bool rejected=false;
        try { (void)controller_ptr->Acquire(execution); } catch(const std::logic_error&) { rejected=true; }
        std::unique_lock<std::mutex> lock(mutex);
        reentry_rejected=rejected; entered=true; wake.notify_all();
        wake.wait(lock,[&] { return release; });
    };
    hs::AdaptiveHotSwapController controller(options); controller_ptr=&controller;
    std::shared_ptr<void> unblock(nullptr,[&](void*) {
        std::lock_guard<std::mutex> lock(mutex); release=true; wake.notify_all();
    });
    const auto ticket=controller.Submit({request});
    {
        std::unique_lock<std::mutex> lock(mutex);
        Check(wake.wait_for(lock,std::chrono::seconds(5),[&] { return entered; }),"observer did not start");
    }
    bool acquired=false;
    try { acquired=bool(controller.Acquire(execution)); } catch(const std::logic_error&) {}
    unblock.reset();
    Check(ticket.Wait().ready() && acquired && reentry_rejected,
          "observer blocked another thread or allowed recursive entry");
    std::cout << "[PASS] observer callback rejects recursion without rejecting concurrent Acquire\n";
}
std::vector<char> File(const std::filesystem::path& path) {
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    Check(stream.good(),"missing fixture: "+path.string());
    const auto size=stream.tellg(); Check(size>=0,"invalid fixture size"); stream.seekg(0);
    std::vector<char> bytes(static_cast<size_t>(size));
    Check(bool(stream.read(bytes.data(),size)),"cannot read fixture: "+path.string()); return bytes;
}
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
void TestMiniMindRequestBatching(const std::filesystem::path& root) {
    namespace rs=api::experimental::restricted_symbolic_shape::v1;
    using Adapter=rs::RestrictedSymbolicShapeAdapter;
    const auto source=frontend::LoadONNXShapeSource((root/"decode.json").string(),(root/"decode.params").string());
    Check(source.input_names.size()==17 && source.declared_output_types.size()==17,"request model ports changed");
    std::ifstream receipt_file(root/"export_receipt.txt"); std::string receipt;
    Check(bool(std::getline(receipt_file,receipt)) && receipt=="158ea389d1a332ee51df9901459c387034fd6b1eed4436fcbcd4740f9fba5fb9",
          "request test requires the locked eight-layer decode fixture");
    std::vector<rs::InputAxisSymbol> axes{{0,0,"B",1,3,1}};
    for(size_t i=1;i<17;++i) { axes.push_back({i,0,"B",1,3,1}); axes.push_back({i,1,"P",0,8,1}); }
    const auto mint=[&](int level,const char* name) {
        return Adapter::MintBoundedCompileRequest(Adapter::Prepare(source.function,Config(level,name),axes,source.declared_output_types));
    };
    std::cout << "[INFO] request model: compile baseline" << std::endl;
    const auto baseline=api::Compiler::CompileBounded(mint(0,"batch-model-reference"));
    Check(baseline.plan().calls().size()==774,"request model lost an ordinary kernel");
    std::vector<runtime::StateOutputBinding> bindings; std::vector<Array<int64_t>> shapes;
    for(size_t i=1;i<17;++i) {
        bindings.push_back({baseline.plan().input_value_ids()[i],baseline.plan().output_value_ids()[i],1,-1,1});
        shapes.push_back({3,9,4,96});
    }
    const auto graph=baseline.BindBoundedStateOutputs(bindings,shapes,1e20f).BindRequestBatching(2);
    const auto plan=graph.plan();
    Request initial(mint(0,"batch-model-first"),graph,{773});
    const uint64_t route_bytes=initial.dispatch_key().canonical_bytes().size()+initial.plan_abi().canonical_bytes().size();
    Check(route_bytes<2ULL*1024*1024*1024,"request model exceeds explicit route budget");
    auto health=std::make_shared<BundleHealth>(); hs::Options options; options.health_authority=health;
    options.max_routes=1; options.max_route_metadata_bytes=route_bytes+64;
    options.max_quarantine_tombstone_bytes=2ULL*1024*1024*1024;
    hs::AdaptiveHotSwapController controller(options);
    std::cout << "[INFO] request model: publish first generation" << std::endl;
    const auto first=controller.CompileAndPublish({std::move(initial)});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto session=controller.CreateStatefulSession(execution);
    const runtime::RuntimeSession reference(baseline.module(),baseline.plan());
    struct ModelRequest final {
        uint64_t id;
        Array<runtime::NDArray> past;
        runtime::NDArray token;
        int64_t extent;
        size_t fixture_case, fixture_row;
    };
    const auto admit=[&](size_t case_index,size_t row) {
        Check((case_index==0 && row==0) || (case_index==2 && row<2),"invalid request seed case");
        const int64_t batch=case_index==0?1:2, past=case_index==0?0:4;
        const auto tokens=File(root/("input_"+std::to_string(case_index)+".bin"));
        Check(tokens.size()==size_t(batch)*sizeof(int64_t),"request token fixture size changed");
        auto token=runtime::NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
        token.CopyFromBytes(tokens.data()+row*sizeof(int64_t),sizeof(int64_t));
        Array<runtime::NDArray> states;
        for(size_t layer=0;layer<16;++layer) {
            auto state=runtime::NDArray::Empty({1,past,4,96},runtime::DataTypeFromString("float32"),Device::CPU());
            const auto bytes=File(root/("past_"+std::to_string(case_index)+"_"+std::to_string(layer)+".bin"));
            Check(bytes.size()==size_t(batch)*state.NBytes(),"request state fixture size changed");
            if(state.NBytes()) state.CopyFromBytes(bytes.data()+row*state.NBytes(),state.NBytes());
            states.push_back(state);
        }
        return ModelRequest{session.AdmitRequest(states,past),states,token,past,case_index,row};
    };
    auto a=admit(2,0), b=admit(0,0), c=admit(2,1);
    size_t batches=0, request_steps=0, onnx_values=0, independent_values=0;
    double worst=0; std::vector<hs::Generation> generations;
    const auto stream=DeviceStream::Default(Device::CPU());
    const auto execute=[&](const std::vector<ModelRequest*>& requests,
                           const std::shared_ptr<const hs::GenerationLease>& lease,
                           bool initial_fixture) {
        const auto stats=ci::GetPrimitiveCacheStats();
        std::vector<Array<runtime::NDArray>> independent;
        for(const auto* request:requests) {
            Array<runtime::NDArray> inputs{request->token};
            for(const auto& state:request->past) inputs.push_back(state);
            independent.push_back(reference.Run(inputs,{{"stage","request_reference"},{"model","minimind"},
                {"export_receipt",receipt},{"request_id",std::to_string(request->id)},
                {"past",std::to_string(request->extent)},{"step",std::to_string(batches)}}));
        }
        const auto result=controller.RunNextBatch(session,stream,{{"stage","request_decode"},{"model","minimind"},
            {"export_receipt",receipt},{"step",std::to_string(batches)},
            {"state_extent_after",std::to_string(requests[0]->extent+1)},
            {"request_ids","caller-must-not-override"}});
        Check(result.lease==lease && result.results.size()==requests.size(),"model batch selected the wrong generation or request count");
        for(size_t row=0;row<requests.size();++row) {
            auto& request=*requests[row]; const auto& actual=result.results[row];
            Check(actual.request_id==request.id && actual.outputs.size()==1 && session.RequestExtent(request.id)==request.extent+1,
                  "model batch lost request ownership or committed extent");
            for(size_t output=0;output<17;++output) {
                const auto value=output==0?actual.outputs[0]:session.CopyRequestState(request.id,bindings[output-1].state_value_id);
                const auto values=Read(value), expected=Read(independent[row][output]);
                Check(values==expected,"request hot swap differs from independent LLVM logits/KV");
                independent_values+=values.size();
                if(initial_fixture) {
                    const auto bytes=File(root/("ref_"+std::to_string(request.fixture_case)+"_"+std::to_string(output)+".bin"));
                    const size_t batch=request.fixture_case==0?1:2;
                    Check(bytes.size()==batch*values.size()*sizeof(float),"request ONNX reference size changed");
                    std::vector<float> expected_row(values.size());
                    std::memcpy(expected_row.data(),bytes.data()+request.fixture_row*value.NBytes(),value.NBytes());
                    for(size_t i=0;i<values.size();++i) {
                        Check(std::isfinite(values[i]) && std::isfinite(expected_row[i]),"nonfinite request reference");
                        worst=std::max(worst,std::abs(double(values[i])-expected_row[i]));
                    }
                    onnx_values+=values.size();
                }
            }
            request.past={}; for(size_t i=1;i<17;++i) request.past.push_back(independent[row][i]);
            const auto logits=Read(actual.outputs[0]);
            const int64_t token=std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
            request.token=runtime::NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
            request.token.CopyFromBytes(&token,sizeof(token)); ++request.extent;
        }
        Check(worst<=5e-5 && SameStats(stats,ci::GetPrimitiveCacheStats()),"model request batch exceeded tolerance or compiled at runtime");
        ++batches; request_steps+=requests.size(); generations.push_back(lease->generation());
        (void)CompletedRun(*lease,"request_decode",receipt);
        return result;
    };
    for(const auto* request:{&a,&b,&c}) session.EnqueueRequest(request->id,{request->token});
    const auto retained=execute({&a,&c},first,true); const auto retained_values=Read(retained.results[0].outputs[0]);
    Check(session.RequestExtent(b.id)==0,"incompatible queued request was changed");
    std::cout << "[INFO] request model: replace unit 773 while B remains queued" << std::endl;
    const auto second=controller.CompileAndPublish({Request(mint(1,"batch-model-second"),first->compiled_graph(),{773})});
    Check(first->plan_abi()==second->plan_abi() && first->dispatch_key()==second->dispatch_key(),"model request route/ABI changed");
    for(size_t i=0;i<774;++i) Check((first->compiled_graph().artifact_pins()[i].record().artifact_key==
        second->compiled_graph().artifact_pins()[i].record().artifact_key)==(i!=773),"model replacement changed unselected code");
    (void)execute({&b},second,true);
    session.EnqueueRequest(a.id,{a.token}); session.ReleaseRequest(a.id);
    auto d=admit(0,0); Check(d.id!=a.id,"model request id was reused");
    Rejects([&] { session.EnqueueRequest(a.id,{a.token}); });
    session.EnqueueRequest(d.id,{d.token}); (void)execute({&d},second,true);
    for(const auto* request:{&b,&c,&d}) session.EnqueueRequest(request->id,{request->token});
    health->threshold=0;
    Check(controller.EvaluateHealth(second) && !controller.EvaluateHealth(second) && controller.Acquire(execution)==first,
          "completed model request evidence did not quarantine and roll back exactly once");
    (void)execute({&b,&d},first,false); (void)execute({&c},first,false);
    Check(controller.RunNextBatch(session,stream).results.empty() && Read(retained.results[0].outputs[0])==retained_values,
          "request departure invalidated output rows or left queued work");
    Check(batches==5 && request_steps==7 && onnx_values==99328 && independent_values==179968 &&
        generations==std::vector<hs::Generation>({1,2,2,1,1}),"model request evidence coverage changed");
    std::cout << "[PASS] full MiniMind request hot swap: batches=5 request_steps=7 calls=774; generations=1,2,2,1,1; "
                 "3870 batched vs 5418 independent LLVM calls; 16 KV states; reference_values=" << onnx_values
              << " independent_values=" << independent_values << " max_abs_error=" << worst
              << "; departure/reuse, queued rollback, no runtime compilation; route_bytes=" << route_bytes << "; receipt=" << receipt << '\n';
}

void TestMiniMindBoundedState(const std::filesystem::path& root,
                            const std::filesystem::path& prefill_root) {
    namespace rs = api::experimental::restricted_symbolic_shape::v1;
    using Adapter = rs::RestrictedSymbolicShapeAdapter;
    const auto line=[](const std::filesystem::path& path) {
        std::ifstream file(path); std::string value;
        Check(bool(std::getline(file,value)) && !value.empty(),"missing receipt: "+path.string()); return value;
    };
    size_t compared=0; double worst=0;
    const auto compare=[&](const std::vector<float>& values,const std::filesystem::path& path) {
        const auto bytes=File(path); Check(bytes.size()==values.size()*4,"bounded reference size mismatch: "+path.string());
        std::vector<float> expected(values.size()); std::memcpy(expected.data(),bytes.data(),bytes.size());
        for(size_t i=0;i<values.size();++i) {
            Check(std::isfinite(values[i]) && std::isfinite(expected[i]),"nonfinite bounded model value");
            worst=std::max(worst,std::abs(double(values[i])-expected[i]));
        }
        compared+=values.size(); Check(worst<5e-5,"bounded model differs from ONNX reference: "+path.string());
    };
    std::cout << "[INFO] bounded state model: compile decode" << std::endl;
    const auto source=frontend::LoadONNXShapeSource((root/"decode.json").string(),(root/"decode.params").string());
    Check(source.input_names.size()==17 && source.declared_output_types.size()==17,"bounded model ports changed");
    std::vector<rs::InputAxisSymbol> axes{{0,0,"B",1,3,1}};
    for(size_t i=1;i<17;++i) { axes.push_back({i,0,"B",1,3,1}); axes.push_back({i,1,"P",0,8,1}); }
    const auto mint=[&](int level,const std::string& name) {
        return Adapter::MintBoundedCompileRequest(Adapter::Prepare(source.function,Config(level,name),axes,source.declared_output_types));
    };
    const auto baseline=api::Compiler::CompileBounded(mint(0,"bounded-model-seed"));
    Check(baseline.plan().calls().size()==774,"bounded decode lost an ordinary model unit");
    std::vector<runtime::StateOutputBinding> bindings; std::vector<Array<int64_t>> shapes;
    for(size_t i=1;i<17;++i) {
        bindings.push_back({baseline.plan().input_value_ids()[i],baseline.plan().output_value_ids()[i],1,-1,1});
        shapes.push_back({1,8,4,96});
    }
    const auto graph=baseline.BindBoundedStateOutputs(bindings,shapes,1e20f);
    const auto plan=graph.plan();
    std::cout << "[INFO] bounded state model: prepare initial identity" << std::endl;
    Request initial(mint(0,"bounded-model-first"),graph,{773});
    const uint64_t route_bytes=initial.dispatch_key().canonical_bytes().size()+initial.plan_abi().canonical_bytes().size();
    Check(route_bytes<2ULL*1024*1024*1024,"bounded model exceeds explicit route budget");
    auto health=std::make_shared<BundleHealth>(); hs::Options options; options.health_authority=health;
    options.max_routes=1; options.max_route_metadata_bytes=route_bytes+64;
    options.max_quarantine_tombstone_bytes=2ULL*1024*1024*1024;
    hs::AdaptiveHotSwapController controller(options);
    std::cout << "[INFO] bounded state model: publish first generation" << std::endl;
    const auto first=controller.CompileAndPublish({std::move(initial)});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    auto session=controller.CreateStatefulSession(execution);
    std::vector<const void*> addresses; std::vector<float> logits;
    {
        std::cout << "[INFO] bounded state model: compile and run prefill" << std::endl;
        const auto input=frontend::LoadONNXShapeSource((prefill_root/"prefill.json").string(),(prefill_root/"prefill.params").string());
        const auto prefill=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(input.function,
            Config(0,"bounded-model-prefill"),{{0,0,"B",1,3,1},{0,1,"S",1,8,1}},input.declared_output_types)));
        Check(prefill.plan().calls().size()==742,"bounded prefill lost a model unit");
        const auto ids=runtime::NDArray::Empty({1,4},runtime::DataTypeFromString("int64"),Device::CPU());
        const auto bytes=File(root/"loop_seed_ids.bin"); Check(bytes.size()==ids.NBytes(),"invalid prefill seed IDs");
        ids.CopyFromBytes(bytes.data(),bytes.size());
        const auto stats=ci::GetPrimitiveCacheStats();
        const auto seed=runtime::RuntimeSession(prefill.module(),prefill.plan()).Run({ids},
            {{"stage","prefill"},{"model","minimind"},{"export_receipt",line(prefill_root/"export_receipt.txt")}});
        Check(seed.size()==17 && SameStats(stats,ci::GetPrimitiveCacheStats()),"prefill seed ports changed or execution compiled");
        const auto all_logits=Read(seed[0]); Check(all_logits.size()==4*6400,"invalid prefill vocabulary");
        logits.assign(all_logits.end()-6400,all_logits.end());
        compare(logits,root/"loop_seed_0.bin");
        for(size_t i=1;i<17;++i) {
            compare(Read(seed[i]),root/("loop_seed_"+std::to_string(i)+".bin"));
            session.InitializeState(bindings[i-1].state_value_id,seed[i],4);
            addresses.push_back(session.StateValue(bindings[i-1].state_value_id).storage().data());
        }
    }
    const auto receipt=line(root/"export_receipt.txt");
    std::ifstream steps(root/"loop_steps.txt");
    std::shared_ptr<const hs::GenerationLease> second;
    std::vector<hs::Generation> generations;
    for(int step=0;step<4;++step) {
        std::cout << "[INFO] bounded state model: step " << step << std::endl;
        if(step==1) {
            second=controller.CompileAndPublish({Request(mint(1,"bounded-model-second"),first->compiled_graph(),{773})});
            Check(second->plan_abi()==first->plan_abi() && second->dispatch_key()==first->dispatch_key(),"bounded replacement changed execution applicability");
            for(size_t i=0;i<774;++i) Check(
                (first->compiled_graph().artifact_pins()[i].record().artifact_key==second->compiled_graph().artifact_pins()[i].record().artifact_key)==(i!=773),
                "bounded model replacement changed an unselected artifact");
        }
        const int64_t token=std::max_element(logits.begin(),logits.end())-logits.begin();
        int64_t expected_token=-1,past=-1;
        Check(bool(steps>>expected_token>>past) && expected_token==token && past==4+step,"bounded greedy differs from independent reference");
        const auto ids=runtime::NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU()); ids.CopyFromBytes(&token,sizeof(token));
        const auto stats=ci::GetPrimitiveCacheStats();
        const auto result=controller.RunAsync(session,{ids},DeviceStream::Default(Device::CPU()),
            {{"stage","bounded_decode"},{"model","minimind"},{"export_receipt",receipt},{"past",std::to_string(past)},
             {"step",std::to_string(step)},{"token_id",std::to_string(token)},{"state_extent_after",std::to_string(past+1)}});
        Check(result.completion.IsReady() && result.outputs.size()==1 && result.lease==(step==1?second:first),"bounded model selected the wrong generation");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"bounded model Run touched cache");
        generations.push_back(result.lease->generation()); logits=Read(result.outputs[0]);
        compare(logits,root/("loop_ref_"+std::to_string(step)+"_0.bin"));
        for(size_t i=1;i<17;++i) {
            const auto id=bindings[i-1].state_value_id;
            const auto state=session.StateValue(id); const auto values=Read(state);
            const auto end=values.begin()+(past+1)*384;
            Check(session.StateExtent(id)==past+1 && state.storage().data()==addresses[i-1] &&
                std::all_of(end,values.end(),[](float value) { return value==1e20f; }),"bounded state address, extent or invalid capacity changed");
            compare(std::vector<float>(values.begin(),end),root/("loop_ref_"+std::to_string(step)+"_"+std::to_string(i)+".bin"));
        }
        (void)CompletedRun(*result.lease,"bounded_decode",receipt);
        if(step==1) {
            health->threshold=0;
            Check(controller.EvaluateHealth(second) && !controller.EvaluateHealth(second) && controller.Acquire(execution)==first,
                "bounded completed evidence did not quarantine and roll back exactly once");
        }
    }
    Check(generations==std::vector<hs::Generation>({first->generation(),second->generation(),first->generation(),first->generation()}),"bounded generation sequence changed");
    const auto before=Events(*first).size();
    const auto stats=ci::GetPrimitiveCacheStats();
    const auto ids=runtime::NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
    Rejects([&] { (void)controller.RunAsync(session,{ids},DeviceStream::Default(Device::CPU())); });
    const auto after=Events(*first);
    Check(after.size()==before+1 && SameStats(stats,ci::GetPrimitiveCacheStats()),"full bounded model capacity rejection launched work");
    const auto& rejected=*after.back().getAsObject();
    Check(String(rejected,"event_type")=="runtime_session_run" && String(rejected,"status")=="error" &&
        rejected.getObject("metrics")->getNumber("submit_count")==0,"capacity rejection did not record a zero-submit error");
    std::cout << "[PASS] full MiniMind bounded stateful hot swap: 742 prefill + 3096 decode LLVM calls; unit 773; "
                 "generations=1,2,1,1; 16 stable KV; extent=4->8; reference_values=" << compared
              << "; max_abs_error=" << worst << "; no runtime compilation; route_bytes=" << route_bytes << "; receipt=" << receipt << '\n';
}
#endif

void TestMiniMindState(const std::filesystem::path& root) {
    const auto line=[&](const char* name) {
        std::ifstream stream(root/name); std::string value;
        Check(bool(std::getline(stream,value)) && !value.empty(),std::string("missing fixture field: ")+name);
        return value;
    };
    const auto floats=[&](const std::filesystem::path& path) {
        const auto bytes=File(path); Check(bytes.size()%sizeof(float)==0,"invalid float reference size");
        std::vector<float> values(bytes.size()/sizeof(float));
        std::memcpy(values.data(),bytes.data(),bytes.size()); return values;
    };
    size_t compared=0;
    const auto compare=[&](const runtime::NDArray& actual, const std::filesystem::path& path) {
        const auto values=Read(actual), expected=floats(path);
        Check(values.size()==expected.size(),"model reference shape mismatch: "+path.string());
        double worst=0;
        for(size_t i=0;i<values.size();++i) {
            Check(std::isfinite(values[i]) && std::isfinite(expected[i]),"nonfinite stateful model value");
            worst=std::max(worst,std::fabs(double(values[i])-expected[i]));
        }
        compared+=values.size(); Check(worst<=1e-4,"stateful model differs from ONNX: "+path.string());
        return worst;
    };
    const auto prepare=[&](const char* name) {
        const auto imported=frontend::LoadONNXImportSpec((root/(std::string(name)+".json")).string(),
            (root/(std::string(name)+".params")).string());
        auto function=relay::InferTypePass(imported.function);
        return relay::InferTypePass(relay::RunRelayPassPipeline(function,{kxc::String("fold_constant"),kxc::String("simplify_expr")}));
    };
    const int64_t capacity=std::stoll(line("capacity.txt"));
    int64_t extent=std::stoll(line("seed_extent.txt"));
    const float sentinel=std::stof(line("sentinel.txt"));
    Check(capacity==32 && extent==16 && line("layout.txt")=="1 4 96 8" &&
          line("graph.txt")=="decode_capacity" && line("sampling.txt")=="greedy_argmax" && std::fabs(sentinel)<=1e6,
          "stateful adaptive fixture must be the locked eight-layer B1/C32/P16 model");
    const auto function=prepare("decode_capacity");
    std::cout << "[INFO] state model: compile baseline" << std::endl;
    const auto baseline=api::Compiler::Compile(function,Config(0,"state-model-seed"));
    const auto input_ids=baseline.plan().input_value_ids(), output_ids=baseline.plan().output_value_ids();
    Check(input_ids.size()==19 && output_ids.size()==17,"stateful model boundary is incomplete");
    std::vector<runtime::StateOutputBinding> bindings;
    for(size_t i=0;i<16;++i) bindings.push_back({input_ids[i+3],output_ids[i+1],1,capacity,1});
    const auto graph=baseline.BindStateOutputs(bindings,sentinel);
    const auto plan=graph.plan();
    const int64_t replaced=static_cast<int64_t>(plan.calls().size())-1;
    std::cout << "[INFO] state model: prepare initial identity" << std::endl;
    Request initial(function,Config(0,"state-model-first"),graph,{replaced});
    const uint64_t route_bytes=initial.dispatch_key().canonical_bytes().size()+initial.plan_abi().canonical_bytes().size();
    Check(route_bytes<2ULL*1024*1024*1024,"stateful model route exceeds the explicit fixture budget");
    auto health=std::make_shared<BundleHealth>();
    hs::Options options; options.health_authority=health; options.max_routes=1;
    options.max_route_metadata_bytes=route_bytes+64;
    options.max_quarantine_tombstone_bytes=2ULL*1024*1024*1024;
    hs::AdaptiveHotSwapController controller(options);
    std::cout << "[INFO] state model: publish first generation" << std::endl;
    const auto first=controller.CompileAndPublish({std::move(initial)});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    std::cout << "[INFO] state model: initialize from actual prefill" << std::endl;
    auto session=controller.CreateStatefulSession(execution);
    const auto stream=DeviceStream::Default(Device::CPU());
    const auto prefill=api::Compiler::Compile(prepare("prefill"),Config(0,"state-model-prefill"));
    Check(prefill.plan().input_value_ids().size()==1,"text prefill requires one input");
    runtime::NDArray ids;
    for(const auto& spec:prefill.plan().values()) if(spec->value_id==prefill.plan().input_value_ids()[0]) {
        ids=runtime::NDArray::Empty(spec.shape(),spec->dtype,Device::CPU());
    }
    const auto input_bytes=File(root/"prefill_input_ids.bin");
    Check(ids.defined() && ids.NBytes()==input_bytes.size(),"prefill input fixture has the wrong shape");
    ids.CopyFromBytes(input_bytes.data(),input_bytes.size());
    const auto prefill_outputs=runtime::RuntimeSession(prefill.module(),prefill.plan()).Run({ids},
        {{"stage","prefill"},{"model","minimind"},{"export_receipt",line("prefill_export_receipt.txt")}});
    Check(prefill_outputs.size()==17,"actual prefill must produce every KV seed");
    double worst_prefill=compare(prefill_outputs[0],root/"prefill_reference_logits.bin");
    std::vector<const void*> addresses;
    for(size_t i=0;i<16;++i) {
        const auto name="seed_present_"+std::string(i%2?"v_":"k_")+std::to_string(i/2)+".bin";
        worst_prefill=std::max(worst_prefill,compare(prefill_outputs[i+1],root/name));
        session.InitializeState(bindings[i].state_value_id,prefill_outputs[i+1],extent);
        addresses.push_back(session.StateValue(bindings[i].state_value_id).storage().data());
    }
    const auto vocab=floats(root/"prefill_logits.bin").size();
    const auto greedy=[&](const runtime::NDArray& value) {
        const auto values=Read(value); Check(vocab>0 && values.size()>=vocab,"missing logits row");
        const auto begin=values.end()-static_cast<std::ptrdiff_t>(vocab);
        return static_cast<int64_t>(std::max_element(begin,values.end())-begin);
    };
    int64_t next_token=greedy(prefill_outputs[0]);
    const auto receipt=line("export_receipt.txt");
    std::ifstream step_file(root/"steps.txt"); int index=0,step_count=0; int64_t token=0,past=0;
    std::shared_ptr<const hs::GenerationLease> second;
    std::vector<hs::Generation> generations;
    double worst_logits=0,worst_state=0;
    while(step_file>>index>>token>>past) {
        Check(index==step_count && past==extent && token==next_token,"greedy model step differs from its independent reference");
        std::cout << "[INFO] state model: step " << index << std::endl;
        if(index==1) {
            second=controller.CompileAndPublish({Request(function,Config(1,"state-model-second"),first->compiled_graph(),{replaced})});
            Check(second->plan_abi()==first->plan_abi() && second->generation()>first->generation() &&
                  second->compiled_graph().artifact_pins()[replaced].record().artifact_key!=first->compiled_graph().artifact_pins()[replaced].record().artifact_key,
                  "full stateful model did not replace the compiled primitive");
        }
        auto token_tensor=runtime::NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
        token_tensor.CopyFromBytes(&token,sizeof(token));
        auto position=runtime::NDArray::Empty({1},runtime::DataTypeFromString("int64"),Device::CPU());
        position.CopyFromBytes(&extent,sizeof(extent));
        std::vector<float> mask(static_cast<size_t>(capacity+1),0);
        std::fill(mask.begin(),mask.begin()+extent,1); mask.back()=1;
        const auto attention_mask=Tensor(mask).CreateView({1,capacity+1},{},0);
        const auto stats=ci::GetPrimitiveCacheStats();
        const auto result=controller.RunAsync(session,{token_tensor,position,attention_mask},stream,
            {{"stage","decode"},{"model","minimind"},{"export_receipt",receipt},
             {"state_extent",std::to_string(extent)},{"token_index",std::to_string(index)},{"token_id",std::to_string(token)}});
        Check(result.completion.IsReady() && result.outputs.size()==1 && result.lease==(index==1?second:first),
              "stateful model used the wrong generation or state boundary");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"full stateful model execution touched the primitive cache");
        generations.push_back(result.lease->generation());
        worst_logits=std::max(worst_logits,compare(result.outputs[0],root/("ref_step"+std::to_string(index)+"_logits.bin")));
        for(size_t i=0;i<16;++i) {
            const auto id=bindings[i].state_value_id;
            Check(session.StateValue(id).storage().data()==addresses[i] && session.StateExtent(id)==extent+1,
                  "full model state moved or lost its committed extent across replacement");
            worst_state=std::max(worst_state,compare(session.StateValue(id),
                root/("ref_step"+std::to_string(index)+"_state_"+std::to_string(i)+".bin")));
        }
        (void)CompletedRun(*result.lease,"decode",receipt);
        if(index==1) {
            health->threshold=0;
            Check(controller.EvaluateHealth(second) && !controller.EvaluateHealth(second) && controller.Acquire(execution)==first,
                  "full model completed evidence failed to quarantine and roll back once");
        }
        next_token=greedy(result.outputs[0]); ++extent; ++step_count;
    }
    Check(step_count==4 && extent==20 && generations==std::vector<hs::Generation>({first->generation(),second->generation(),first->generation(),first->generation()}),
          "full model must execute four greedy steps across generation 1/2/1/1");
    std::cout << "[PASS] full MiniMind stateful hot swap: " << prefill.plan().calls().size() << " prefill + "
              << step_count*plan.calls().size() << " decode LLVM calls, unit " << replaced
              << "; generations=1,2,1,1; 16 stable KV, extent=16->20; reference_values=" << compared
              << "; max_abs_error prefill=" << worst_prefill << " logits=" << worst_logits << " state=" << worst_state
              << "; no runtime compilation; route_bytes=" << route_bytes << "; receipt=" << receipt << '\n';
}

void TestMiniMindPrefill() {
    const char* directory=std::getenv("KXC_MINIMIND_ADAPTIVE_PREFILL_DIR");
    if(!directory || !*directory) {
        std::cout << "[SKIP] full MiniMind prefill: set KXC_MINIMIND_ADAPTIVE_PREFILL_DIR to the decode-loop fixture\n"; return;
    }
    const std::filesystem::path root(directory);
    const auto imported=frontend::LoadONNXImportSpec((root/"prefill.json").string(),(root/"prefill.params").string());
    auto function=relay::InferTypePass(imported.function);
    function=relay::RunRelayPassPipeline(function,{kxc::String("fold_constant"),kxc::String("simplify_expr")});
    function=relay::InferTypePass(function);
    const auto baseline=api::Compiler::Compile(function,Config(0,"model-seed"));
    Check(baseline.plan().state_value_ids().empty() && imported.output_names.size()==17,"fixture must be a stateless eight-layer prefill");
    const auto input_bytes=File(root/"prefill_input_ids.bin");
    Array<int64_t> shape;
    for(const auto& value:baseline.plan().values()) {
        if(value->value_id==baseline.plan().input_value_ids()[0]) shape=value.shape();
    }
    const auto input=runtime::NDArray::Empty(shape,runtime::DataTypeFromString("int64"),Device::CPU());
    Check(input_bytes.size()==input.NBytes(),"prefill input shape differs from fixture");
    input.CopyFromBytes(input_bytes.data(),input_bytes.size());
    std::ifstream receipt_file(root/"prefill_export_receipt.txt"); std::string receipt;
    Check(bool(std::getline(receipt_file,receipt)) && !receipt.empty(),"missing model export receipt");
    const int64_t replaced=static_cast<int64_t>(baseline.plan().calls().size())-1;
    Request initial(function,Config(0,"model-baseline"),baseline,{replaced});
    const uint64_t route_bytes=initial.dispatch_key().canonical_bytes().size()+initial.plan_abi().canonical_bytes().size();
    Check(route_bytes<2ULL*1024*1024*1024,"locked fixture exceeds the explicit 2 GiB route ceiling");
    std::cout << "[INFO] model dispatch/ABI canonical bytes: " << route_bytes << std::endl;
    hs::Options options; options.health_authority=std::make_shared<BundleHealth>();
    // Canonical plan ABI includes the model's constant bytes and exceeds the
    // small-graph default. This budget covers only one locked model route.
    // Keep a finite, explicit fixture budget; do not change controller defaults.
    options.max_routes=1;
    options.max_route_metadata_bytes=route_bytes+64;  // Two length-prefixed canonical keys.
    hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({std::move(initial)});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    const auto run=[&] {
        const auto before=ci::GetPrimitiveCacheStats();
        auto result=controller.RunAsync(execution,{input},DeviceStream::Default(Device::CPU()),{{"stage","prefill"},{"export_receipt",receipt}});
        result.completion.Wait();
        Check(SameStats(before,ci::GetPrimitiveCacheStats()),"model Run touched the primitive cache");
        Check(result.outputs.size()==17,"model lost logits or KV outputs");
        (void)CompletedRun(*result.lease,"prefill",receipt);
        Check(controller.EvaluateHealth(result.lease) && !controller.EvaluateHealth(result.lease),"model evidence is not one-shot");
        return result;
    };
    const auto old=run();
    // opt 0/1 preserve this pre-folded model's ordered call topology. opt 2
    // rewrites reshapes and is not compatible with a one-unit replacement.
    const auto second=controller.CompileAndPublish({Request(function,Config(1,"model-candidate"),first->compiled_graph(),{replaced})});
    Check(second->generation()>first->generation() && second->dispatch_key()==first->dispatch_key() && second->plan_abi()==first->plan_abi() &&
          second->compiled_graph().artifact_pins()[replaced].record().artifact_key!=first->compiled_graph().artifact_pins()[replaced].record().artifact_key,
          "model did not replace an actual compiled primitive");
    const auto current=run(); Check(current.lease==second && old.lease==first,"model execution used the wrong route head");
    double worst=0;
    for(size_t i=0;i<current.outputs.size();++i) {
        const auto a=Read(old.outputs[i]), b=Read(current.outputs[i]);
        Check(a==b,"model candidates differ numerically");
        const auto bytes=File(root/(i==0?"prefill_reference_logits.bin":"seed_"+imported.output_names[i]+".bin"));
        Check(bytes.size()==a.size()*sizeof(float),"model reference shape differs");
        std::vector<float> expected(a.size()); std::memcpy(expected.data(),bytes.data(),bytes.size());
        for(size_t j=0;j<a.size();++j) {
            Check(std::isfinite(a[j]) && std::isfinite(expected[j]),"model contains a non-finite result");
            worst=std::max(worst,std::fabs(double(a[j])-expected[j]));
        }
    }
    Check(worst<=1e-4,"model differs from its ONNX reference");
    std::cout << "[PASS] full MiniMind prefill: " << baseline.plan().calls().size() << " LLVM calls per generation, unit " << replaced
              << " replaced, 17 outputs bitwise equal, ONNX max absolute error " << worst << ", route/ABI canonical bytes "
              << first->dispatch_key().canonical_bytes().size()+first->plan_abi().canonical_bytes().size() << ", receipt " << receipt << '\n';
}
void RunTest() {
    ci::ClearPrimitiveCacheForTesting();
    const auto function=Graph();
    const auto config=Config(1,"baseline");
    const auto baseline=api::Compiler::Compile(function,Config(1,"seed"));
    auto health=std::make_shared<BundleHealth>();
    std::vector<hs::Event> health_events;
    hs::Options options; options.health_authority=health;
    options.observer=[&](const hs::Event& event) {
        if(event.kind==hs::EventKind::kHealthDecision || event.kind==hs::EventKind::kQuarantined || event.kind==hs::EventKind::kRolledBack) health_events.push_back(event);
    };
    hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({Request(function,config,baseline,{0,1})});
    const Execution execution(first->dispatch_key(),first->plan_abi());
    const Array<runtime::NDArray> inputs{Tensor({1,2}),Tensor({3,4})};
    const auto different_function=Graph(3);
    const auto different_config=Config(1,"different-length",false);
    const auto different_graph=api::Compiler::Compile(different_function,different_config);
    const Request different(different_function,different_config,different_graph,{0,1});
    const auto preflight=ci::GetPrimitiveCacheStats();
    const auto event_count=Events(*first).size();
    for(const auto& invalid:{Execution(different.dispatch_key(),first->plan_abi()),Execution(first->dispatch_key(),different.plan_abi())}) {
        Rejects([&] { (void)controller.Acquire(invalid); });
        Rejects([&] { (void)controller.RunAsync(invalid,inputs,DeviceStream::Default(Device::CPU())); });
    }
    Rejects([&] { (void)hs::preparation::PrepareCandidate(Request(function,config,baseline,{0,1}),baseline,""); });
    Check(SameStats(preflight,ci::GetPrimitiveCacheStats()) && Events(*first).size()==event_count,
          "invalid route, ABI or receipt performed compile or execution work");
    hs::Options invalid_options; invalid_options.generation_authority=std::make_shared<WrongReceipt>();
    hs::AdaptiveHotSwapController invalid_controller(invalid_options);
    const auto invalid_receipt=invalid_controller.Submit({Request(function,Config(1,"wrong-receipt"),baseline,{0,1})}).Wait();
    Check(!invalid_receipt.ready() && invalid_receipt.failure.category==hs::FailureCategory::kPermanent &&
          invalid_receipt.failure.diagnostic.find("receipt mismatch")!=std::string::npos,"mismatched receipt produced a lease");
    Rejects([&] { (void)invalid_controller.RunAsync(execution,inputs,DeviceStream::Default(Device::CPU())); });
    const auto run=[&](const char* stage) {
        auto result=controller.RunAsync(execution,inputs,DeviceStream::Default(Device::CPU()),{{"stage",stage},{"export_receipt","cpp:add-mul-float32-v1"}});
        result.completion.Wait(); Check(Read(result.outputs[0])==std::vector<float>({12,24}),"wrong LLVM output"); return result;
    };
    Check(!controller.EvaluateHealth(first) && health->verifications==0,"missing evidence was consumed");
    auto old=run("baseline");
    Check(CompletedRun(*first).duration>=0,"baseline has no run proof");
    health->wrong_generation=true;
    Check(!controller.EvaluateHealth(first) && health->verifications==0,"wrong generation reached consumption");
    health->wrong_generation=false; health->empty_evidence=true;
    Check(!controller.EvaluateHealth(first) && health->verifications==0,"empty evidence reached consumption");
    health->empty_evidence=false;
    Check(controller.EvaluateHealth(first) && !controller.EvaluateHealth(first),"health evidence is not one-shot");
    const auto second=controller.CompileAndPublish({Request(function,Config(2,"candidate"),first->compiled_graph(),{1})});
    Check(second->generation()>first->generation() && second->plan_abi()==first->plan_abi() &&
          second->compiled_graph().artifact_pins()[1].record().artifact_key!=first->compiled_graph().artifact_pins()[1].record().artifact_key,
          "replacement did not change the compiled artifact");
    Check(controller.Acquire(execution)==second,"route did not move to candidate");
    const auto before=ci::GetPrimitiveCacheStats();
    auto current=run("candidate");
    Check(SameStats(before,ci::GetPrimitiveCacheStats()),"Run touched the primitive cache");
    const auto verified=health->verifications;
    Check(!controller.EvaluateHealth(first) && health->verifications==verified,"stale generation consumed evidence");
    health->throw_evaluation=true; Check(!controller.EvaluateHealth(second),"health exception was accepted"); health->throw_evaluation=false;
    for(const char* field:{"generation","dispatch_key","plan_abi","plan_variant","validation_receipt","adaptive_contract"}) {
        Rejects([&] { (void)controller.RunAsync(execution,inputs,DeviceStream::Default(Device::CPU()),{{field,"forged"}}); });
    }
    health->threshold=0;
    Check(CompletedRun(*second).duration>0 && controller.EvaluateHealth(second),"measured run did not trigger explicit quarantine");
    Check(controller.Acquire(execution)==first && !controller.EvaluateHealth(second),"rollback or stale rejection failed");
    auto rolled_back=run("rollback");
    Check(rolled_back.lease==first && Read(old.outputs[0])==Read(current.outputs[0]),"rollback or held output changed");
    const auto quarantined=controller.Submit({Request(function,Config(2,"tombstone"),first->compiled_graph(),{1})}).Wait();
    Check(!quarantined.ready() && quarantined.failure.category==hs::FailureCategory::kPermanent &&
          quarantined.failure.diagnostic.find("quarantined")!=std::string::npos && controller.Acquire(execution)==first,
          "quarantined selection was republished");
    for(const auto& event:health_events) {
        Check(event.dispatch_key_digest==first->dispatch_key().digest() && event.plan_abi_digest==first->plan_abi().digest(),"health event lost route association");
    }
    Check(std::any_of(health_events.begin(),health_events.end(),[](const auto& e){return e.kind==hs::EventKind::kRolledBack;}),"no rollback event");
    const auto no_profile=api::Compiler::Compile(function,Config(1,"off",false));
    Check(no_profile.artifact_pins()[0].record().artifact_key==baseline.artifact_pins()[0].record().artifact_key &&
          no_profile.artifact_pins()[1].record().artifact_key==baseline.artifact_pins()[1].record().artifact_key,"profiling changed artifact identity");
    Check(Read(runtime::RuntimeSession(no_profile.module(),no_profile.plan()).Run(inputs)[0])==std::vector<float>({12,24}),"profiling-off result differs");
    std::cout << "[PASS] actual LLVM baseline/candidate/rollback, generation bundle association, one-shot health, stale/missing/forged evidence and profiling-off equivalence\n";
}
}  // namespace
#endif
int main(int argc, char** argv) {
#if KXC_USE_LLVM
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    if(argc==2 && std::string(argv[1])=="--request-model") {
        const char* decode=std::getenv("KXC_MINIMIND_BOUNDED_DECODE_DIR");
        if(!decode || !*decode) { std::cout << "[SKIP] bounded decode fixture required\n"; return 77; }
        try { TestMiniMindRequestBatching(decode); return 0; }
        catch(const std::exception& e) { std::cerr << "[FAIL] " << e.what() << '\n'; return 1; }
    }
    if(argc==2 && std::string(argv[1])=="--bounded-state-model") {
        const char* decode=std::getenv("KXC_MINIMIND_BOUNDED_DECODE_DIR");
        const char* prefill=std::getenv("KXC_MINIMIND_BOUNDED_PREFILL_DIR");
        if(!decode || !*decode || !prefill || !*prefill) { std::cout << "[SKIP] bounded prefill/decode fixtures required\n"; return 77; }
        try { TestMiniMindBoundedState(decode,prefill); return 0; }
        catch(const std::exception& e) { std::cerr << "[FAIL] " << e.what() << '\n'; return 1; }
    }
#endif
    if(argc==2 && std::string(argv[1])=="--state-model") {
        const char* directory=std::getenv("KXC_MINIMIND_ADAPTIVE_STATE_DIR");
        if(!directory || !*directory) { std::cout << "[SKIP] set KXC_MINIMIND_ADAPTIVE_STATE_DIR\n"; return 77; }
        try { TestMiniMindState(directory); return 0; }
        catch(const std::exception& e) { std::cerr << "[FAIL] " << e.what() << '\n'; return 1; }
    }
    if(argc!=1) { std::cerr << "usage: adaptive_runtime_test [--state-model]\n"; return 1; }
    try { RunTest(); TestObserverIsolation(); TestRunningGenerationLifetime(); TestStatefulReplacement(); TestStatefulPublicationLifetime(); TestStatefulReplacementFailure();
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
        TestBoundedReplacement();
        TestBoundedWiringRejection();
        TestRequestBatchReplacement();
        TestRequestBatchPublicationLifetime();
#endif
        TestMiniMindPrefill(); return 0; } catch(const std::exception& e) { std::cerr << "[FAIL] " << e.what() << '\n'; return 1; }
#else
    std::cout << "[SKIP] adaptive runtime requires LLVM\n"; return 0;
#endif
}
