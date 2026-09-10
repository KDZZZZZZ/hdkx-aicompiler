// Real LLVM/CUDA request admission, homogeneous merging, departure and KV-slot reuse.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/memory_plan.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/device_api.h"
#include "support/request_batching_graph.h"

namespace {
using namespace kxc;
using runtime::NDArray;
using runtime::RuntimeSession;
namespace ci = api::internal;
bool cuda = false;
std::vector<DeviceStream> streams;
size_t next_stream = 0;
Device TestDevice() { return cuda ? Device::CUDA() : Device::CPU(); }
std::vector<runtime::RequestResult> Batch(const RuntimeSession& session,
                                         const runtime::ExecutionMetadata& metadata = {}) {
    return cuda ? session.RunNextBatch(streams.at(next_stream++ % 2),metadata)
                : session.RunNextBatch(metadata);
}

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Action> void Rejects(Action action, const std::string& message) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(message) != std::string::npos,
              "wrong rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + message);
}
NDArray Tensor(Array<int64_t> shape, float start = -3, Device device = TestDevice()) {
    auto value = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), device);
    std::vector<float> data(value.NBytes() / 4);
    for (size_t i = 0; i < data.size(); ++i) data[i] = start + static_cast<float>(i);
    if (!data.empty()) value.CopyFromBytes(data.data(), value.NBytes());
    return value;
}
std::vector<float> Read(const NDArray& value) {
    std::vector<float> data(value.NBytes() / 4);
    if (!data.empty()) value.CopyToBytes(data.data(), value.NBytes());
    return data;
}
api::CompiledGraph Compile(const api::CompileConfig& config, int64_t axis = 1, int64_t append = 1) {
    return api::Compiler::CompileBounded(test_support::RequestBatchingGraph(config,axis,append));
}
runtime::ExecutablePlan StatePlan(const api::CompiledGraph& graph, int64_t slots = 3,
                                  int64_t axis = 1, int64_t append = 1) {
    auto shape = axis == 1 ? Array<int64_t>{slots,6,4} : Array<int64_t>{slots,2,6,4};
    return graph.plan().BindBoundedStateOutputs(
        {{graph.plan().input_value_ids()[0],graph.plan().output_value_ids()[1],axis,-1,append}},
        {shape},1e20f);
}
api::PlanAbiFingerprint Abi(const api::CompiledGraph& graph, const runtime::ExecutablePlan& plan) {
    std::vector<api::OrderedArtifactIdentity> artifacts;
    for (size_t i = 0; i < plan.calls().size(); ++i) {
        artifacts.push_back({i,std::string(plan.calls()[i]->symbol),graph.artifact_pins()[i].record().artifact_key});
    }
    return api::BuildPlanAbiFingerprint(graph.module(),plan,artifacts);
}
bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
        a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
        a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
size_t Submits(const std::shared_ptr<profiling::ProfileContext>& profile, bool all_activity = false) {
    profile->Flush();
    std::ifstream in(std::filesystem::path(profile->bundle_dir())/"events.jsonl");
    Check(in.good(),"missing request batch profile");
    std::string line; size_t count = 0;
    while (std::getline(in,line)) {
        count += line.find("\"event_type\":\"kernel_submit\"") != std::string::npos ||
            (all_activity && (line.find("\"event_type\":\"alloc\"") != std::string::npos ||
                             line.find("\"event_type\":\"copy\"") != std::string::npos));
    }
    return count;
}

void TestQueue(const api::CompiledGraph& graph, const std::shared_ptr<profiling::ProfileContext>& profile) {
    const auto base = StatePlan(graph), plan = base.BindRequestBatching(2);
    const int64_t state_id = plan.state_value_ids()[0];
    Check(Abi(graph,plan) != Abi(graph,base) && Abi(graph,plan) != Abi(graph,base.BindRequestBatching(1)),
          "request contract or max batch absent from identity");
    Check(Abi(graph,plan).canonical_bytes().find("executable-plan-abi-v12-request-batching-v1") != std::string::npos,
          "request ABI version missing");
    Rejects([&] { (void)graph.plan().BindRequestBatching(2); },"bounded stateful");
    Rejects([&] { (void)base.BindRequestBatching(0); },"max batch");
    Rejects([&] { (void)base.BindRequestBatching(4); },"max batch");
    const auto reject_guards = [&](std::vector<runtime::GraphInputAxisGuard> guards) {
        Rejects([&] { (void)runtime::ExecutablePlan(base.values(),base.calls(),base.input_value_ids(),
            base.constant_value_ids(),base.output_value_ids(),base.state_value_ids(),base.mode(),
            guards,{},-1,base.state_output_bindings(),runtime::RequestBatchingContract{2}); },"Request batching");
    };
    auto guards = base.graph_input_guards(); guards[0].lower = 2; reject_guards(guards);
    guards = base.graph_input_guards(); guards[2].equal_to = std::nullopt; reject_guards(guards);
    guards = base.graph_input_guards(); guards.back().equal_to = runtime::GraphInputAxisReference{0,0};
    reject_guards(guards);
    Rejects([&] { (void)runtime::internal::PlanMemory(plan); },"does not reuse");
    RuntimeSession session(graph.module(),plan);
    RuntimeSession ordinary(graph.module(),base);
    Rejects([&] { (void)ordinary.AdmitRequest(); },"explicit request batching");
    Rejects([&] { (void)session.Run({}); },"request API");
    Rejects([&] { (void)session.StateValue(state_id); },"request API");
    Rejects([&] { (void)session.StateExtent(state_id); },"request API");
    Rejects([&] { session.InitializeState(state_id,Tensor({3,0,4}),0); },"request API");
    Check(session.RunNextBatch({}).empty(),"empty queue did work");
    const auto idle_activity = Submits(profile,true);
    Rejects([&] { (void)session.RunNextBatch(DeviceStream{},{}); },"defined DeviceStream");
    if (cuda) {
        Rejects([&] { (void)session.RunNextBatch(DeviceStream::Default(Device::CPU()),{}); },"stream device");
        Rejects([&] { (void)session.AdmitRequest({Tensor({1,1,4},0,Device::CPU())},1); },"initial state layout");
    }
    Check(Submits(profile,true) == idle_activity,"empty-queue preflight allocated, copied or launched");
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto checkpoint = Submits(profile);
    Rejects([&] { (void)session.AdmitRequest({},1); },"all initial states");
    Rejects([&] { (void)session.AdmitRequest({Tensor({2,1,4})},1); },"initial state");
    Rejects([&] { (void)session.AdmitRequest({Tensor({1,0,4})},1); },"too short");
    auto seed = Tensor({1,1,4},10);
    const auto a = session.AdmitRequest({seed},1);
    const auto b = session.AdmitRequest();
    const auto c = session.AdmitRequest({Tensor({1,1,4},20)},1);
    Rejects([&] { (void)session.AdmitRequest(); },"slots are full");
    auto input = Tensor({1,1,4});
    const auto token = Read(input);
    if (cuda) {
        const auto activity = Submits(profile,true);
        Rejects([&] { session.EnqueueRequest(a,{Tensor({1,1,4},0,Device::CPU()),Tensor({1,2})}); },"device");
        Check(Submits(profile,true) == activity,"wrong-device input was copied before rejection");
    }
    Rejects([&] { session.EnqueueRequest(a,{}); },"input count");
    Rejects([&] { session.EnqueueRequest(a,{Tensor({2,1,4}),Tensor({1,2})}); },"batch must be one");
    Rejects([&] { session.EnqueueRequest(a,{Tensor({1,2,4}),Tensor({1,2})}); },"shape");
    Rejects([&] { session.EnqueueRequest(a,{input,Tensor({1,5})}); },"guard");
    session.EnqueueRequest(a,{input,Tensor({1,2})});
    session.EnqueueRequest(b,{input,Tensor({1,2})});
    session.EnqueueRequest(c,{input,Tensor({1,2})});
    Rejects([&] { session.EnqueueRequest(a,{input,Tensor({1,2})}); },"already has");
    Check(Submits(profile) == checkpoint,"admission or enqueue launched a kernel");
    input.CopyFrom(Tensor({1,1,4},1000));
    seed.CopyFrom(Tensor({1,1,4},2000));
    const auto queued_activity = Submits(profile,true);
    Rejects([&] { (void)session.RunNextBatch(DeviceStream{},{}); },"defined DeviceStream");
    if (cuda) Rejects([&] { (void)session.RunNextBatch(DeviceStream::Default(Device::CPU()),{}); },"stream device");
    Check(Submits(profile,true) == queued_activity,"invalid stream packed or launched queued inputs");
    const auto results = Batch(session,{{"case","merge_ac_skip_b"}});
    Check(results.size() == 2 && results[0].request_id == a && results[1].request_id == c,
          "same-P peers were not merged in FIFO order");
    Check(Submits(profile)-checkpoint == graph.plan().calls().size(),"two requests did not use one graph run");
    const auto retained = results[0].outputs[0];
    auto expected = std::vector<float>{10,11,12,13}; expected.insert(expected.end(),token.begin(),token.end());
    Check(Read(session.CopyRequestState(a,state_id)) == expected,"snapshot or scatter changed request A");
    for (float& value : expected) value = std::max(value,0.0f);
    Check(Read(retained) == expected,"merged result differs from scalar oracle");
    Check(session.RequestExtent(a) == 2 && session.RequestExtent(c) == 2 && session.RequestExtent(b) == 0,
          "batch committed the wrong request extent");
    const auto first_b = Batch(session);
    Check(first_b.size() == 1 && first_b[0].request_id == b && session.RequestExtent(b) == 1,
          "different-P oldest request was not executed next");
    auto diagnostic = session.CopyRequestState(a,state_id);
    diagnostic.CopyFrom(Tensor({1,2,4},5000));
    Check(Read(session.CopyRequestState(a,state_id))[0] == 10,"diagnostic exposed mutable slot storage");
    session.EnqueueRequest(a,{Tensor({1,1,4}),Tensor({1,2})});
    session.ReleaseRequest(a); // queued cancellation and ABA-safe reuse
    const auto replacement = session.AdmitRequest();
    Check(replacement != a && session.RequestExtent(replacement) == 0,"slot reuse retained old identity or extent");
    Rejects([&] { session.EnqueueRequest(a,{input,Tensor({1,2})}); },"released request");
    Rejects([&] { session.ReleaseRequest(a); },"released request");
    Rejects([&] { (void)session.CopyRequestState(a,state_id); },"released request");
    Check(Batch(session).empty(),"released queued work still ran");
    session.EnqueueRequest(replacement,{Tensor({1,1,4},-8),Tensor({1,2})});
    const auto reused = Batch(session);
    Check(Read(reused[0].outputs[0]) == std::vector<float>({0,0,0,0}),"released KV leaked into replacement");
    Check(Read(retained) == expected,"request departure invalidated returned output");
    // Equal P but different non-batch shape must remain separate.
    session.EnqueueRequest(b,{Tensor({1,1,4}),Tensor({1,2})});
    session.EnqueueRequest(replacement,{Tensor({1,1,4}),Tensor({1,3})});
    Check(Batch(session).size() == 1 && Batch(session).size() == 1,
          "different non-batch input shapes were merged");
    // Three equally eligible requests are split by max_batch_size, then the old remainder runs first.
    session.EnqueueRequest(b,{Tensor({1,1,4}),Tensor({1,2})});
    session.EnqueueRequest(c,{Tensor({1,1,4}),Tensor({1,2})});
    session.EnqueueRequest(replacement,{Tensor({1,1,4}),Tensor({1,2})});
    const auto pair = Batch(session);
    Check(pair.size() == 2 && pair[0].request_id == b && pair[1].request_id == c,"batch size ceiling drifted");
    const auto last = Batch(session);
    Check(last.size() == 1 && last[0].request_id == replacement,"FIFO remainder lost");
    while (session.RequestExtent(b) < 6) {
        session.EnqueueRequest(b,{Tensor({1,1,4}),Tensor({1,2})}); (void)Batch(session);
    }
    const auto full_count = Submits(profile);
    Rejects([&] { session.EnqueueRequest(b,{input,Tensor({1,2})}); },"capacity before launch");
    Check(session.RequestExtent(b) == 6 && Submits(profile) == full_count,"capacity rejection mutated state or launched");
    RuntimeSession independent(graph.module(),plan);
    const auto other = independent.AdmitRequest();
    Check(independent.RequestExtent(other) == 0,"another session inherited request state");
    Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"request serving touched primitive cache");
    std::cout << "request batching: 3 slots, max batch 2, homogeneous FIFO, snapshot, departure, reuse, capacity, identity passed\n";
}

class Observer final : public runtime::ExecutionObserver {
public:
    std::function<void()> begin;
    size_t submitted{0};
    runtime::ExecutionRunCorrelation OnRunStart(const runtime::ExecutionRunStart&) override { return {}; }
    void OnRunEnd(const runtime::ExecutionRunEnd&,const runtime::ExecutionRunCorrelation&) override {}
    void OnKernelBegin(const runtime::KernelSubmitInfo& kernel,const runtime::ExecutionRunCorrelation&) override {
        if (kernel.call_index == 0 && begin) begin();
    }
    runtime::ExecutionCompletionCallback OnKernelSubmitted(const runtime::KernelSubmitInfo&,
        const runtime::ExecutionRunCorrelation&) override { ++submitted; return {}; }
    void OnAllocation(const runtime::AllocationInfo&,const runtime::ExecutionRunCorrelation&) override {}
    void OnCopy(const runtime::CopyInfo&,const runtime::ExecutionRunCorrelation&) override {}
    runtime::ExecutionCompletionCallback OnCopySubmitted(const runtime::CopyInfo&,
        const runtime::ExecutionRunCorrelation&) override { return {}; }
};
class FailingLauncher final : public codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }
    AsyncOperation Launch(const Array<NDArray>&,const DeviceStream&,const ObjectRef&) const override {
        throw std::runtime_error("injected second-kernel failure");
    }
};
api::CompiledModule Observe(const api::CompiledGraph& graph,const std::shared_ptr<Observer>& observer,
                           bool fail_second = false) {
    const auto* node = graph.module().As<api::CompiledModuleNode>();
    std::vector<ci::CompiledModuleEntry> entries;
    for (const auto& entry : node->entries_) {
        auto copy = entry.second;
        if (fail_second && entry.first == std::string(graph.plan().calls()[1]->symbol)) {
            copy.executable = codegen::CompiledKernel(copy.signature,copy.launch_metadata,std::make_shared<FailingLauncher>());
        }
        entries.push_back(std::move(copy));
    }
    return ci::BuildCompiledModule(node->target_,std::move(entries),node->constants_,nullptr,observer);
}
void TestConcurrency(const api::CompiledGraph& graph) {
    const auto observer = std::make_shared<Observer>();
    RuntimeSession session(Observe(graph,observer),StatePlan(graph).BindRequestBatching(2));
    const auto request = session.AdmitRequest();
    session.EnqueueRequest(request,{Tensor({1,1,4}),Tensor({1,2})});
    bool reentered = false;
    observer->begin = [&] {
        Rejects([&] { session.ReleaseRequest(request); },"already in progress"); reentered = true;
    };
    Check(session.RunNextBatch().size() == 1 && reentered,"observer reentry did not reject safely");
    std::promise<void> entered, release;
    auto ready = entered.get_future();
    auto proceed = release.get_future().share();
    observer->begin = [&] { entered.set_value(); proceed.wait(); };
    session.EnqueueRequest(request,{Tensor({1,1,4}),Tensor({1,2})});
    auto run = std::async(std::launch::async,[&] { return session.RunNextBatch(); });
    try {
        Check(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready,"worker did not enter runtime");
        Rejects([&] { session.ReleaseRequest(request); },"already in progress");
        Rejects([&] { (void)session.AdmitRequest(); },"already in progress");
        Rejects([&] { (void)session.RunNextBatch(); },"already in progress");
        Rejects([&] { (void)session.RequestExtent(request); },"already in progress");
    } catch (...) { release.set_value(); run.wait(); throw; }
    release.set_value();
    Check(run.get().size() == 1 && session.RequestExtent(request) == 2,"concurrent rejection changed committed run");
    observer->begin = {};
    Rejects([&] { (void)RuntimeSession(graph.module(),StatePlan(graph,3,1,2).BindRequestBatching(2)); },
            "append differs from the module output extent contract");
    // A real first kernel is submitted; the second launcher throws before
    // state copy, and all further operations on the session reject.
    RuntimeSession failed(Observe(graph,observer,true),StatePlan(graph).BindRequestBatching(2));
    const auto bad = failed.AdmitRequest();
    failed.EnqueueRequest(bad,{Tensor({1,1,4}),Tensor({1,2})});
    Rejects([&] { (void)failed.RunNextBatch(); },"injected second-kernel failure");
    Rejects([&] { (void)failed.RunNextBatch(); },"reconstructed");
    Rejects([&] { (void)failed.RequestExtent(bad); },"reconstructed");
    std::cout << "request ordering: observer reentry and concurrent operations rejected; post-submit failure poisons session\n";
}
void TestAxisAndLifetime(const api::CompileConfig& config) {
    auto graph = Compile(config,2,2);
    const auto plan = StatePlan(graph,3,2,2).BindRequestBatching(2);
    RuntimeSession session(graph.module(),plan);
    const auto a = session.AdmitRequest({Tensor({1,2,1,4},10)},1);
    const auto b = session.AdmitRequest({Tensor({1,2,1,4},20)},1);
    const auto state_id = plan.state_value_ids()[0];
    const auto token = Tensor({1,2,2,4},-4);
    const auto expected = RuntimeSession(graph.module(),graph.plan()).Run({Tensor({1,2,1,4},10),token,Tensor({1,2})});
    session.EnqueueRequest(a,{token,Tensor({1,2})});
    session.EnqueueRequest(b,{token,Tensor({1,2})});
    graph = {};
    ci::ClearPrimitiveCacheForTesting();
    const auto outputs = Batch(session);
    Check(outputs.size() == 2 && Read(outputs[0].outputs[0]) == Read(expected[0]) &&
          Read(session.CopyRequestState(a,state_id)) == Read(expected[1]) && session.RequestExtent(b) == 3,
          "axis 2 multi-token gather/scatter or executable lifetime failed");
    std::cout << "axis 2 two-token append: real " << (cuda ? "CUDA" : "LLVM")
              << " batched outputs and KV bitwise match independent execution after graph/cache release\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        cuda = argc == 2 && std::string(argv[1]) == "--cuda";
        Check(argc == 1 || cuda,"usage: request_batching_test [--cuda]");
        if (cuda && !CollectDeviceAttributes(TestDevice()).exists) {
            std::cout << "[SKIP] no CUDA device\n"; return 77;
        }
        if (cuda) streams = {DeviceStream::Create(TestDevice()),DeviceStream::Create(TestDevice())};
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true; options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path()/"out"/
            (cuda ? "request_batching_cuda_profile" : "request_batching_profile")).string();
        options.enable_cupti = cuda;
        options = profiling::ApplyEnvironmentOverrides(options);
        const auto profile = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(profile,"request_batching");
        const auto config = api::CompileConfig::Create(BuildTarget(TestDevice()),cuda ? 3 : 2,options);
        const auto graph = Compile(config);
        if (cuda) for (const auto& symbol : graph.module().symbols()) {
            Check(graph.module().launch_metadata(symbol)->backend == codegen::CodeGenBackend::kCUDA,
                  "request kernel did not compile for CUDA");
        }
        TestQueue(graph,profile); TestConcurrency(graph); TestAxisAndLifetime(config);
        profile->Flush();
        streams.clear();
        std::cout << "[PASS] " << (cuda ? "cuda" : "llvm")
                  << "_request_batching_queue_state_streams_lifetime_and_failure\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
