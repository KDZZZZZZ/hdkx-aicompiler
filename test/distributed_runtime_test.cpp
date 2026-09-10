// Process-local CPU placement, DRef/CCL ownership and real LLVM worker execution.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "kxc/distributed/executor.h"
#include "kxc/distributed/worker.h"
#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"
#include "runtime/internal/compiled_module_node.h"
#include "compiler/internal/primitive_cache.h"
#if KXC_USE_LLVM
#include <llvm/Support/JSON.h>
#include <llvm/Support/Error.h>
#endif

namespace {
using namespace kxc;
using namespace kxc::disco;
using runtime::NDArray;
void Check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void Rejects(F f, const std::string& part = "") {
    try { f(); }
    catch (const std::exception& error) {
        Check(std::string(error.what()).find(part) != std::string::npos,
              "unexpected error: " + std::string(error.what()) + "; wanted: " + part);
        return;
    }
    throw std::runtime_error("expected rejection: " + part);
}
NDArray Tensor(const std::vector<float>& data, Array<int64_t> shape = {}) {
    if (shape.empty()) shape = {static_cast<int64_t>(data.size())};
    auto result = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), Device::CPU());
    Check(result.NBytes() == data.size() * sizeof(float), "fixture shape mismatch");
    result.CopyFromBytes(data.data(), result.NBytes()); return result;
}
std::vector<float> Read(const NDArray& array) {
    std::vector<float> result(array.NBytes() / sizeof(float));
    array.CopyToBytes(result.data(), array.NBytes()); return result;
}
void Equal(const NDArray& array, const std::vector<float>& expected) {
    Check(Read(array) == expected, "CPU numeric mismatch");
}
DiscoPlacement Placement(int workers = 2, int groups = 1, bool reverse = false) {
    Array<VirtualDevice> devices;
    for (int i = 0; i < workers; ++i) devices.push_back(VirtualDevice(Device::CPU(), BuildTarget(Device::CPU()), "global", reverse ? workers - 1 - i : i));
    return BuildDiscoPlacement(devices, groups);
}
ExecutionPlan Plan(Array<ObjectRef> nodes, const std::vector<Array<int64_t>>& shapes,
                   Array<int> inputs, int output, DiscoPlacement placement = Placement(),
                   const std::vector<int>& homes = {}) {
    Map<int, VirtualDevice> devices; Map<int, Array<int64_t>> value_shapes;
    Map<int, std::string> dtypes;
    for (size_t i = 0; i < shapes.size(); ++i) {
        devices.Set(static_cast<int>(i), placement->workers[homes.empty() ? 0 : homes.at(i)]->virtual_device);
        value_shapes.Set(static_cast<int>(i), shapes[i]); dtypes.Set(static_cast<int>(i), "float32");
    }
    return ExecutionPlan(nodes, devices, inputs, {}, value_shapes, dtypes, static_cast<int>(shapes.size()),
                         PassContext::FromTarget(BuildTarget(Device::CPU())), placement, output);
}

class CountingCCL final : public CCLBackend {
public:
    int calls = 0;
    bool fail = false;
    std::shared_ptr<CCLBackend> real = CreateCpuCCLBackend();
    void Hit() { ++calls; if (fail) throw std::runtime_error("injected CCL failure"); }
    void Copy(const DiscoSession& s,const DRef& a,const DRef& b,int x,int y) override { Hit(); real->Copy(s,a,b,x,y); }
    void AllReduce(const DiscoSession& s,const DRef& a,const DRef& b,const std::string& kind,bool group) override { Hit(); real->AllReduce(s,a,b,kind,group); }
    void BroadcastFromWorker0(const DiscoSession& s,const DRef& a,const DRef& b,bool group) override { Hit(); real->BroadcastFromWorker0(s,a,b,group); }
    void ScatterFromWorker0(const DiscoSession& s,const DRef& a,const DRef& b,bool group) override { Hit(); real->ScatterFromWorker0(s,a,b,group); }
    void GatherToWorker0(const DiscoSession& s,const DRef& a,const DRef& b,bool group) override { Hit(); real->GatherToWorker0(s,a,b,group); }
    void SendToWorker(const DiscoSession& s,const DRef& a,const DRef& b,int worker) override { Hit(); real->SendToWorker(s,a,b,worker); }
    void RecvFromWorker(const DiscoSession& s,const DRef& a,const DRef& b,int worker) override { Hit(); real->RecvFromWorker(s,a,b,worker); }
    void SyncWorker(const DiscoSession& s,int worker) override { Hit(); real->SyncWorker(s,worker); }
};

void SessionOwnership() {
    Rejects([] { DiscoSession::ThreadedSession(0, 1); });
    Rejects([] { DiscoSession::ThreadedSession(4, 3); });
    Rejects([] { DiscoSession::ThreadedSession(1, 0); });
    auto session = DiscoSession::ThreadedSession(2, 1), foreign = DiscoSession::ThreadedSession(2, 1);
    auto ref = session.NewDRef(); Check(!session.Get(0, ref).defined(), "new registers must be uninitialized");
    Rejects([&] { foreign.Get(0, ref); }, "owned by this session");
    Rejects([&] { foreign.Set(0, ref, Tensor({1})); }, "owned by this session");
    Rejects([&] { DRef forged(200, session); });
    Rejects([&] { DRef wrong(0, BuildTarget(Device::CPU())); });
    const int id = ref.reg_id(); auto duplicate = DRef(id, session);
    session.Set(0, ref, Tensor({3})); ref = DRef(); Equal(session.Get(0, duplicate), {3});
    duplicate = DRef(); Rejects([&] { session->GetRegister(0, id); });
    Rejects([&] { DRef stale(id, session); });
    // An external deleter proves that the final DRef clears the actual tensor storage.
    struct RetireProbe {
        DiscoSession session;
        int freed{0};
        int active_callbacks{0};
        int closed_callbacks{0};
    } probe{session};
    const auto tracked_array = [&] {
        auto node = new runtime::NDArrayNode();
        node->storage = Storage::FromExternal(Device::CPU(), new float[1], sizeof(float),
            [](void* data, void* context) {
                delete[] static_cast<float*>(data);
                auto& p = *static_cast<RetireProbe*>(context); ++p.freed;
                try { p.session.SyncWorker(0); ++p.active_callbacks; }
                catch (const std::exception&) { ++p.closed_callbacks; }
            }, &probe);
        return NDArray(ObjectRef(node));
    };
    {
        auto local = session.NewDRef();
        session.Set(0, local, tracked_array());
        session.Set(0, local, tracked_array()); // Replacing a slot also retires outside the lock.
    }
    Check(probe.freed == 2 && probe.active_callbacks == 2, "replacement/final DRef did not release storage with safe callback reentry");
    auto retired_at_shutdown = session.NewDRef(); session.Set(0, retired_at_shutdown, tracked_array());
    auto live = session.NewDRef(); session.Set(0, live, Tensor({7})); auto held = session.Get(0, live);
    session.Shutdown(); session.Shutdown(); Equal(held, {7});
    Check(probe.freed == 3 && probe.closed_callbacks == 1, "shutdown did not retire storage with safe callback reentry");
    Rejects([&] { session.Get(0, live); }, "shut down");
    Rejects([&] { session.Set(0, live, held); }, "shut down");
    Rejects([&] { session.NewDRef(); }, "shut down");
    Rejects([&] { session.SyncWorker(0); }, "shut down");
}

void Collectives() {
    auto session = DiscoSession::ThreadedSession(4, 2); auto ccl = CreateCpuCCLBackend();
    auto source = session.NewDRef(), dest = session.NewDRef();
    for (int worker = 0; worker < 4; ++worker) session.Set(worker, source, Tensor({float(worker + 1), float(worker + 1)}));
    ccl->AllReduce(session, source, dest, "sum", true);
    for (int worker = 0; worker < 4; ++worker) Equal(session.Get(worker, dest), {worker < 2 ? 3.f : 7.f, worker < 2 ? 3.f : 7.f});
    Check(session.Get(0, dest).storage().data() != session.Get(1, dest).storage().data(), "workers unexpectedly alias outputs");
    ccl->AllReduce(session, source, dest, "sum", false); Equal(session.Get(3, dest), {10, 10});
    ccl->BroadcastFromWorker0(session, source, dest, true); Equal(session.Get(1, dest), {1, 1}); Equal(session.Get(3, dest), {3, 3});
    session.Set(0, source, Tensor({1, 2, 3, 4})); session.Set(2, source, Tensor({5, 6, 7, 8}));
    ccl->ScatterFromWorker0(session, source, dest, true);
    Equal(session.Get(1, dest), {3, 4}); Equal(session.Get(3, dest), {7, 8});
    auto gathered = session.NewDRef(); ccl->GatherToWorker0(session, dest, gathered, true);
    Equal(session.Get(0, gathered), {1, 2, 3, 4}); Equal(session.Get(2, gathered), {5, 6, 7, 8});
    Check(!session.Get(1, gathered).defined(), "gather populated a non-root worker");
    auto transfer = session.NewDRef(); ccl->SendToWorker(session, gathered, transfer, 3); Equal(session.Get(3, transfer), {1, 2, 3, 4});
    ccl->RecvFromWorker(session, transfer, gathered, 3); Equal(session.Get(0, gathered), {1, 2, 3, 4});
    // Late-group validation must leave all destination sentinels intact.
    auto incomplete = session.NewDRef(); session.Set(0, incomplete, Tensor({1, 2}));
    for (int worker = 0; worker < 4; ++worker) session.Set(worker, dest, Tensor({99}));
    Rejects([&] { ccl->BroadcastFromWorker0(session, incomplete, dest, true); });
    Rejects([&] { ccl->ScatterFromWorker0(session, incomplete, dest, true); });
    Rejects([&] { ccl->AllReduce(session, incomplete, dest, "sum", true); });
    for (int worker = 0; worker < 4; ++worker) Equal(session.Get(worker, dest), {99});
    session.Set(2, incomplete, Tensor({3, 4, 5}));
    Rejects([&] { ccl->ScatterFromWorker0(session, incomplete, dest, true); }, "divisible");
    for (int worker = 0; worker < 4; ++worker) Equal(session.Get(worker, dest), {99});
    session.Set(1, incomplete, Tensor({1})); session.Set(3, incomplete, Tensor({1}));
    Rejects([&] { ccl->AllReduce(session, incomplete, dest, "sum", false); }, "equal shape");
    Rejects([&] { ccl->AllReduce(session, source, dest, "max", true); }, "only sum");
    session.Set(1, incomplete, NDArray::Zeros({2}, runtime::DataTypeFromString("int32"), Device::CPU()));
    Rejects([&] { ccl->GatherToWorker0(session, incomplete, dest, true); }, "dtype");
    session.Set(1, incomplete, Tensor({1}));
    Rejects([&] { ccl->GatherToWorker0(session, incomplete, dest, true); }, "shard shape and count");
    auto one = DiscoSession::ThreadedSession(1, 1); auto half = one.Empty({1}, "float16", false, false);
    Rejects([&] { ccl->AllReduce(one, half, one.NewDRef(), "sum", true); }, "supports float32");
    auto ints = DiscoSession::ThreadedSession(2, 1); auto input = ints.NewDRef(), output = ints.NewDRef();
    for (int worker = 0; worker < 2; ++worker) {
        auto array = NDArray::Empty({1}, runtime::DataTypeFromString("int32"), Device::CPU());
        int32_t value = worker ? 1 : std::numeric_limits<int32_t>::max(); array.CopyFromBytes(&value, sizeof(value)); ints.Set(worker, input, array);
    }
    ccl->AllReduce(ints, input, output, "sum", false);
    int32_t wrapped = 0; ints.Get(0, output).CopyToBytes(&wrapped, sizeof(wrapped)); Check(wrapped == std::numeric_limits<int32_t>::min(), "integer allreduce must wrap without signed C++ overflow");
    auto raw = Tensor({0, 0}); auto unaligned = raw.CreateView({1}, {}, 1);
    const float two = 2; unaligned.CopyFromBytes(&two, sizeof(two));
    ints.Set(0, input, Tensor({1})); ints.Set(1, input, unaligned);
    ccl->AllReduce(ints, input, output, "sum", false); Equal(ints.Get(0, output), {3});
    auto invalid = Tensor({1}); const_cast<runtime::NDArrayNode*>(invalid.operator->())->byte_offset = 4;
    ints.Set(1, input, invalid);
    Rejects([&] { ccl->AllReduce(ints, input, output, "sum", false); }, "range");
}

void PlansAndAdmission() {
    auto placement = Placement();
    Check(placement->workers[0]->target.CanonicalBytes() == BuildTarget(Device::CPU()).CanonicalBytes(), "Target canonical owner mismatch");
    Check(placement.FindWorker(VirtualDevice(Device::CPU(), BuildTarget(Device::CPU()), "global", 1)) == 1, "equivalent VirtualDevice lookup failed");
    Rejects([] { Placement(4, 3); });
    Rejects([] { BuildDiscoPlacement({VirtualDevice()}); });
    Rejects([&] { WorkerPlacement(0, 0, 0, Device::CPU(), Target(), placement->workers[0]->virtual_device); });
    CommExecAttrs attrs;
    auto make = [&](DiscoPlacement p) { return Plan({CommExec("device.copy", attrs, {0}, {1}, {1, 0}),
                                                  CommExec("device.copy", attrs, {1}, {2}, {0, 1}), BarrierExec("done", {1, 0})},
                                                 {{2}, {2}, {2}}, {0}, 2, p, {0, 1, 0}); };
    auto plan = make(placement); const auto json = SerializeExecutionPlanToJson(plan);
    Check(json == SerializeExecutionPlanToJson(make(Placement(2, 1, true))), "placement input order changed JSON");
    auto restored = DeserializeExecutionPlanFromJson(json);
    Check(json == SerializeExecutionPlanToJson(restored), "JSON round trip lost contract fields");
    auto legacy = json;
    legacy.replace(legacy.find("\"schema_version\":3"), 18, "\"schema_version\":2");
    const std::string output_field = "\"output_value_ids\":[2]";
    Check(legacy.find(output_field) != std::string::npos, "legacy fixture output missing");
    legacy.replace(legacy.find(output_field), output_field.size(), "\"output_value\":2");
    Check(SerializeExecutionPlanToJson(DeserializeExecutionPlanFromJson(legacy)) == json,
          "v2 single-output migration changed the execution contract");
    Check(restored->placement->workers[0]->target.CanonicalBytes() == BuildTarget(Device::CPU()).CanonicalBytes(), "JSON lost Target capabilities");
    const auto plan_file = std::filesystem::current_path() / "out" / "distributed_plan.json";
    std::filesystem::create_directories(plan_file.parent_path());
    SaveExecutionPlanToJsonFile(plan, plan_file.string());
    Rejects([&] { SaveExecutionPlanToJsonFile(ExecutionPlan(), plan_file.string()); });
    Check(SerializeExecutionPlanToJson(LoadExecutionPlanFromJsonFile(plan_file.string())) == json, "invalid save destroyed the existing plan");
    auto session = DiscoSession::ThreadedSession(2, 1); auto input = session.NewDRef(); session.Set(0, input, Tensor({4, 9}));
    auto ccl = std::make_shared<CountingCCL>(); ExecutionPlanExecutor executor(session, ccl);
    Map<int, DRef> initial; initial.Set(0, input);
    auto output = executor.ExecuteForOutput(restored, initial); Equal(session.Get(0, output), {4, 9});
    Check(initial.size() == 1 && !initial.count(1), "executor mutated caller's initial Map");
    auto prior = session.Get(0, output); session.Set(0, input, Tensor({8, 7}));
    auto next = executor.ExecuteForOutput(restored, initial); Equal(session.Get(0, next), {8, 7}); Equal(prior, {4, 9});
    Check(output.reg_id() != next.reg_id(), "runs reused stale value bindings");
    auto zero = [&](const ExecutionPlan& bad, const std::string& part) {
        ccl->calls = 0; Rejects([&] { executor.Execute(bad, initial); }, part); Check(ccl->calls == 0, "admission failure performed CCL work");
    };
    zero(Plan({CommExec("device.copy", attrs, {0}, {1}, {0}), KernelExec("diagnostic", {1}, {2}, {0}, "missing")}, {{2}, {2}, {2}}, {0}, 2), "bound CompiledModule");
    zero(Plan({CommExec("device.copy", attrs, {1}, {2}, {0}), CommExec("device.copy", attrs, {0}, {1}, {0})}, {{2}, {2}, {2}}, {0}, 2), "use before definition");
    zero(Plan({CommExec("device.copy", attrs, {0}, {1}, {0}), CommExec("device.copy", attrs, {1}, {1}, {0})}, {{2}, {2}}, {0}, 1), "duplicate definition");
    zero(Plan({ObjectRef()}, {{2}}, {0}, 0), "unknown node");
    zero(Plan({BuildTarget(Device::CPU())}, {{2}}, {0}, 0), "unknown node");
    zero(Plan({CommExec("device.unknown", attrs, {0}, {1}, {0})}, {{2}, {2}}, {0}, 1), "unknown communication");
    zero(Plan({CommExec("device.copy", attrs, {0}, {1}, {})}, {{2}, {2}}, {0}, 1), "explicit worker_set");
    zero(Plan({CommExec("device.copy", attrs, {0}, {1}, {0, 0})}, {{2}, {2}}, {0}, 1), "unique valid workers");
    attrs.async = true;
    auto async_plan = Plan({CommExec("device.copy", attrs, {0}, {1}, {0})}, {{2}, {2}}, {0}, 1);
    zero(DeserializeExecutionPlanFromJson(SerializeExecutionPlanToJson(async_plan)), "unsupported communication attrs");
    attrs.async = false; attrs.src_virtual_device = VirtualDevice(Device::CPU(), BuildTarget(Device::CPU()), "global", 99);
    zero(Plan({CommExec("device.copy", attrs, {0}, {1}, {0})}, {{2}, {2}}, {0}, 1), "unavailable on worker -1");
    auto replace = [&](std::string from, std::string to, std::string part) {
        auto invalid = json; const auto at = invalid.find(from); Check(at != std::string::npos, "JSON fixture token missing");
        invalid.replace(at, from.size(), to); Rejects([&] { DeserializeExecutionPlanFromJson(invalid); }, part);
    };
    replace("\"schema_version\":3", "\"schema_version\":1", "schema_version");
    replace("\"kind\":\"comm\"", "\"kind\":\"future\"", "node kind");
    replace("\"nodes\":[", "\"nodes\":[null,", "root.nodes[0]");
    replace("\"reserved\":{}", "\"reserved\":{\"new_semantics\":1}", "unknown field");
    replace("\"async\":false", "\"async\":false,\"new_semantics\":1", "unknown field");
    replace("\"value_id\":1,\"shape\"", "\"value_id\":0,\"shape\"", "duplicate value_info");
    replace("\"value_id\":1,\"virtual_device\"", "\"value_id\":0,\"virtual_device\"", "duplicate value_virtual_devices");
    auto unicode = json;
    const auto tag_at = unicode.find("\"tag\":\"done\"");
    unicode.replace(tag_at, std::string("\"tag\":\"done\"").size(), "\"tag\":\"\\u4e2d\\u6587\\ud83d\\ude00\"");
    const auto decoded = DeserializeExecutionPlanFromJson(unicode);
    Check(decoded->nodes[2].As<BarrierExecNode>()->tag == "中文😀", "JSON escaped Unicode lost identity");
    replace("\"tag\":\"done\"", "\"tag\":\"\\ud800\"", "surrogate");
    replace("\"tag\":\"done\"", "\"tag\":\"bad\ntext\"", "unescaped control");
    ccl->fail = true; Rejects([&] { executor.Execute(plan, initial); }, "node[0]: ExecutionPlan communication device.copy workers 0 -> 1: injected CCL failure");
}

void GroupPlans() {
    const auto placement = Placement(4, 2);
    auto session = DiscoSession::ThreadedSession(4, 2); auto ccl = std::make_shared<CountingCCL>();
    ExecutionPlanExecutor executor(session, ccl);
    auto source = session.NewDRef();
    for (int worker = 0; worker < 4; ++worker) session.Set(worker, source, Tensor({float(worker + 1)}));
    Map<int, DRef> initial; initial.Set(0, source); CommExecAttrs attrs;
    auto reduction = Plan({CommExec("device.allreduce", attrs, {0}, {1}, {3, 2, 1, 0})}, {{1}, {1}}, {0}, 1, placement);
    auto result = executor.ExecuteForOutput(reduction, initial);
    Equal(session.Get(0, result), {3}); Equal(session.Get(3, result), {7});
    auto transfer = [&] { return Plan({CommExec("device.copy", attrs, {0}, {1}, {0, 3})}, {{1}, {1}}, {0}, 1, placement, {0, 3}); };
    ccl->calls = 0; Rejects([&] { executor.Execute(transfer(), initial); }, "crosses groups"); Check(ccl->calls == 0, "cross-group admission performed communication");
    attrs.in_group = false; Equal(session.Get(3, executor.ExecuteForOutput(transfer(), initial)), {1});
    ccl->calls = 0;
    auto partial = Plan({CommExec("device.allreduce", attrs, {0}, {1}, {0, 1})}, {{1}, {1}}, {0}, 1, placement);
    Rejects([&] { executor.Execute(partial, initial); }, "every session worker"); Check(ccl->calls == 0, "partial collective performed communication");
    attrs.group_id = 1;
    auto selected = Plan({CommExec("device.allreduce", attrs, {0}, {1}, {0, 1, 2, 3})}, {{1}, {1}}, {0}, 1, placement);
    Rejects([&] { executor.Execute(selected, initial); }, "unsupported communication attrs");
    auto other = DiscoSession::ThreadedSession(4, 1);
    Rejects([&] { ExecutionPlanExecutor(other).Execute(reduction, initial); }, "group count mismatch");
}

#if KXC_USE_LLVM
struct LaunchProof {
    std::atomic<int> calls{0}; std::atomic<int> fail_worker{-1};
    std::mutex mutex; std::condition_variable wake;
    std::set<int> workers; std::set<std::thread::id> threads;
    int arriving = 0;
};
class ObservedLLVM final : public codegen::KernelLauncher {
public:
    ObservedLLVM(codegen::CompiledKernel real, std::shared_ptr<LaunchProof> proof) : real_(std::move(real)), proof_(std::move(proof)) {}
    bool IsReady() const noexcept override { return real_.IsReady(); }
    AsyncOperation Launch(const Array<NDArray>& arguments, const DeviceStream& stream, const ObjectRef&) const override {
        const int worker = CurrentWorkerId(); ++proof_->calls;
        {
            std::unique_lock<std::mutex> lock(proof_->mutex);
            proof_->workers.insert(worker); proof_->threads.insert(std::this_thread::get_id());
            // The first pair must enter together. A serial worker loop fails this bounded rendezvous.
            if (proof_->arriving < 2) {
                ++proof_->arriving; proof_->wake.notify_all();
                Check(proof_->wake.wait_for(lock, std::chrono::seconds(5), [&] { return proof_->arriving == 2; }), "worker launches did not overlap");
            }
        }
        if (worker == proof_->fail_worker.load()) throw std::runtime_error("injected worker failure");
        return real_.Launch(arguments, stream);
    }
private:
    codegen::CompiledKernel real_; std::shared_ptr<LaunchProof> proof_;
};
api::CompiledModule Observe(const api::CompiledModule& module, const std::shared_ptr<LaunchProof>& proof) {
    std::vector<api::internal::CompiledModuleEntry> entries;
    const auto* node = module.As<api::CompiledModuleNode>();
    for (const auto& item : node->entries_) {
        auto entry = item.second;
        entry.executable = codegen::CompiledKernel(entry.signature, entry.launch_metadata,
            std::make_shared<ObservedLLVM>(entry.executable, proof));
        entries.push_back(std::move(entry));
    }
    return api::internal::BuildCompiledModule(node->target_, std::move(entries),
        api::internal::BorrowCompiledModuleConstants(module));
}
bool SameStats(const api::internal::PrimitiveCacheStats& a, const api::internal::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries && a.accounted_bytes == b.accounted_bytes &&
        a.evictions == b.evictions && a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters && a.failures == b.failures &&
        a.rejections == b.rejections && a.active_pins == b.active_pins;
}
std::string Text(const llvm::json::Object& object, const char* key) {
    auto value = object.getString(key); Check(bool(value), std::string("missing profile field ") + key); return value->str();
}
void ProfileEvidence(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush(); std::ifstream input(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(input.good(), "no distributed profile bundle");
    int kernels = 0, communication = 0, runs = 0, failed_runs = 0, failed_workers = 0, rejected_runs = 0;
    std::set<int64_t> workers;
    std::set<std::string> parents, kernel_parents;
    std::string line;
    while (std::getline(input, line)) {
        auto value = llvm::json::parse(line); if (!value) throw std::runtime_error(llvm::toString(value.takeError()));
        const auto* event = value->getAsObject(); Check(event, "profile event is not an object");
        if (Text(*event, "component") != "execution_plan") continue;
        const auto run = Text(*event, "run_id"), type = Text(*event, "event_type");
        if (run == "distributed-admission-error") {
            Check(type == "execute_plan" && Text(*event, "status") == "error" && Text(*event, "message").find("node[2]") != std::string::npos,
                  "admission error must emit only a failed plan span with node context");
            ++rejected_runs; continue;
        }
        if (run == "distributed-worker-error") {
            if (type == "execute_plan") {
                Check(Text(*event, "status") == "error" && Text(*event, "message").find("worker 1 kernel") != std::string::npos, "worker error lost plan context");
                ++failed_runs;
            }
            if (type == "kernel_exec" && Text(*event, "status") == "error") {
                Check(event->getInteger("worker_id") == 1 && Text(*event, "phase") == "complete", "worker error lost completion/worker fields");
                ++failed_workers;
            }
            continue;
        }
        if (run != "distributed-success") continue;
        Check(Text(*event, "phase") == "complete" && Text(*event, "status") == "ok", "incomplete distributed profile event");
        const auto duration = event->getInteger("duration_ns"); Check(duration && *duration >= 0, "invalid distributed duration");
        auto* fields = event->getObject("fields"); Check(fields && Text(*fields, "contract_version") == "3", "missing distributed contract version");
        if (type == "execute_kernel_node") parents.insert(Text(*event, "span_id"));
        if (type == "kernel_exec") {
            ++kernels; auto worker = event->getInteger("worker_id"); Check(bool(worker), "missing worker id"); workers.insert(*worker);
            Check(!Text(*event, "kernel_symbol").empty(), "missing actual kernel symbol");
            Check(Text(*fields, "node_index") == "1" || Text(*fields, "node_index") == "2", "wrong kernel node association");
            kernel_parents.insert(Text(*event, "parent_span_id"));
        }
        if (type == "comm_exec") ++communication;
        if (type == "execute_plan") ++runs;
    }
    Check(kernels == 4 && communication == 3 && runs == 1 && workers == std::set<int64_t>{0, 1} && parents == kernel_parents,
          "profile must contain all completed worker calls, communication and parent links");
    Check(failed_runs == 1 && failed_workers == 1 && rejected_runs == 1, "profile lost worker/admission failures");
}

void RealLLVM() {
    auto proof = std::make_shared<LaunchProof>();
    auto session = DiscoSession::ThreadedSession(2, 1); auto ccl = std::make_shared<CountingCCL>();
    std::unique_ptr<ExecutionPlanExecutor> executor; ExecutionPlan plan;
    std::vector<float> expected;
    {
        Var x("x", TensorType({2}, "float32")); Constant bias(Tensor({1, -2}));
        Function function({x}, Call(relay::Op::Get("nn_relu"), {Call(relay::Op::Get("add"), {x, bias})}));
        auto graph = api::Compiler::Compile(function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 0));
        Check(graph.plan().calls().size() == 2 && graph.module().constants().size() == 1, "fixture must contain two real kernels and a module constant");
        runtime::RuntimeSession baseline(graph.module(), graph.plan());
        auto a = Read(baseline.Run({Tensor({-3, 4})})[0]); auto b = Read(baseline.Run({Tensor({5, -6})})[0]);
        for (size_t i = 0; i < a.size(); ++i) a[i] += b[i]; expected = a; expected.insert(expected.end(), a.begin(), a.end());
        auto module = Observe(graph.module(), proof);
        auto snapshot = module.constants(); snapshot.begin()->second.CopyFrom(Tensor({99, 99}));
        CommExecAttrs attrs; Array<ObjectRef> nodes{CommExec("device.scatter_from_worker0", attrs, {0}, {1}, {0, 1})};
        for (size_t i = 0; i < graph.plan().calls().size(); ++i) {
            const auto symbol = graph.plan().calls()[i]->symbol;
            nodes.push_back(KernelExec("diagnostic_only", {static_cast<int>(i + 1)}, {static_cast<int>(i + 2)}, {0, 1}, symbol, module.signature(symbol).CanonicalBytes()));
        }
        nodes.push_back(CommExec("device.allreduce", attrs, {3}, {4}, {0, 1}));
        nodes.push_back(CommExec("device.gather_to_worker0", attrs, {4}, {5}, {0, 1}));
        plan = Plan(nodes, {{4}, {2}, {2}, {2}, {2}, {4}}, {0}, 5);
        const auto json = SerializeExecutionPlanToJson(plan); plan = DeserializeExecutionPlanFromJson(json);
        Check(json == SerializeExecutionPlanToJson(plan), "kernel ABI JSON round trip failed");
        executor = std::make_unique<ExecutionPlanExecutor>(session, module, ccl);
    } // Graph, caller module and baseline session die; executor must retain launch ownership.
    auto input = session.NewDRef(); session.Set(0, input, Tensor({-3, 4, 5, -6})); Map<int, DRef> initial; initial.Set(0, input);
    profiling::ProfileOptions options; options.enabled = true; options.record_pass_ir = false;
    options.bundle_dir = (std::filesystem::current_path() / "out" / "distributed_runtime").string();
    auto context = profiling::ProfileContext::Create(options);
    api::internal::ClearPrimitiveCacheForTesting();
    const auto before = api::internal::GetPrimitiveCacheStats();
    Check(before.entries == 0, "lifetime fixture requires artifacts evicted before execution");
    {
        profiling::ActivationScope activation(context, "distributed-success");
        auto result = executor->Execute(plan, initial); Equal(session.Get(0, result.at(5)), expected);
        Check(!session.Get(1, result.at(5)).defined(), "gather wrote a non-root output");
        Check(session.Get(0, result.at(3)).storage().data() != session.Get(1, result.at(3)).storage().data(), "worker kernels aliased outputs");
    }
    Check(SameStats(before, api::internal::GetPrimitiveCacheStats()), "distributed execution touched primitive cache");
    Check(proof->calls == 4 && proof->workers == std::set<int>{0, 1} && proof->threads.size() >= 2 && !proof->threads.count(std::this_thread::get_id()), "LLVM did not execute on the two worker threads");
    Check(ccl->calls == 3 && initial.size() == 1, "unexpected CCL calls or caller Map mutation");
    Array<ObjectRef> bad_nodes; for (const auto& node : plan->nodes) bad_nodes.push_back(node);
    const auto* original = plan->nodes[2].As<KernelExecNode>();
    bad_nodes[2] = KernelExec("diagnostic_only", {2}, {3}, {0, 1}, original->kernel_symbol, "incorrect-ABI");
    auto bad = Plan(bad_nodes, {{4}, {2}, {2}, {2}, {2}, {4}}, {0}, 5);
    ccl->calls = 0;
    { profiling::ActivationScope activation(context, "distributed-admission-error");
      Rejects([&] { executor->Execute(bad, initial); }, "node[2]: ExecutionPlan kernel ABI mismatch"); }
    Check(proof->calls == 4 && ccl->calls == 0, "late ABI mismatch performed early communication/launch");
    bad_nodes[2] = KernelExec("diagnostic_only", {2, 2}, {3}, {0, 1}, original->kernel_symbol, original->kernel_abi);
    auto arity = Plan(bad_nodes, {{4}, {2}, {2}, {2}, {2}, {4}}, {0}, 5);
    Rejects([&] { executor->Execute(arity, initial); }, "value arity mismatch");
    bad_nodes[2] = plan->nodes[2];
    auto wrong_shape = Plan(bad_nodes, {{4}, {2}, {2}, {3}, {2}, {4}}, {0}, 5);
    Rejects([&] { executor->Execute(wrong_shape, initial); }, "node[2]: ExecutionPlan kernel value contract mismatch");
    auto target_node = new TargetNode(*plan->placement->workers[0]->target.operator->()); target_node->attrs.arch += "-incompatible";
    Target incompatible{ObjectRef(target_node)};
    auto mismatched_placement = BuildDiscoPlacement({VirtualDevice(Device::CPU(), incompatible, "global", 0), VirtualDevice(Device::CPU(), incompatible, "global", 1)});
    auto wrong_target = Plan(bad_nodes, {{4}, {2}, {2}, {2}, {2}, {4}}, {0}, 5, mismatched_placement);
    Rejects([&] { executor->Execute(wrong_target, initial); }, "module Target mismatch on worker 0");
    Check(proof->calls == 4 && ccl->calls == 0, "arity/shape/Target admission performed early work");
    proof->fail_worker = 1;
    { profiling::ActivationScope activation(context, "distributed-worker-error");
      Rejects([&] { executor->Execute(plan, initial); }, "node[1]: ExecutionPlan worker 1 kernel"); }
    Check(ccl->calls == 1 && proof->calls == 6, "worker failure did not stop before later nodes");
    proof->fail_worker = -1;
    auto output = executor->ExecuteForOutput(plan, initial); Equal(session.Get(0, output), expected);
    Check(SameStats(before, api::internal::GetPrimitiveCacheStats()), "failure or repeated execution touched primitive cache");
    ProfileEvidence(context);
    std::cout << "[PASS] 4 real LLVM worker calls, 3 collectives, RuntimeSession equality, retained module, zero-work late ABI rejection\n";
}

void WholeGraphBinding() {
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 0);
    Var x("x", TensorType({2}, "float32")), y("y", TensorType({2}, "float32"));
    Constant bias(Tensor({1, -2}));
    Call shared(relay::Op::Get("add"), {x, bias});
    Call positive(relay::Op::Get("nn_relu"), {shared});
    Call residual(relay::Op::Get("add"), {shared, y});
    auto graph = api::Compiler::Compile(Function({x, y}, Tuple({positive, residual, shared})), config);
    Check(graph.plan().calls().size() == 3, "whole-graph fixture lost its shared producer");
    auto proof = std::make_shared<LaunchProof>(); proof->arriving = 2;
    auto module = Observe(graph.module(), proof);
    const auto placement = Placement();
    const auto before = api::internal::GetPrimitiveCacheStats();
    const auto plan = ExecutionPlan::FromExecutablePlan(module, graph.plan(), placement, {0, 1}, {0, 1, 1}, 0);
    Check(plan->input_value_ids.size() == 2 && plan->constant_value_ids.empty() &&
          plan->output_value_ids.size() == 3 && plan->nodes.size() == 6,
          "binding must reuse one shared transfer and leave constants module-owned");
    const auto json = SerializeExecutionPlanToJson(plan);
    const auto restored = DeserializeExecutionPlanFromJson(json);
    Check(json == SerializeExecutionPlanToJson(restored), "multi-output binding JSON changed order");
    Rejects([&] { ExecutionPlan::FromExecutablePlan(module, graph.plan(), placement, {0}, {0, 1, 1}, 0); });
    Rejects([&] { ExecutionPlan::FromExecutablePlan(module, graph.plan(), placement, {0, 1}, {0, 1, 9}, 0); });
    Rejects([&] { ExecutionPlan::FromExecutablePlan(module, graph.plan(), placement, {0, 1}, {0, 1, 1}, -1); });
    auto altered = new TargetNode(*BuildTarget(Device::CPU()).operator->()); altered->attrs.arch += "-wrong";
    const auto wrong = BuildDiscoPlacement({VirtualDevice(Device::CPU(), Target(ObjectRef(altered)), "global", 0)});
    Rejects([&] { ExecutionPlan::FromExecutablePlan(module, graph.plan(), wrong, {0, 0}, {0, 0, 0}, 0); }, "Targets");
    Check(proof->calls == 0 && SameStats(before, api::internal::GetPrimitiveCacheStats()), "binding performed compilation or execution");
    {
        Var past("past", TensorType({1, 4, 2}, "float32")), next("next", TensorType({1, 1, 2}, "float32"));
        Call present(relay::Op::Get("concatenate"), {past, next}, relay::ConcatenateAttrs::Create(1));
        const auto capacity = api::Compiler::Compile(Function({past, next},
            Tuple({Call(relay::Op::Get("nn_relu"), {next}), present})), config);
        const auto state = capacity.BindStateOutputs({{capacity.plan().input_value_ids()[0], capacity.plan().output_value_ids()[1], 1, 4, 1}});
        Rejects([&] { ExecutionPlan::FromExecutablePlan(state.module(), state.plan(), placement, {0}, {0, 0}, 0); }, "state-free");
        const auto constant_output = api::Compiler::Compile(Function({x}, Tuple({shared, bias})), config);
        Rejects([&] { ExecutionPlan::FromExecutablePlan(constant_output.module(), constant_output.plan(), placement, {0}, {0}, 0); }, "constants as graph outputs");
        Check(proof->calls == 0, "unsupported state/constant-output binding launched kernels");
    }
    const Array<NDArray> inputs{Tensor({-3, 4}), Tensor({3, 5})};
    Array<NDArray> expected;
    { runtime::RuntimeSession reference(graph.module(), graph.plan()); expected = reference.Run(inputs); }
    Equal(expected[0], {0, 2}); Equal(expected[1], {1, 7}); Equal(expected[2], {-2, 2});
    auto session = DiscoSession::ThreadedSession(2);
    auto ccl = std::make_shared<CountingCCL>();
    ExecutionPlanExecutor executor(session, module, ccl);
    Map<int, DRef> initial;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto value = session.NewDRef(); session.Set(static_cast<int>(i), value, inputs[i]);
        initial.Set(plan->input_value_ids[i], value);
    }
    auto bad_node = new ExecutionPlanNode(*plan.operator->());
    Map<int, VirtualDevice> bad_homes;
    for (const auto& entry : plan->value_virtual_devices) {
        bad_homes.Set(entry.first, entry.first == plan->output_value_ids[1]
            ? placement->workers[1]->virtual_device : entry.second);
    }
    bad_node->value_virtual_devices = bad_homes;
    Rejects([&] { executor.ExecuteForOutputs(ExecutionPlan(ObjectRef(bad_node)), initial); }, "unavailable on worker 1");
    Check(ccl->calls == 0 && proof->calls == 0, "unavailable secondary output performed early copy/kernel work");
    auto empty_node = new ExecutionPlanNode(*plan.operator->()); empty_node->output_value_ids = {};
    Rejects([&] { SerializeExecutionPlanToJson(ExecutionPlan(ObjectRef(empty_node))); }, "ordered outputs");
    graph = api::CompiledGraph();
    module = api::CompiledModule(ObjectRef());
    api::internal::ClearPrimitiveCacheForTesting();
    const auto evicted = api::internal::GetPrimitiveCacheStats();
    const auto outputs = executor.ExecuteForOutputs(restored, initial);
    Check(ccl->calls == 3 && proof->calls == 3 && proof->workers == std::set<int>{0, 1}, "whole graph did not use both workers and exactly three copies");
    for (size_t i = 0; i < outputs.size(); ++i) Equal(session.Get(0, outputs[i]), Read(expected[i]));
    ccl->fail = true;
    Rejects([&] { executor.ExecuteForOutputs(restored, initial); }, "injected CCL failure");
    Check(proof->calls == 4, "copy failure did not stop after the first real producer");
    ccl->fail = false;
    const auto again = executor.ExecuteForOutputs(restored, initial);
    for (size_t i = 0; i < again.size(); ++i) {
        Equal(session.Get(0, again[i]), Read(expected[i])); Equal(session.Get(0, outputs[i]), Read(expected[i]));
        Check(session.Get(0, outputs[i]).storage().get() != session.Get(0, again[i]).storage().get(), "runs shared output storage");
    }
    Check(SameStats(evicted, api::internal::GetPrimitiveCacheStats()), "bound graph execution used the primitive cache");
    std::cout << "[PASS] compiled shared DAG -> 2 workers -> 3 ordered outputs; 3 kernels/3 copies; constants stay in module; "
                 "v3 round trip, secondary-output preflight, eviction lifetime and copy-failure retry\n";
}

template<class T> std::vector<T> Binary(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Check(file && file.tellg() == static_cast<std::streamoff>(count * sizeof(T)), "bad fixture: " + path);
    file.seekg(0); std::vector<T> values(count);
    Check(bool(file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T))), "cannot read fixture: " + path);
    return values;
}

void FullMiniMind(const std::string& root) {
    int batch = 0, heads = 0, dim = 0, layers = 0, extent = 0;
    std::ifstream layout(root + "/layout.txt"), seed(root + "/seed_extent.txt");
    Check(bool(layout >> batch >> heads >> dim >> layers) && bool(seed >> extent) &&
          batch == 1 && heads == 4 && dim == 96 && layers == 8 && extent == 16, "expected locked B1/S16 MiniMind fixture");
    std::string receipt; std::ifstream receipt_file(root + "/prefill_export_receipt.txt");
    Check(bool(std::getline(receipt_file, receipt)) && receipt.rfind("sha256:minimind_prefill_static.onnx:", 0) == 0, "missing export receipt");
    const auto imported = frontend::LoadONNXImportSpec(root + "/prefill.json", root + "/prefill.params");
    Check(imported.input_names == std::vector<std::string>{"input_ids"} && imported.output_names.size() == 17, "model I/O contract drifted");
    auto function = relay::InferTypePass(imported.function);
    function = relay::RunRelayPassPipeline(function, {"fold_constant", "simplify_expr"});
    function = relay::InferTypePass(function);
    const auto graph = api::Compiler::Compile(function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 0));
    Check(graph.plan().calls().size() == 650 && graph.plan().output_value_ids().size() == 17, "full model must retain 650 calls and 17 outputs");
    auto input = NDArray::Empty({1, 16}, runtime::DataTypeFromString("int64"), Device::CPU());
    const auto tokens = Binary<int64_t>(root + "/prefill_input_ids.bin", 16);
    input.CopyFromBytes(tokens.data(), input.NBytes());
    Array<NDArray> expected;
    { runtime::RuntimeSession reference(graph.module(), graph.plan()); expected = reference.Run({input}); }
    auto proof = std::make_shared<LaunchProof>(); proof->arriving = 2;
    const auto module = Observe(graph.module(), proof);
    const auto placement = Placement();
    auto session = DiscoSession::ThreadedSession(2);
    auto ccl = std::make_shared<CountingCCL>();
    ExecutionPlanExecutor executor(session, module, ccl);
    profiling::ProfileOptions options; options.enabled = true; options.record_pass_ir = false;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.bundle_dir = (std::filesystem::current_path() / "out/distributed_runtime/minimind").string();
    const auto context = profiling::ProfileContext::Create(options);
    const auto before = api::internal::GetPrimitiveCacheStats();
    std::filesystem::create_directories(context->bundle_dir());
    std::string previous_json; Array<DRef> retained;
    size_t compared = 0; double worst = 0;
    for (int run = 0; run < 2; ++run) {
        Array<int> workers;
        for (size_t i = 0; i < graph.plan().calls().size(); ++i) workers.push_back((i < 325 ? 0 : 1) ^ run);
        const auto plan = ExecutionPlan::FromExecutablePlan(module, graph.plan(), placement, {0}, workers, run);
        const auto json = SerializeExecutionPlanToJson(plan);
        Check(json != previous_json && plan->constant_value_ids.empty(), "placement did not change plan or copied the constant pool");
        previous_json = json;
        const auto restored = DeserializeExecutionPlanFromJson(json);
        Check(SerializeExecutionPlanToJson(restored) == json, "full graph JSON round trip changed the plan");
        SaveExecutionPlanToJsonFile(restored, context->bundle_dir() + "/plan-" + std::to_string(run) + ".json");
        int copies = 0, kernels = 0;
        for (const auto& node : plan->nodes) { copies += node.As<CommExecNode>() != nullptr; kernels += node.As<KernelExecNode>() != nullptr; }
        Check(kernels == 650 && copies > 0, "full distributed graph is missing kernels/transfers");
        Map<int, DRef> initial;
        auto value = session.NewDRef(); session.Set(0, value, input); initial.Set(plan->input_value_ids[0], value);
        const auto calls_before = proof->calls.load(); const int copies_before = ccl->calls;
        Array<DRef> outputs;
        {
            profiling::ActivationScope activation(context, "minimind-distributed-" + std::to_string(run));
            profiling::EventSpec spec; spec.component = "model"; spec.event_type = "distributed_model_run";
            spec.fields = {{"model", "minimind"}, {"stage", "prefill"}, {"export_receipt", receipt},
                {"batch", "1"}, {"sequence_length", "16"}, {"placement", std::to_string(run)}};
            profiling::ScopedSpan span(context, std::move(spec));
            outputs = executor.ExecuteForOutputs(restored, initial);
            span.AddMetric("kernel_count", kernels); span.AddMetric("copy_count", copies); span.AddMetric("output_count", outputs.size());
        }
        Check(outputs.size() == 17 && proof->calls == calls_before + 650 && ccl->calls == copies_before + copies,
              "full model lost actual calls/copies/outputs");
        for (size_t index = 0; index < outputs.size(); ++index) {
            const auto actual = session.Get(run, outputs[index]);
            const auto values = Read(actual);
            Check(values == Read(expected[index]), "distributed model differs from independent RuntimeSession");
            const auto name = index == 0 ? "/prefill_reference_logits.bin" :
                "/seed_present_" + std::string(index % 2 ? "k_" : "v_") + std::to_string((index - 1) / 2) + ".bin";
            const auto reference = Binary<float>(root + name, index == 0 ? 102400 : 6144);
            Check(values.size() == reference.size(), "model output order/shape mismatch");
            for (size_t i = 0; i < values.size(); ++i) {
                Check(std::isfinite(values[i]) && std::isfinite(reference[i]), "nonfinite model output");
                const double error = std::abs(double(values[i]) - reference[i]); worst = std::max(worst, error);
                Check(error <= 5e-5, "distributed model exceeds ONNX tolerance"); ++compared;
            }
            if (run) {
                Equal(session.Get(0, retained[index]), Read(expected[index]));
                Check(session.Get(0, retained[index]).storage().get() != actual.storage().get(), "different placements reused old output storage");
            }
        }
        if (!run) retained = outputs;
        Check(SameStats(before, api::internal::GetPrimitiveCacheStats()), "distributed binding/execution compiled or looked up cache entries");
        std::cout << "[PASS] MiniMind distributed placement=" << run << " kernels=" << kernels << " copies=" << copies << " outputs=17\n";
    }
    context->Flush();
    Check(proof->workers == std::set<int>{0, 1} && proof->calls == 1300 && compared == 401408, "model coverage count mismatch");
    std::cout << "[PASS] full MiniMind distributed prefill: 2 placements, 2 workers, 1300 LLVM calls; reference_values="
              << compared << " independent_values=" << compared << " max_abs_error=" << worst
              << "; 16 KV outputs; no runtime compilation; receipt=" << receipt << '\n';
}
#endif
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            Check(argc == 2 && std::string(argv[1]) == "--model", "usage: distributed_runtime_test [--model]");
#if KXC_USE_LLVM
            const char* root = std::getenv("KXC_MINIMIND_DISTRIBUTED_DIR");
            if (!root || !*root) { std::cout << "[SKIP] KXC_MINIMIND_DISTRIBUTED_DIR is unset\n"; return 77; }
            FullMiniMind(root);
            return 0;
#else
            return 77;
#endif
        }
        SessionOwnership(); Collectives(); PlansAndAdmission(); GroupPlans();
#if KXC_USE_LLVM
        RealLLVM();
        WholeGraphBinding();
#else
        std::cout << "[SKIP] real LLVM evidence requires KXC_USE_LLVM\n";
#endif
        std::cout << "[PASS] distributed ownership, groups, persistence and admission\n"; return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
