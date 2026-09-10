// Actual bounded decode and download-free position-window/control checks.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#if defined(KXC_ADAPTIVE_CUDA_TEST)
#include "kxc/compiler/adaptive_hot_swap.h"
#include "../src/compiler/internal/compiled_graph_access.h"
#include "../src/runtime/internal/compiled_module_node.h"
#endif
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/device_api.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;
bool cuda = false;
std::vector<DeviceStream> cuda_streams;
size_t next_stream = 0;
size_t reference_values = 0;
Device TestDevice() { return cuda ? Device::CUDA() : Device::CPU(); }
const char* Backend() { return cuda ? "CUDA" : "LLVM"; }
const std::vector<std::pair<int64_t, int64_t>> kCases{{1,0},{1,1},{2,4},{3,8}};
const std::vector<restricted::InputAxisSymbol> kAxes{{0,0,"B",1,3,1},{0,1,"P",0,8,1},{1,0,"B",1,3,1}};

void Check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
Array<NDArray> Run(const runtime::RuntimeSession& session, const Array<NDArray>& inputs,
                   runtime::ExecutionMetadata metadata = {}, bool commits_state = false) {
    if (!cuda) return session.Run(inputs, metadata);
    metadata.insert({"model", "minimind_bounded"});
    const auto result = session.RunAsync(inputs, cuda_streams.at(next_stream++ % 2), metadata);
    if (commits_state) Check(result.completion.IsReady(), "bounded state returned before CUDA commit completed");
    result.completion.Wait();
    for (const auto& output : result.outputs) {
        Check(output.device() == TestDevice(), "CUDA consumer received an output on another device");
    }
    return result.outputs;
}
template<class Action> void Rejects(Action action, const std::string& diagnostic) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "wrong rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + diagnostic);
}
bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
        a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
        a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
bool SameShape(const Array<int64_t>& a, const Array<int64_t>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
Expr Op(const std::string& name, Array<Expr> args, relay::Attrs attrs = {}) {
    return Call(relay::Op::Get(name), std::move(args), std::move(attrs));
}
NDArray Empty(Array<int64_t> shape, const std::string& dtype = "float32") {
    return NDArray::Empty(shape, runtime::DataTypeFromString(dtype), TestDevice());
}
template<class T> std::vector<T> Read(const NDArray& array) {
    Check(array.NBytes() % sizeof(T) == 0, "invalid tensor byte length");
    std::vector<T> values(array.NBytes() / sizeof(T));
    if (!values.empty()) array.CopyToBytes(values.data(), array.NBytes());
    return values;
}
template<class T> std::vector<T> ReadFile(const std::filesystem::path& file, size_t count) {
    std::ifstream in(file, std::ios::binary);
    Check(in.good(), "missing fixture: " + file.string());
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    Check(in.gcount() == static_cast<std::streamsize>(count * sizeof(T)) &&
          in.peek() == std::char_traits<char>::eof(), "fixture byte length mismatch: " + file.string());
    return values;
}
Expr Integer(int64_t value) {
    auto data = NDArray::Empty({1}, runtime::DataTypeFromString("int64"), Device::CPU());
    data.CopyFromBytes(&value, sizeof(value));
    return Constant(data);
}
NDArray Weights(Array<int64_t> shape, int seed, Device device = TestDevice()) {
    auto data = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), device);
    std::vector<float> values(data.NBytes() / 4);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(double(i + seed) * .31) * .5;
    data.CopyFromBytes(values.data(), data.NBytes());
    return data;
}
Expr Extent(Expr anchor, int64_t axis = 1) {
    return Op("gather", {Op("shape_of", {anchor}), Integer(axis)}, relay::GatherAttrs::Create(0));
}
std::pair<size_t, size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(in.good(), "missing bounded decode profile");
    std::pair<size_t, size_t> counts{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"") != std::string::npos ||
            line.find("\"event_type\":\"copy\"") != std::string::npos) ++counts.second;
    }
    return counts;
}

Expr Window(Expr table, Expr past, int64_t count = 1) {
    const auto start = Extent(past);
    return Op("slice",{table,start,Op("add",{start,Integer(count)}),Integer(0),Integer(1)});
}
void TestWindow(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context, int64_t window_count) {
    const Var past("past",TensorType({1,4,4},"float32")), token("token",TensorType({1,1,4},"float32"));
    const auto table = Weights({8+window_count,4},7,Device::CPU());
    const auto present = Op("concatenate",{past,token},relay::ConcatenateAttrs::Create(1));
    const auto total = Op("add",{Extent(past),Integer(1)});
    const auto previous = Op("subtract",{Extent(present),Integer(1)});
    const auto target = Op("concatenate",{Integer(1),previous},relay::ConcatenateAttrs::Create(0));
    const auto fill = Op("constant_of_shape",{target},relay::ConstantOfShapeAttrs::Create({},0,0));
    auto zero = NDArray::Empty({1,1},runtime::DataTypeFromString("float32"),Device::CPU());
    const float value = 0; zero.CopyFromBytes(&value,4);
    const auto mask = Op("concatenate",{fill,Constant(zero)},relay::ConcatenateAttrs::Create(1));
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(
        Function({past,token},Tuple({Window(Constant(table),past,window_count),present,total,mask})),config,kAxes)));
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto before = Counts(context);
    const auto positions = Read<float>(table);
    for (const auto& [b,p] : kCases) {
        const auto input = Weights({b,p,4},1), next = Weights({b,1,4},2);
        const auto out = Run(session,{input,next},{{"stage","decode_window_controls"},
            {"batch",std::to_string(b)},{"past",std::to_string(p)}});
        Check(out.size() == 4 && SameShape(out[0].shape(),{window_count,4}) && SameShape(out[1].shape(),{b,p+1,4}) &&
            SameShape(out[3].shape(),{1,p+1}),"decode window/control output shape mismatch");
        const auto window = Read<float>(out[0]), a = Read<float>(input), z = Read<float>(next), actual = Read<float>(out[1]);
        Check(std::equal(window.begin(),window.end(),positions.begin()+p*4),"position window used a representative offset");
        std::vector<float> reference;
        for (int64_t row = 0; row < b; ++row) {
            reference.insert(reference.end(),a.begin()+row*p*4,a.begin()+(row+1)*p*4);
            reference.insert(reference.end(),z.begin()+row*4,z.begin()+(row+1)*4);
        }
        Check(actual == reference && Read<int64_t>(out[2]) == std::vector<int64_t>{p+1},"decode payload or P+1 shape control mismatch");
        const auto padding = Read<float>(out[3]);
        Check(std::all_of(padding.begin(),padding.end(),[](float v) { return v == 0; }),"total-current mask prefix is wrong");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"decode control Run changed cache");
    }
    const auto after = Counts(context);
    Check(after.first == before.first + compiled.plan().calls().size()*4,"decode control kernels were not executed");
    for (const auto& input : Array<NDArray>{Empty({1,9,4}),Empty({4,4,4}),Empty({1,4}),Empty({1,4,4},"int32")}) {
        Rejects([&] { (void)Run(session,{input,Empty({1,1,4})}); },"RuntimeSession");
        Check(Counts(context) == after && SameStats(stats,ci::GetPrimitiveCacheStats()),"bad window input allocated, launched or touched cache");
    }
    const auto reject = [&](Expr expression, const std::string& message) {
        Rejects([&] { (void)Adapter::Prepare(Function({past,token},expression),config,kAxes); },message);
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"invalid control proof changed cache");
    };
    reject(Window(Constant(table),past,window_count+1),"upper bound exceeds table capacity");
    reject(Op("subtract",{Extent(past),Integer(1)}),"nonnegative constant offset");
    reject(Op("add",{Extent(past),Extent(past)}),"nonnegative constant");
    reject(Op("subtract",{past,token}),"proved shape controls");
    reject(Op("slice",{Constant(table),past},relay::SliceAttrs::Create({}, {}, {}, {}, 0,1,1)),"arity mismatch");
    reject(Op("slice",{Constant(table)},relay::SliceAttrs::Create({0},{1},{0},{1},-1,-1,1)),"prepared shape anchor");
    std::cout << "decode position/control C=" << window_count << ": four B/P shapes, " << compiled.plan().calls().size()*4
              << " " << Backend() << " calls, exact window/payload/control/mask; four runtime and six proof rejections\n";
}

api::PlanAbiFingerprint StatePlanAbi(const api::CompiledGraph& graph, const runtime::ExecutablePlan& plan) {
    std::vector<api::OrderedArtifactIdentity> artifacts;
    const auto calls = plan.calls();
    const auto& pins = graph.artifact_pins();
    for (size_t i = 0; i < calls.size(); ++i) {
        artifacts.push_back({i,std::string(calls[i]->symbol),pins[i].record().artifact_key});
    }
    return api::BuildPlanAbiFingerprint(graph.module(),plan,artifacts);
}
void TestStateBridge(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context) {
    const Var past("past",TensorType({1,2,4},"float32")), token("token",TensorType({1,1,4},"float32"));
    const auto present = Op("concatenate",{past,token},relay::ConcatenateAttrs::Create(1));
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(
        Function({past,token},Tuple({Op("nn_relu",{present}),present})),config,
        {{0,0,"B",1,3,1},{0,1,"P",0,5,1},{1,0,"B",1,3,1}})));
    const auto base = compiled.plan();
    Check(base.calls().size() == 2,"state bridge should execute ordinary concat and relu");
    const int64_t state_id = base.input_value_ids()[0], source_id = base.output_value_ids()[1];
    const std::vector<runtime::StateOutputBinding> bindings{{state_id,source_id,1,-1,1}};
    const auto stats = ci::GetPrimitiveCacheStats();
    size_t runs = 0;
    std::vector<NDArray> retained_states;
    std::vector<std::vector<float>> retained_state_values;
    NDArray retained_output;
    std::vector<float> retained_output_values;
    for (int64_t batch : {1,2,3}) {
        for (float sentinel : {1e20f,-1e20f}) {
            const auto plan = base.BindBoundedStateOutputs(bindings,{{batch,6,4}},sentinel);
            const auto abi = StatePlanAbi(compiled,plan);
            Check(abi.canonical_bytes().find("executable-plan-abi-v11-bounded-external-stateful-v1") != std::string::npos,
                  "bounded state ABI kind missing");
            Check(abi != api::BuildPlanAbiFingerprint(compiled) &&
                  abi != StatePlanAbi(compiled,base.BindBoundedStateOutputs(bindings,{{batch,5,4}},sentinel)),
                  "state capacity is absent from plan ABI");
            runtime::RuntimeSession session(compiled.module(),plan);
            const void* address = session.StateValue(state_id)->dl_tensor.data;
            int64_t length = batch-1;
            auto seed = Weights({batch,length,4},7);
            auto expected = Read<float>(seed);
            session.InitializeState(state_id,seed,length);
            seed = {};
            const auto checkpoint = Counts(context);
            for (const auto& bad : Array<NDArray>{Empty({batch,2,4}),Empty({4,1,4}),Empty({batch,1,4},"int32"),
                    Empty({batch,4}),Empty({batch == 1 ? 2 : 1,1,4})}) {
                Rejects([&] { (void)Run(session,{bad}); },"RuntimeSession");
                Check(Counts(context) == checkpoint && session.StateExtent(state_id) == length,
                      "invalid state input allocated/launched or changed extent");
            }
            for (; length < 6; ++length) {
                const auto input = Weights({batch,1,4},static_cast<int>(length+11));
                const auto tokens = Read<float>(input);
                std::vector<float> appended;
                for (int64_t b = 0; b < batch; ++b) {
                    appended.insert(appended.end(),expected.begin()+b*length*4,expected.begin()+(b+1)*length*4);
                    appended.insert(appended.end(),tokens.begin()+b*4,tokens.begin()+(b+1)*4);
                }
                const auto out = Run(session,{input},{{"stage","bounded_state_bridge"},
                    {"batch",std::to_string(batch)},{"past",std::to_string(length)},
                    {"state_extent_after",std::to_string(length+1)},{"plan_abi",abi.digest()}},true);
                ++runs;
                auto clipped = appended;
                for (auto& value : clipped) value = std::max(0.0f,value);
                Check(out.size() == 1 && Read<float>(out[0]) == clipped,"state prefix packing or readback mismatch");
                if (!retained_output.defined()) {
                    retained_output = out[0];
                    retained_output_values = clipped;
                }
                Check(session.StateExtent(state_id) == length+1 && session.StateValue(state_id)->dl_tensor.data == address,
                      "state extent or fixed storage changed incorrectly");
                const auto stored = Read<float>(session.StateValue(state_id));
                for (int64_t b = 0; b < batch; ++b) {
                    for (int64_t p = 0; p < 6; ++p) {
                        for (int64_t d = 0; d < 4; ++d) {
                            const float want = p <= length ? appended[(b*(length+1)+p)*4+d] : sentinel;
                            Check(stored[(b*6+p)*4+d] == want,"state append changed prefix or invalid capacity");
                        }
                    }
                }
                expected = std::move(appended);
                Check(StatePlanAbi(compiled,plan) == abi && SameStats(stats,ci::GetPrimitiveCacheStats()),
                      "state cursor changed ABI or touched compiler cache");
            }
            const auto full = Counts(context);
            const auto saved = Read<float>(session.StateValue(state_id));
            Rejects([&] { (void)Run(session,{Empty({batch,1,4})}); },"capacity");
            Check(Counts(context) == full && Read<float>(session.StateValue(state_id)) == saved &&
                  session.StateExtent(state_id) == 6,"full cache rejection changed state or launched");
            const auto retained = session.StateValue(state_id);
            for (const auto& other : retained_states) {
                Check(other->dl_tensor.data != retained->dl_tensor.data,"independent sessions shared capacity storage");
            }
            retained_states.push_back(retained);
            retained_state_values.push_back(saved);
        }
    }
    for (size_t i = 0; i < retained_states.size(); ++i) {
        Check(Read<float>(retained_states[i]) == retained_state_values[i],"state handle lost storage after session destruction");
    }
    Check(Read<float>(retained_output) == retained_output_values,"later runs or session destruction invalidated an output");
    const auto bad_bind = [&](std::vector<runtime::StateOutputBinding> items, std::vector<Array<int64_t>> shapes) {
        Rejects([&] { (void)base.BindBoundedStateOutputs(items,shapes); },"");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"bad state binding touched cache");
    };
    bad_bind(bindings,{});
    bad_bind(bindings,{{1,-1,4}});
    bad_bind(bindings,{{1,6,5}});
    bad_bind(bindings,{{4,6,4}});
    bad_bind({{state_id,source_id,1,0,1}},{{1,6,4}});
    bad_bind({{state_id,source_id,1,-1,7}},{{1,6,4}});
    bad_bind({{state_id,source_id,1,-1,1,99}},{{1,6,4}});
    bad_bind({bindings[0],bindings[0]},{{1,6,4},{1,6,4}});
    bad_bind({{state_id,base.output_value_ids()[0],1,-1,1}},{{1,6,4}});
    const auto before_bad_module = Counts(context);
    const auto wrong_count = base.BindBoundedStateOutputs({{state_id,source_id,1,-1,2}},{{1,6,4}});
    Rejects([&] { (void)runtime::RuntimeSession(compiled.module(),wrong_count); },"module output extent contract");
    Check(Counts(context) == before_bad_module,"wrong append contract allocated state or launched");
    Check(runs == 30,"bounded state bridge lost a batch/sentinel/length case");
    std::cout << "bounded state bridge: B=1/2/3, two sentinels, " << runs*2
              << " " << Backend() << " calls; exact prefixes/appends, stable capacity addresses, 36 runtime and ten binding/module rejections\n";
}

runtime::ExecutablePlan FullStatePlan(const api::CompiledGraph& graph, int64_t batch, int64_t capacity) {
    const auto base = graph.plan();
    Check(base.input_value_ids().size() == 17 && base.output_value_ids().size() == 17,"invalid full decode binding ports");
    std::vector<runtime::StateOutputBinding> bindings;
    std::vector<Array<int64_t>> shapes;
    for (size_t i = 1; i < 17; ++i) {
        bindings.push_back({base.input_value_ids()[i],base.output_value_ids()[i],1,-1,1});
        shapes.push_back({batch,capacity,4,96});
    }
    return base.BindBoundedStateOutputs(std::move(bindings),std::move(shapes),1e20f);
}
std::vector<float> FullStatePrefix(const runtime::RuntimeSession& session, int64_t state_id, int64_t length) {
    const auto state = session.StateValue(state_id);
    const auto shape = state.shape();
    Check(shape.size() == 4 && shape[2] == 4 && shape[3] == 96 && session.StateExtent(state_id) == length,
          "state layout or committed extent changed");
    const auto data = Read<float>(state);
    std::vector<float> prefix;
    for (int64_t b = 0; b < shape[0]; ++b) {
        const auto begin = data.begin()+b*shape[1]*384;
        prefix.insert(prefix.end(),begin,begin+length*384);
        Check(std::all_of(begin+length*384,begin+shape[1]*384,[](float v) { return v == 1e20f; }),
              "full model changed invalid capacity region");
    }
    return prefix;
}
Array<NDArray> PrefillSeed(const api::CompileConfig& config, const std::filesystem::path& root) {
    const char* directory = std::getenv("KXC_MINIMIND_BOUNDED_PREFILL_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] actual bounded prefill-to-state handoff fixture unset\n"; return {}; }
    const std::filesystem::path prefill_root(directory);
    const auto source = frontend::LoadONNXShapeSource((prefill_root/"prefill.json").string(),(prefill_root/"prefill.params").string());
    Check(source.input_names.size() == 1 && source.input_names[0] == "input_ids" && source.declared_output_types.size() == 17,
          "invalid prefill state handoff ports");
    std::cout << "compiling full bounded prefill for owned-state handoff\n" << std::flush;
    Array<NDArray> seed;
    size_t prefill_calls = 0;
    {
        const auto prefill = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(
            source.function,config,{{0,0,"B",1,3,1},{0,1,"S",1,8,1}},source.declared_output_types)));
        prefill_calls = prefill.plan().calls().size();
        Check(prefill_calls == 742,"state handoff lost a prefill layer");
        auto ids = Empty({1,4},"int64");
        const auto tokens = ReadFile<int64_t>(root/"loop_seed_ids.bin",4);
        ids.CopyFromBytes(tokens.data(),ids.NBytes());
        std::ifstream receipt(prefill_root/"export_receipt.txt"); std::string sha; receipt >> sha;
        const runtime::RuntimeSession prefill_session(prefill.module(),prefill.plan());
        const auto stats = ci::GetPrimitiveCacheStats();
        seed = Run(prefill_session,{ids},{{"stage","minimind_bounded_state_prefill"},
            {"batch","1"},{"sequence","4"},{"layers","8"},{"export_receipt",sha},
            {"plan_abi",api::BuildPlanAbiFingerprint(prefill).digest()}});
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"state seed prefill Run touched cache");
    }
    Check(seed.size() == 17,"state handoff lost prefill outputs");
    return seed;
}
void TestStateGreedy(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context,
                     const api::CompiledGraph& decode, const std::filesystem::path& root,
                     const std::string& decode_sha, Array<NDArray> seed) {
    if (seed.empty()) seed = PrefillSeed(config,root);
    if (seed.empty()) return;
    const auto seed_logits = Read<float>(seed[0]);
    std::vector<float> logits(seed_logits.end()-6400,seed_logits.end());
    const auto plan = FullStatePlan(decode,1,8);
    const auto abi = StatePlanAbi(decode,plan);
    runtime::RuntimeSession session(decode.module(),plan);
    std::vector<const void*> addresses;
    double worst = 0;
    for (size_t i = 0; i < 17; ++i) {
        const auto actual = i == 0 ? logits : Read<float>(seed[i]);
        const auto reference = ReadFile<float>(root/("loop_seed_"+std::to_string(i)+".bin"),actual.size());
        reference_values += actual.size();
        for (size_t j = 0; j < actual.size(); ++j) {
            Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),"nonfinite compiled prefill seed/reference");
            worst = std::max(worst,std::abs(double(actual[j])-reference[j]));
        }
        if (i == 0) continue;
        const int64_t state_id = decode.plan().input_value_ids()[i];
        session.InitializeState(state_id,seed[i],4);
        Check(FullStatePrefix(session,state_id,4) == actual,"prefill state initialization changed payload");
        addresses.push_back(session.StateValue(state_id)->dl_tensor.data);
    }
    seed = {};
    const auto stats = ci::GetPrimitiveCacheStats();
    std::ifstream steps(root/"loop_steps.txt");
    for (int64_t step = 0; step < 4; ++step) {
        const int64_t token = std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
        int64_t expected_token = -1, expected_past = -1;
        Check(bool(steps >> expected_token >> expected_past) && expected_token == token && expected_past == 4+step,
              "owned-state LLVM greedy differs from reference");
        auto ids = Empty({1,1},"int64"); ids.CopyFromBytes(&token,sizeof(token));
        const auto outputs = Run(session,{ids},
            {{"stage","minimind_bounded_state_greedy"},{"batch","1"},{"past",std::to_string(4+step)},
             {"step",std::to_string(step)},{"state_extent_after",std::to_string(5+step)},
             {"layers","8"},{"export_receipt",decode_sha},{"plan_abi",abi.digest()}},true);
        Check(outputs.size() == 1,"owned-state decode must return logits only");
        logits = Read<float>(outputs[0]);
        for (size_t output = 0; output < 17; ++output) {
            const auto actual = output == 0 ? logits
                : FullStatePrefix(session,decode.plan().input_value_ids()[output],5+step);
            const auto reference = ReadFile<float>(root/("loop_ref_"+std::to_string(step)+"_"+std::to_string(output)+".bin"),actual.size());
            reference_values += actual.size();
            for (size_t j = 0; j < actual.size(); ++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),"nonfinite owned-state greedy output/reference");
                worst = std::max(worst,std::abs(double(actual[j])-reference[j]));
            }
            if (output != 0) {
                Check(session.StateValue(decode.plan().input_value_ids()[output])->dl_tensor.data == addresses[output-1],
                      "owned-state decode replaced capacity storage");
            }
        }
        Check(StatePlanAbi(decode,plan) == abi && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "owned-state decode changed identity or touched cache");
    }
    const auto full = Counts(context);
    Rejects([&] { (void)Run(session,{Empty({1,1},"int64")}); },"capacity");
    Check(Counts(context) == full && worst < 5e-5,"full owned state launched or model numeric comparison failed");
    std::cout << "actual bounded " << Backend() << " prefill -> owned state -> greedy: 742 prefill + "
              << decode.plan().calls().size()*4 << " decode " << Backend() << " calls; 16 stable caches, extents 4->8, max_abs_error="
              << worst << "; capacity rejection before allocation/launch\n";
}

void TestRequestBatching(const api::CompiledGraph& graph,
    const std::shared_ptr<profiling::ProfileContext>& context,
    const std::filesystem::path& root, const std::string& sha) {
    const auto plan = FullStatePlan(graph,3,9).BindRequestBatching(2);
    const runtime::RuntimeSession batched(graph.module(),plan), reference(graph.module(),graph.plan());
    const auto batch_abi = StatePlanAbi(graph,plan);
    const auto reference_abi = api::BuildPlanAbiFingerprint(graph).digest();
    const auto stats = ci::GetPrimitiveCacheStats();
    struct Request final {
        uint64_t id{0};
        Array<NDArray> past;
        NDArray token;
        int64_t extent{0};
    };
    const auto admit = [&](size_t case_index, size_t row) {
        const auto [batch,past] = kCases.at(case_index);
        Check(row < static_cast<size_t>(batch),"invalid request fixture row");
        Request request;
        request.extent = past;
        request.token = Empty({1,1},"int64");
        const auto tokens = ReadFile<int64_t>(root/("input_"+std::to_string(case_index)+".bin"),batch);
        request.token.CopyFromBytes(&tokens[row],sizeof(int64_t));
        for (size_t layer = 0; layer < 16; ++layer) {
            auto value = Empty({1,past,4,96});
            const auto data = ReadFile<float>(root/("past_"+std::to_string(case_index)+"_"+
                std::to_string(layer)+".bin"),batch*value.NBytes()/sizeof(float));
            if (value.NBytes()) value.CopyFromBytes(data.data()+row*value.NBytes()/sizeof(float),value.NBytes());
            request.past.push_back(value);
        }
        request.id = batched.AdmitRequest(request.past,past);
        return request;
    };
    auto a = admit(2,0), b = admit(0,0), c = admit(2,1);
    if (cuda) {
        const auto cpu_token = NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
        const auto before = Counts(context);
        Rejects([&] { batched.EnqueueRequest(a.id,{cpu_token}); },"device");
        Rejects([&] { (void)batched.RunNextBatch(DeviceStream{},{}); },"defined DeviceStream");
        Rejects([&] { (void)batched.RunNextBatch(DeviceStream::Default(Device::CPU()),{}); },"stream device");
        Check(Counts(context) == before,"rejected request allocated, copied or launched");
    }
    size_t runs = 0, request_steps = 0;
    size_t onnx_values = 0;
    double onnx_worst = 0;
    const auto execute = [&](const std::vector<Request*>& expected_requests,
                             const std::string& label, bool initial_fixture) {
        std::vector<Array<NDArray>> independent;
        for (const auto* request : expected_requests) {
            Array<NDArray> inputs{request->token};
            for (const auto& state : request->past) inputs.push_back(state);
            independent.push_back(Run(reference,inputs,{{"stage","request_batch_reference"},
                {"case",label},{"batch","1"},{"past",std::to_string(request->extent)},
                {"layers","8"},{"export_receipt",sha},{"plan_abi",reference_abi}}));
        }
        const auto before = Counts(context).first;
        const runtime::ExecutionMetadata metadata{{"model","minimind_bounded"},
            {"stage","minimind_request_batching"},{"case",label},{"layers","8"},{"export_receipt",sha},
            {"batch",std::to_string(expected_requests.size())},
            {"past",std::to_string(expected_requests[0]->extent)},
            {"state_extent_after",std::to_string(expected_requests[0]->extent+1)},
            {"plan_abi",batch_abi.digest()},{"request_ids","caller_must_not_override"}};
        const auto results = cuda
            ? batched.RunNextBatch(cuda_streams.at(next_stream++ % 2),metadata)
            : batched.RunNextBatch(metadata);
        Check(results.size() == expected_requests.size() && Counts(context).first-before == graph.plan().calls().size(),
              "full-model requests were not served by exactly one graph execution");
        ++runs; request_steps += results.size();
        for (size_t row = 0; row < results.size(); ++row) {
            auto& request = *expected_requests[row];
            Check(results[row].request_id == request.id && results[row].outputs.size() == 1,
                  "full-model request identity or private state output leaked");
            Check(results[row].outputs[0].device() == TestDevice() &&
                  Read<float>(results[row].outputs[0]) == Read<float>(independent[row][0]),
                  "batched full-model logits differ from independent execution");
            Check(batched.RequestExtent(request.id) == request.extent+1,"request cursor did not commit exactly one token");
            for (size_t layer = 0; layer < 16; ++layer) {
                const auto state = batched.CopyRequestState(request.id,plan.state_value_ids()[layer]);
                Check(state.device() == TestDevice() && SameShape(state.shape(),{1,request.extent+1,4,96}) &&
                    Read<float>(state) == Read<float>(independent[row][layer+1]),
                    "batched full-model KV differs from independent execution");
            }
            if (initial_fixture) {
                const size_t case_index = request.extent == 0 ? 0 : 2;
                const size_t fixture_row = request.id == c.id ? 1 : 0;
                const int64_t fixture_batch = kCases[case_index].first;
                for (size_t output = 0; output < 17; ++output) {
                    const auto actual = Read<float>(independent[row][output]);
                    const auto expected = ReadFile<float>(root/("ref_"+std::to_string(case_index)+"_"+
                        std::to_string(output)+".bin"),fixture_batch*actual.size());
                    for (size_t i = 0; i < actual.size(); ++i) {
                        Check(std::isfinite(actual[i]) && std::isfinite(expected[fixture_row*actual.size()+i]),
                              "nonfinite request output or reference");
                        onnx_worst = std::max(onnx_worst,std::abs(double(actual[i])-expected[fixture_row*actual.size()+i]));
                    }
                    onnx_values += actual.size();
                }
            }
            request.past = {};
            for (size_t layer = 1; layer < 17; ++layer) request.past.push_back(independent[row][layer]);
            const auto logits = Read<float>(independent[row][0]);
            const int64_t next_token = std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
            request.token = Empty({1,1},"int64"); request.token.CopyFromBytes(&next_token,sizeof(next_token));
            ++request.extent;
        }
        Check(StatePlanAbi(graph,plan) == batch_abi && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "full-model request routing changed identity or touched the primitive cache");
        return results;
    };
    for (const auto* request : {&a,&b,&c}) batched.EnqueueRequest(request->id,{request->token});
    if (cuda) {
        const auto before = Counts(context);
        Rejects([&] { (void)batched.RunNextBatch(DeviceStream::Default(Device::CPU()),{}); },"stream device");
        Check(Counts(context) == before && batched.RequestExtent(a.id) == 4 && batched.RequestExtent(b.id) == 0,
              "wrong stream packed inputs or changed queued requests");
    }
    const auto retained = execute({&a,&c},"admit_ac_p4_skip_b_p0",true);
    const auto retained_logits = Read<float>(retained[0].outputs[0]);
    Check(batched.RequestExtent(b.id) == 0,"incompatible queued request was changed");
    (void)execute({&b},"b_p0",true);
    batched.EnqueueRequest(a.id,{a.token});
    batched.ReleaseRequest(a.id);
    auto d = admit(0,0);
    Check(d.id != a.id,"full-model reused a departed request id");
    Rejects([&] { batched.EnqueueRequest(a.id,{a.token}); },"released request");
    batched.EnqueueRequest(d.id,{d.token});
    (void)execute({&d},"replacement_d_p0",true);
    for (const auto* request : {&b,&c,&d}) batched.EnqueueRequest(request->id,{request->token});
    (void)execute({&b,&d},"merge_bd_p1_skip_c_p5",false);
    (void)execute({&c},"c_p5",false);
    Check(batched.RunNextBatch().empty() && Read<float>(retained[0].outputs[0]) == retained_logits,
          "full-model departure left queued work or invalidated output storage");
    Check(onnx_worst <= (cuda ? 5e-5 : 1e-3),"request batching exceeded full-model ONNX tolerance");
    Check(runs == 5 && request_steps == 7,"full-model request scenario drifted");
    std::cout << "actual eight-layer request batching: 3 KV slots, 4 admitted requests, 7 request steps in 5 batches; "
              << runs*graph.plan().calls().size() << ' ' << Backend() << " submits vs " << request_steps*graph.plan().calls().size()
              << " independent; logits and 16 KV states bitwise equal, ONNX max_abs_error=" << onnx_worst
              << "; departure/reuse, P=0/1/4/5 grouping, cache unchanged\n";
    if (cuda) std::cout << "[PASS] bounded_minimind_requests: batches=5 request_steps=7 calls=774 streams=2 reference_values="
                       << onnx_values << '\n';
}

void TestActual(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context,
                Array<NDArray> seed = {}, bool batching_only = false) {
    const char* directory = std::getenv("KXC_MINIMIND_BOUNDED_DECODE_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] full MiniMind decode: set KXC_MINIMIND_BOUNDED_DECODE_DIR\n"; return; }
    const std::filesystem::path root(directory);
    const auto source = frontend::LoadONNXShapeSource((root/"decode.json").string(),(root/"decode.params").string());
    Check(source.input_names.size() == 17 && source.input_names[0] == "input_ids" && source.declared_output_types.size() == 17,
          "full decode boundary drifted");
    std::vector<restricted::InputAxisSymbol> axes{{0,0,"B",1,3,1}};
    for (size_t i = 1; i < 17; ++i) {
        Check(source.input_names[i] == "past_" + std::string(i%2 ? "k_" : "v_") + std::to_string((i-1)/2),"decode KV order drifted");
        axes.push_back({i,0,"B",1,3,1}); axes.push_back({i,1,"P",0,8,1});
    }
    std::cout << "full MiniMind decode source loaded\n" << std::flush;
    const auto prepared = Adapter::Prepare(source.function,config,axes,source.declared_output_types);
    std::cout << "full MiniMind decode shape preparation complete\n" << std::flush;
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    std::cout << "full MiniMind decode request minted\n" << std::flush;
    const auto compiled = api::Compiler::CompileBounded(request);
    const size_t calls = compiled.plan().calls().size();
    Check(calls == 774,"full decode must retain all 774 ordinary units");
    if (cuda) {
        for (const auto& symbol : compiled.module().symbols()) {
            const auto launch = compiled.module().launch_metadata(symbol);
            Check(launch->backend == codegen::CodeGenBackend::kCUDA && launch->device == TestDevice(),
                  "full bounded decode selected a non-CUDA implementation");
        }
    }
    std::cout << "full MiniMind decode compiled: " << calls << " calls per Run\n" << std::flush;
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    std::ifstream receipt(root/"export_receipt.txt"), cases(root/"cases.txt");
    std::string sha; receipt >> sha;
    Check(sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string::npos,"invalid decode receipt");
    if (batching_only) {
        TestRequestBatching(compiled,context,root,sha);
        return;
    }
    int64_t b = 0,p = 0; size_t case_index = 0; double worst = 0;
    while (cases >> b >> p) {
        Check(case_index < kCases.size() && kCases[case_index] == std::make_pair(b,p),"unexpected decode case");
        Array<NDArray> inputs;
        auto ids = Empty({b,1},"int64");
        const auto tokens = ReadFile<int64_t>(root/("input_"+std::to_string(case_index)+".bin"),b);
        ids.CopyFromBytes(tokens.data(),ids.NBytes()); inputs.push_back(ids);
        for (size_t i = 0; i < 16; ++i) {
            auto state = Empty({b,p,4,96});
            const auto values = ReadFile<float>(root/("past_"+std::to_string(case_index)+"_"+std::to_string(i)+".bin"),state.NBytes()/4);
            if (!values.empty()) state.CopyFromBytes(values.data(),state.NBytes());
            inputs.push_back(state);
        }
        const auto outputs = Run(session,inputs,{{"stage","minimind_bounded_decode"},{"batch",std::to_string(b)},
            {"past",std::to_string(p)},{"layers","8"},{"export_receipt",sha},
            {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(outputs.size() == 17,"full decode must return logits and 16 KV outputs");
        for (size_t i = 0; i < outputs.size(); ++i) {
            Check(SameShape(outputs[i].shape(),i == 0 ? Array<int64_t>{b,1,6400} : Array<int64_t>{b,p+1,4,96}),"full decode shape mismatch");
            const auto actual = Read<float>(outputs[i]);
            const auto reference = ReadFile<float>(root/("ref_"+std::to_string(case_index)+"_"+std::to_string(i)+".bin"),actual.size());
            reference_values += actual.size();
            for (size_t j = 0; j < actual.size(); ++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),"nonfinite decode output");
                worst = std::max(worst,std::abs(double(actual[j])-reference[j]));
            }
            if (i != 0) {
                const auto old = Read<float>(inputs[i]);
                for (int64_t row = 0; row < b; ++row) {
                    Check(std::equal(old.begin()+row*p*384,old.begin()+(row+1)*p*384,actual.begin()+row*(p+1)*384),"decode changed past KV prefix");
                }
            }
        }
        const auto state_plan = FullStatePlan(compiled,b,9);
        runtime::RuntimeSession owned(compiled.module(),state_plan);
        for (size_t i = 1; i < 17; ++i) {
            if (case_index == 2 && i == 16) {
                const auto before_bad = Counts(context);
                Rejects([&] { (void)Run(owned,{inputs[0]}); },"RuntimeSession");
                Check(Counts(context) == before_bad,"mismatched layer extents allocated/launched");
            }
            owned.InitializeState(compiled.plan().input_value_ids()[i],inputs[i],p);
        }
        const auto owned_output = Run(owned,{inputs[0]},{{"stage","minimind_bounded_owned_decode"},
            {"batch",std::to_string(b)},{"past",std::to_string(p)},{"state_extent_after",std::to_string(p+1)},
            {"layers","8"},{"export_receipt",sha},{"plan_abi",StatePlanAbi(compiled,state_plan).digest()}},true);
        Check(owned_output.size() == 1 && Read<float>(owned_output[0]) == Read<float>(outputs[0]),
              "owned-state logits differ from fresh-output LLVM");
        for (size_t i = 1; i < 17; ++i) {
            Check(FullStatePrefix(owned,compiled.plan().input_value_ids()[i],p+1) == Read<float>(outputs[i]),
                  "owned-state present differs from fresh-output LLVM");
        }
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"full decode Run changed cache");
        ++case_index;
    }
    Check(case_index == 4 && cases.eof() && worst < 5e-5,"full decode differs from ONNX reference");
    // Start from ONNX prefill's K/V and logits, then feed each actual LLVM
    // result to the next Run. This test owns external inputs; it is not a
    // substitute for RuntimeSession persistent-state integration.
    auto logits = ReadFile<float>(root/"loop_seed_0.bin",6400);
    Array<NDArray> states;
    for (size_t i = 1; i < 17; ++i) {
        auto value = Empty({1,4,4,96});
        const auto data = ReadFile<float>(root/("loop_seed_"+std::to_string(i)+".bin"),value.NBytes()/4);
        value.CopyFromBytes(data.data(),value.NBytes()); states.push_back(value);
    }
    std::ifstream loop_steps(root/"loop_steps.txt");
    Check(loop_steps.good(),"missing greedy reference steps");
    double loop_worst = 0;
    for (int64_t step = 0; step < 4; ++step) {
        const int64_t token = std::distance(logits.begin(),std::max_element(logits.begin(),logits.end()));
        int64_t expected_token = -1, expected_past = -1;
        Check(bool(loop_steps >> expected_token >> expected_past) && expected_token == token && expected_past == 4+step,
              "LLVM greedy token or growing past differs from reference");
        auto ids = Empty({1,1},"int64"); ids.CopyFromBytes(&token,sizeof(token));
        Array<NDArray> inputs{ids};
        for (const auto& state : states) inputs.push_back(state);
        const auto outputs = Run(session,inputs,{{"stage","minimind_bounded_decode_loop"},
            {"batch","1"},{"past",std::to_string(4+step)},{"step",std::to_string(step)},
            {"layers","8"},{"export_receipt",sha},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(outputs.size() == 17,"greedy decode lost KV outputs");
        for (size_t output = 0; output < outputs.size(); ++output) {
            Check(SameShape(outputs[output].shape(),output == 0 ? Array<int64_t>{1,1,6400}
                : Array<int64_t>{1,5+step,4,96}),"greedy output shape did not grow");
            const auto actual = Read<float>(outputs[output]);
            const auto reference = ReadFile<float>(root/("loop_ref_"+std::to_string(step)+"_"+std::to_string(output)+".bin"),actual.size());
            reference_values += actual.size();
            for (size_t j = 0; j < actual.size(); ++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),"nonfinite greedy output");
                loop_worst = std::max(loop_worst,std::abs(double(actual[j])-reference[j]));
            }
            if (output == 0) logits = actual;
            else {
                const auto old = Read<float>(states[output-1]);
                Check(std::equal(old.begin(),old.end(),actual.begin()),"greedy decode changed a past KV prefix");
            }
        }
        states = {};
        for (size_t i = 1; i < outputs.size(); ++i) states.push_back(outputs[i]);
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"greedy Run touched cache");
    }
    loop_steps >> std::ws;
    Check(loop_steps.eof() && loop_worst < 5e-5,"greedy decode reference mismatch");
    const auto counts = Counts(context);
    const auto blank_inputs = [](int64_t batch = 1, int64_t length = 4) {
        Array<NDArray> inputs{Empty({batch,1},"int64")};
        for (int i = 0; i < 16; ++i) inputs.push_back(Empty({batch,length,4,96}));
        return inputs;
    };
    std::vector<Array<NDArray>> invalid{blank_inputs(1,9),blank_inputs(4),blank_inputs(0),{}};
    auto bad = blank_inputs(); bad[0] = Empty({1,2},"int64"); invalid.push_back(bad);
    bad = blank_inputs(); bad[16] = Empty({1,3,4,96}); invalid.push_back(bad);
    bad = blank_inputs(); bad[16] = Empty({2,4,4,96}); invalid.push_back(bad);
    bad = blank_inputs(); bad[0] = Empty({1},"int64"); invalid.push_back(bad);
    bad = blank_inputs(); bad[0] = Empty({1,1},"int32"); invalid.push_back(bad);
    if (cuda) {
        bad = blank_inputs();
        bad[0] = NDArray::Empty({1,1},runtime::DataTypeFromString("int64"),Device::CPU());
        invalid.push_back(bad);
    }
    for (const auto& inputs : invalid) {
        Rejects([&] { (void)Run(session,inputs); },"RuntimeSession");
        Check(Counts(context) == counts && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "invalid full decode input allocated, launched or touched cache");
    }
    if (cuda) {
        Rejects([&] { (void)session.RunAsync(blank_inputs(),DeviceStream::Default(Device::CPU())); },"RuntimeSession");
        Check(Counts(context) == counts && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "wrong stream allocated, launched or touched cache");
    }
    std::cout << "actual full eight-layer MiniMind decode: four B/P references, " << calls*4
              << " " << Backend() << " calls, all 17 outputs and exact past prefixes, max_abs_error=" << worst << '\n';
    std::cout << "actual greedy decode: four steps P=4..7, " << calls*4 << " " << Backend() << " calls, 17 outputs, max_abs_error="
              << loop_worst << (cuda ? "; eleven full-model preflight rejections\n" : "; nine full-model preflight rejections\n");
    std::cout << "actual owned-state decode: four B/P cases, " << calls*4
              << " " << Backend() << " calls; logits and 16 caches bitwise match fresh-output " << Backend() << "; mismatched layer extents reject before launch\n";
    TestStateGreedy(config,context,compiled,root,sha,std::move(seed));
    if (!cuda) TestRequestBatching(compiled,context,root,sha);
    if (cuda) std::cout << "[PASS] bounded_minimind_decode: fresh_runs=8 state_runs=8 calls=774 streams=2 reference_values="
                       << reference_values << '\n';
}

// Model-level current-token polymorphism.  The ordinary bounded decode test
// fixes input_ids to C=1; this opt-in fixture proves that one prepared plan can
// serve both multi-token prefill-shaped calls and later multi-token decode
// calls.  The external K/V extent is fixed at the representative P=4 in this
// slice; K/V remain caller-owned, so this test does not claim persistent state
// ownership or multi-token state append semantics.
void TestActualMultiToken(const api::CompileConfig& config,
                          const std::shared_ptr<profiling::ProfileContext>& context) {
    const char* directory = std::getenv("KXC_MINIMIND_BOUNDED_MULTITOKEN_DIR");
    if (!directory || !*directory) {
        std::cout << "[SKIP] multi-token MiniMind decode: set KXC_MINIMIND_BOUNDED_MULTITOKEN_DIR\n";
        return;
    }
    const std::filesystem::path root(directory);
    const auto source = frontend::LoadONNXShapeSource(
        (root / "decode_multi.json").string(),
        (root / "decode_multi.params").string());
    Check(source.input_names.size() == 17 && source.input_names[0] == "input_ids" &&
              source.declared_output_types.size() == 17,
          "multi-token decode boundary drifted");
    std::vector<restricted::InputAxisSymbol> axes{{0, 0, "B", 1, 3, 1},
                                                   {0, 1, "C", 1, 3, 1}};
    for (size_t i = 1; i < 17; ++i) {
        Check(source.input_names[i] == "past_" +
                  std::string(i % 2 ? "k_" : "v_") + std::to_string((i - 1) / 2),
              "multi-token KV order drifted");
        axes.push_back({i, 0, "B", 1, 3, 1});
    }
    const auto prepared = Adapter::Prepare(source.function, config, axes,
                                            source.declared_output_types);
    const auto compiled = api::Compiler::CompileBounded(
        Adapter::MintBoundedCompileRequest(prepared));
    const size_t calls = compiled.plan().calls().size();
    Check(calls == 782, "multi-token decode must retain all ordinary units: got " + std::to_string(calls));
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    std::ifstream cases(root / "cases.txt");
    Check(cases.good(), "missing multi-token cases");
    std::string sha;
    std::ifstream receipt(root / "export_receipt.txt");
    receipt >> sha;
    Check(sha.size() == 64, "invalid multi-token receipt");
    int64_t batch = 0, past = 0, current = 0;
    size_t case_index = 0;
    double worst = 0.0;
    size_t worst_case = 0, worst_output = 0;
    while (cases >> batch >> past >> current) {
        Check(case_index < 4, "unexpected multi-token case count");
        Array<NDArray> inputs;
        auto ids = Empty({batch, current}, "int64");
        const auto token_values = ReadFile<int64_t>(
            root / ("input_" + std::to_string(case_index) + ".bin"),
            static_cast<size_t>(batch * current));
        ids.CopyFromBytes(token_values.data(), ids.NBytes());
        inputs.push_back(ids);
        for (size_t i = 0; i < 16; ++i) {
            auto state = Empty({batch, past, 4, 96});
            const auto values = ReadFile<float>(
                root / ("past_" + std::to_string(case_index) + "_" +
                        std::to_string(i) + ".bin"),
                state.NBytes() / sizeof(float));
            if (!values.empty()) state.CopyFromBytes(values.data(), state.NBytes());
            inputs.push_back(state);
        }
        const auto outputs = session.Run(
            inputs, {{"stage", "minimind_bounded_multitoken"},
                     {"batch", std::to_string(batch)},
                     {"past", std::to_string(past)},
                     {"current", std::to_string(current)},
                     {"export_receipt", sha},
                     {"plan_abi", api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(outputs.size() == 17, "multi-token decode lost outputs");
        for (size_t output = 0; output < outputs.size(); ++output) {
            const Array<int64_t> expected_shape =
                output == 0 ? Array<int64_t>{batch, current, 6400}
                            : Array<int64_t>{batch, past + current, 4, 96};
            Check(SameShape(outputs[output].shape(), expected_shape),
                  "multi-token output shape mismatch");
            const auto actual = Read<float>(outputs[output]);
            const auto reference = ReadFile<float>(
                root / ("ref_" + std::to_string(case_index) + "_" +
                        std::to_string(output) + ".bin"),
                actual.size());
            for (size_t j = 0; j < actual.size(); ++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),
                      "nonfinite multi-token output");
                const double error = std::abs(double(actual[j]) - reference[j]);
                if (error > worst) { worst = error; worst_case = case_index; worst_output = output; }
            }
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()),
              "multi-token Run touched compiler cache");
        ++case_index;
    }
    Check(case_index == 4 && cases.eof() && worst < 5e-5,
          "multi-token decode differs from ONNX reference: max_abs_error=" + std::to_string(worst) +
          " case=" + std::to_string(worst_case) + " output=" + std::to_string(worst_output));

    // The same compiled plan consumes the four-token prefill's K/V and emits
    // two current-token rows, proving a real prefill→decode handoff.
    auto ids = Empty({1, 2}, "int64");
    const auto decode_ids = ReadFile<int64_t>(root / "multitoken_decode_ids.bin", 2);
    ids.CopyFromBytes(decode_ids.data(), ids.NBytes());
    Array<NDArray> handoff_inputs{ids};
    for (size_t i = 1; i < 17; ++i) {
        auto state = Empty({1, 4, 4, 96});
        const auto values = ReadFile<float>(
            root / ("multitoken_prefill_ref_" + std::to_string(i) + ".bin"),
            state.NBytes() / sizeof(float));
        state.CopyFromBytes(values.data(), state.NBytes());
        handoff_inputs.push_back(state);
    }
    const auto handoff = session.Run(
        handoff_inputs,
        {{"stage", "minimind_multitoken_prefill_to_decode"},
         {"past", "4"}, {"current", "2"}, {"export_receipt", sha},
         {"plan_abi", api::BuildPlanAbiFingerprint(compiled).digest()}});
    Check(handoff.size() == 17 && SameShape(handoff[0].shape(), {1, 2, 6400}),
          "prefill-to-decode handoff logits shape mismatch");
    for (size_t output = 0; output < handoff.size(); ++output) {
        const auto actual = Read<float>(handoff[output]);
        const auto reference = ReadFile<float>(
            root / ("multitoken_decode_ref_" + std::to_string(output) + ".bin"),
            actual.size());
        for (size_t j = 0; j < actual.size(); ++j) {
            Check(std::isfinite(actual[j]) && std::isfinite(reference[j]),
                  "nonfinite prefill-to-decode handoff");
            worst = std::max(worst, std::abs(double(actual[j]) - reference[j]));
        }
    }
    const auto count_before_reject = Counts(context);
    auto invalid = Empty({1, 4}, "int64");
    Array<NDArray> invalid_inputs{invalid};
    for (size_t i = 1; i < 17; ++i) invalid_inputs.push_back(Empty({1, 4, 4, 96}));
    Rejects([&] { (void)session.Run(invalid_inputs); }, "RuntimeSession");
    Check(Counts(context) == count_before_reject &&
              SameStats(stats, ci::GetPrimitiveCacheStats()),
          "out-of-range current-token input launched or compiled");
    std::cout << "actual multi-token MiniMind decode: one 782-call LLVM plan, "
              << "four B/C cases at P=4 plus 4-token prefill -> 2-token decode handoff, "
              << "max_abs_error=" << worst << "; C range [1,3] rejects C=4 before launch\n";
}

#if defined(KXC_ADAPTIVE_CUDA_TEST)
namespace adaptive_cuda {
namespace hs=api::adaptive::hot_swap;
using Request=hs::ProductionCompileRequest;
using Execution=hs::ProductionExecutionRequest;
std::filesystem::path adaptive_root;
std::vector<DeviceStream> adaptive_streams;
size_t adaptive_next_stream=0;
DeviceStream NextStream() { return adaptive_streams.at(adaptive_next_stream++%adaptive_streams.size()); }
api::CompileConfig AdaptiveConfig(int level,const std::string& name) {
    profiling::ProfileOptions options; options.enabled=options.enable_cupti=true;
    options.record_pass_ir=options.record_execution_plan_details=false;
    options.ir_capture_mode=profiling::IRCaptureMode::kDisabled; options.bundle_dir=(adaptive_root/name).string();
    return api::CompileConfig::Create(BuildTarget(Device::CUDA()),level,options);
}
bool SameCache(const ci::PrimitiveCacheStats& a,const ci::PrimitiveCacheStats& b) {
    return a.hits==b.hits&&a.misses==b.misses&&a.entries==b.entries&&a.accounted_bytes==b.accounted_bytes&&a.evictions==b.evictions&&a.in_flight==b.in_flight&&a.merged_waiters==b.merged_waiters&&a.failures==b.failures&&a.rejections==b.rejections&&a.active_pins==b.active_pins;
}
std::shared_ptr<profiling::ProfileContext> Profile(const hs::GenerationLease& lease) {
    const auto profile=lease.compiled_graph().module().As<api::CompiledModuleNode>()->profile_context_;
    Check(profile&&profile->options().enable_cupti,"adaptive CUDA candidate has no CUPTI profile"); return profile;
}
std::vector<std::string> Lines(const hs::GenerationLease& lease) {
    auto profile=Profile(lease); profile->Flush(); std::ifstream file(std::filesystem::path(profile->bundle_dir())/"events.jsonl");
    Check(file.good(),"adaptive CUDA bundle missing"); std::vector<std::string> lines; std::string line; while(std::getline(file,line)) lines.push_back(line); return lines;
}
bool Has(const std::string& line,const std::string& key,const std::string& value) { return line.find("\""+key+"\":\""+value+"\"")!=std::string::npos; }
std::string Field(const std::string& line,const std::string& key) { const auto t="\""+key+"\":\""; auto p=line.find(t); Check(p!=std::string::npos,"missing profile field"); p+=t.size(); auto e=line.find('"',p); Check(e!=std::string::npos,"unterminated profile field"); return line.substr(p,e-p); }
class Health final: public hs::HealthAuthority {
public:
    int64_t threshold=std::numeric_limits<int64_t>::max(); size_t consumed=0;
    hs::HealthDecision Evaluate(const hs::GenerationLease& lease) override {
        const auto lines=Lines(lease); size_t kernels=0; int64_t device_ns=0;
        for(const auto& line:lines) if(Has(line,"event_type","cuda_kernel")&&Has(line,"status","ok")) { ++kernels; auto p=line.find("\"duration_ns\":"); Check(p!=std::string::npos,"CUDA activity duration missing"); device_ns+=std::stoll(line.substr(p+15)); }
        Check(kernels==lease.compiled_graph().plan().calls().size()&&device_ns>0,"CUDA health saw no complete device kernels");
        Profile(lease)->WriteArtifact("health-evidence.tsv",std::to_string(lease.generation())+"\t"+std::to_string(kernels)+"\t"+std::to_string(device_ns)+"\n");
        return {lease.generation(),device_ns>threshold?hs::HealthDisposition::kQuarantine:hs::HealthDisposition::kHealthy,std::to_string(lease.generation())+":cuda",std::to_string(lease.generation())+":cuda"};
    }
    bool VerifyAndConsume(const hs::HealthDecision& decision,const hs::GenerationLease& lease) noexcept override { if(decision.generation!=lease.generation()||decision.evidence_id!=std::to_string(lease.generation())+":cuda") return false; ++consumed; return true; }
};
void RunAdaptiveState(const std::filesystem::path&) {
    const Var a("a",TensorType({257},"float32")), b("b",TensorType({257},"float32"));
    const auto add=relay::Op::Get("add"); const Function function({a,b},Call(add,{Call(add,{a,a}),Call(add,{b,b})}));
    const auto baseline=api::Compiler::Compile(function,AdaptiveConfig(3,"adaptive-reference")); Check(baseline.plan().calls().size()==3,"adaptive CUDA graph changed");
    auto health=std::make_shared<Health>(); hs::Options options; options.health_authority=health; hs::AdaptiveHotSwapController controller(options);
    const auto first=controller.CompileAndPublish({Request(function,AdaptiveConfig(3,"adaptive-first"),baseline,{1})}); const Execution execution(first->dispatch_key(),first->plan_abi());
    std::vector<float> left(257),right(257),expected(257); for(size_t i=0;i<left.size();++i){left[i]=float(i+1);right[i]=float(3*i+2);expected[i]=2*left[i]+2*right[i];}
    auto x=runtime::NDArray::Empty({257},runtime::DataTypeFromString("float32"),Device::CUDA()), y=runtime::NDArray::Empty({257},runtime::DataTypeFromString("float32"),Device::CUDA()); x.CopyFromBytes(left.data(),x.NBytes()); y.CopyFromBytes(right.data(),y.NBytes());
    std::shared_ptr<const hs::GenerationLease> second; std::vector<hs::Generation> generations;
    for(int step=0;step<3;++step){
        if(step==1){second=controller.CompileAndPublish({Request(function,AdaptiveConfig(2,"adaptive-second"),first->compiled_graph(),{1})}); Check(second->dispatch_key()==first->dispatch_key()&&second->plan_abi()==first->plan_abi()&&second->compiled_graph().artifact_pins()[1].record().artifact_key!=first->compiled_graph().artifact_pins()[1].record().artifact_key,"CUDA replacement identity changed incorrectly");}
        auto before=ci::GetPrimitiveCacheStats(); auto result=controller.RunAsync(execution,{x,y},NextStream(),{{"stage","adaptive_static"},{"model","adaptive"},{"step",std::to_string(step)}}); result.completion.Wait(); generations.push_back(result.lease->generation());
        Check(result.lease==(step==1?second:first)&&result.outputs.size()==1&&Read<float>(result.outputs[0])==expected&&SameCache(before,ci::GetPrimitiveCacheStats()),"CUDA hot swap changed output or compiled during Run");
        if(step==1){health->threshold=0; Check(controller.EvaluateHealth(second)&&!controller.EvaluateHealth(second)&&controller.Acquire(execution)==first,"CUDA hot swap rollback failed"); Profile(*second)->Flush();}
    }
    Check(generations==std::vector<hs::Generation>({1,2,1}),"CUDA generation sequence changed"); auto before=ci::GetPrimitiveCacheStats(); auto bad=runtime::NDArray::Empty({257},runtime::DataTypeFromString("float32"),Device::CPU()); Rejects([&]{(void)controller.RunAsync(execution,{bad,y},NextStream());},"device"); Check(SameCache(before,ci::GetPrimitiveCacheStats()),"wrong-device rejection touched cache"); Profile(*first)->Flush(); std::cout<<"[PASS] cuda adaptive hot swap: generations=1,2,1; three real CUDA kernels; one-shot CUPTI health rollback; zero-submit wrong-device rejection\n";
}
}
#endif

}  // namespace

int main(int argc, char** argv) {
#if defined(KXC_ADAPTIVE_CUDA_TEST)
    try { std::cout<<"[INFO] adaptive CUDA entry\n"; Check(argc==2,"provide adaptive bundle root"); if(!CollectDeviceAttributes(Device::CUDA()).exists){std::cout<<"[SKIP] no CUDA device\n";return 77;} cuda=true; adaptive_cuda::adaptive_root=argv[1]; adaptive_cuda::adaptive_streams={DeviceStream::Create(Device::CUDA()),DeviceStream::Create(Device::CUDA())}; adaptive_cuda::RunAdaptiveState({}); adaptive_cuda::adaptive_streams.clear(); return 0; } catch(const std::exception& e){std::cerr<<"[FAIL] "<<e.what()<<"\n";return 1;}
#else
    try {
        const std::string mode = argc == 2 ? argv[1] : "";
        const bool batching_only = mode == "--cuda-batching" || mode == "--batching";
        cuda = mode == "--cuda" || mode == "--cuda-batching";
        Check(argc == 1 || (argc == 2 && (cuda || batching_only)),
              "usage: minimind_bounded_decode_test [--cuda|--cuda-batching|--batching]");
        if (cuda && !CollectDeviceAttributes(TestDevice()).exists) {
            std::cout << "[SKIP] no CUDA device\n"; return 77;
        }
        for (const char* name : {"KXC_MINIMIND_BOUNDED_DECODE_DIR", "KXC_MINIMIND_BOUNDED_PREFILL_DIR"}) {
            if (batching_only && std::string(name) == "KXC_MINIMIND_BOUNDED_PREFILL_DIR") continue;
            const char* fixture = std::getenv(name);
            if ((cuda || batching_only) && (!fixture || !*fixture)) {
                std::cout << "[SKIP] set " << name << '\n'; return 77;
            }
        }
        if (cuda) cuda_streams = {DeviceStream::Create(TestDevice()),DeviceStream::Create(TestDevice())};
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true; options.ir_capture_mode = profiling::IRCaptureMode::kDisabled; options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path()/"out"/"minimind_bounded_decode_profile").string();
        options.enable_cupti = cuda;
        options = profiling::ApplyEnvironmentOverrides(options);
        const std::filesystem::path bundle_root(options.bundle_dir);
        Array<NDArray> seed;
        for (size_t stage = 0; stage < (cuda && !batching_only ? 3U : 1U); ++stage) {
            if (cuda || batching_only) options.bundle_dir = (bundle_root/(batching_only ? "batching" :
                stage == 0 ? "prefill" : stage == 1 ? "decode" : "auxiliary")).string();
            const auto context = profiling::ProfileContext::Create(options);
            const profiling::ActivationScope activation(context,"minimind_bounded_decode");
            const auto config = api::CompileConfig::Create(BuildTarget(TestDevice()),cuda ? 3 : 2,options);
            if (batching_only) TestActual(config,context,{},true);
            else if (cuda && stage == 0) seed = PrefillSeed(config,std::getenv("KXC_MINIMIND_BOUNDED_DECODE_DIR"));
            else if (cuda && stage == 1) TestActual(config,context,std::move(seed));
            else {
                TestWindow(config,context,1); TestWindow(config,context,2); TestWindow(config,context,0);
                TestStateBridge(config,context);
                if (!cuda) TestActual(config,context);
                if (!cuda) TestActualMultiToken(config,context);
            }
        }
        cuda_streams.clear();
        if (cuda) std::cout << (batching_only
            ? "[PASS] full_minimind_bounded_cuda_request_batching_no_runtime_compile\n"
            : "[PASS] full_minimind_bounded_cuda_decode_and_owned_state_no_runtime_compile\n");
        else if (batching_only) std::cout << "[PASS] full_minimind_bounded_llvm_request_batching_no_runtime_compile\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
#endif
}
