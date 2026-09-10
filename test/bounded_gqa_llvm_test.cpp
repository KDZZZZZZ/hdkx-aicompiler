// Actual ONNX GQA controls and a download-free production proof/numeric fixture.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;
void Check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
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
Expr Op(const std::string& name, Array<Expr> args, relay::Attrs attrs = {}) {
    return Call(relay::Op::Get(name), std::move(args), std::move(attrs));
}
Expr Integer(int64_t value, bool scalar = false) {
    auto data = NDArray::Empty(scalar ? Array<int64_t>{} : Array<int64_t>{1},
                               runtime::DataTypeFromString("int64"), Device::CPU());
    data.CopyFromBytes(&value, sizeof(value));
    return Constant(std::move(data));
}
Expr Vectorize(Expr scalar) { return Op("unsqueeze", {scalar}, relay::UnsqueezeAttrs::Create({0})); }
Expr Concat(Array<Expr> vectors) {
    Expr result = vectors[0];
    for (size_t i = 1; i < vectors.size(); ++i) {
        result = Op("concatenate", {result, vectors[i]}, relay::ConcatenateAttrs::Create(0));
    }
    return result;
}
Expr Repeat(Expr data, bool true_branch = false, int64_t sentinel = -1) {
    const Expr shape = Op("shape_of", {data});
    Array<Expr> axes;
    for (int64_t axis = 0; axis < 4; ++axis) {
        axes.push_back(Op("gather", {shape, Integer(axis, true)}, relay::GatherAttrs::Create(0)));
    }
    const Expr target = Concat({Vectorize(axes[0]), Vectorize(axes[1]), Vectorize(axes[2]),
                               Integer(2), Vectorize(axes[3])});
    const Expr flat = Op("reshape_dynamic", {target, Integer(-1)});
    const Expr ones = Op("constant_of_shape", {Op("shape_of", {flat})},
                         relay::ConstantOfShapeAttrs::Create({}, 2, 1));
    const Expr marker = Op("mul", {ones, Integer(sentinel, true)});
    const Expr condition = Op("equal", {flat, true_branch ? flat : marker});
    const Expr selected = Op("where", {condition, true_branch ? flat : ones, true_branch ? ones : flat});
    const Expr expanded = Op("expand_dynamic", {
        Op("unsqueeze", {data}, relay::UnsqueezeAttrs::Create({3})), selected});
    const Expr head_count = true_branch
        ? Op("divide", {Op("add", {axes[2], axes[2]}), Integer(1, true)})
        : Op("mul", {axes[2], Integer(2, true)});
    const Expr heads = Vectorize(head_count);
    return Op("reshape_dynamic", {expanded,
        Concat({Vectorize(axes[0]), Vectorize(axes[1]), heads, Vectorize(axes[3])})});
}
std::vector<restricted::InputAxisSymbol> Axes() {
    return {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"B",1,3,1},{1,1,"S",1,8,1}};
}
NDArray Input(int64_t b, int64_t s, int64_t h, int64_t d, int seed = 0) {
    auto result = NDArray::Empty({b,s,h,d}, runtime::DataTypeFromString("float32"), Device::CPU());
    std::vector<float> values(result.NBytes() / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i + seed) * 0.125f;
    if (!values.empty()) result.CopyFromBytes(values.data(), result.NBytes());
    return result;
}
std::vector<float> Read(const NDArray& data) {
    std::vector<float> values(data.NBytes() / sizeof(float));
    if (!values.empty()) data.CopyToBytes(values.data(), data.NBytes());
    return values;
}
void CheckRepeated(const NDArray& input, const NDArray& output) {
    const auto in_shape = input.shape(), out_shape = output.shape();
    Check(out_shape.size() == 4 && out_shape[0] == in_shape[0] && out_shape[1] == in_shape[1] &&
        out_shape[2] == in_shape[2] * 2 && out_shape[3] == in_shape[3], "GQA output shape mismatch");
    Check(input->dl_tensor.data != output->dl_tensor.data, "GQA result must be fresh");
    const auto in = Read(input), out = Read(output);
    const size_t width = static_cast<size_t>(in_shape[2] * in_shape[3]);
    for (size_t i = 0; i < out.size(); ++i) {
        const size_t row = i / (2 * width), column = i % (2 * width);
        const size_t source = row * width + column / (2 * in_shape[3]) * in_shape[3] + column % in_shape[3];
        Check(out[i] == in[source], "GQA head repetition changed data or ordering");
    }
}
std::pair<size_t,size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(in.good(), "missing GQA profile");
    std::pair<size_t,size_t> counts{};
    std::string line;
    while (std::getline(in,line)) {
        if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++counts.second;
    }
    return counts;
}
void TestSynthetic(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context) {
    const Var k("k", TensorType({1,4,2,3}, "float32")), v("v", TensorType({1,4,2,3}, "float32"));
    const auto compile = [&](bool truth) {
        const auto prepared = Adapter::Prepare(Function({k,v}, Tuple({Repeat(k,truth), Repeat(v,truth)})), config, Axes());
        return api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    };
    const auto compiled = compile(false);
    Check(compiled.plan().calls().size() == 10 && compiled.plan().constant_value_ids().empty(),
          "GQA controls must leave ten data/shape calls and no constant pool");
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto abi = api::BuildPlanAbiFingerprint(compiled);
    for (const auto& [b,s] : std::vector<std::pair<int64_t,int64_t>>{{1,1},{1,4},{2,3},{3,8}}) {
        const Array<NDArray> inputs{Input(b,s,2,3), Input(b,s,2,3,37)};
        const auto outputs = session.Run(inputs, {{"stage","gqa_synthetic"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"plan_abi",abi.digest()}});
        Check(outputs.size() == 2, "GQA result count mismatch");
        for (size_t i = 0; i < 2; ++i) CheckRepeated(inputs[i], outputs[i]);
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()), "GQA Run touched primitive cache");
    }
    const auto counts = Counts(context);
    Check(counts.first == 40, "four synthetic GQA shapes must launch forty kernels");
    const auto invalid = [&](NDArray a, NDArray b) {
        Rejects([&]{ (void)session.Run({a,b}); }, "");
        Check(Counts(context) == counts, "invalid GQA input allocated or launched");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()), "invalid GQA input compiled");
    };
    invalid(Input(0,1,2,3),Input(0,1,2,3));
    invalid(Input(4,1,2,3),Input(4,1,2,3));
    invalid(Input(1,0,2,3),Input(1,0,2,3));
    invalid(Input(1,9,2,3),Input(1,9,2,3));
    invalid(Input(1,1,3,3),Input(1,1,2,3));
    invalid(Input(1,1,2,4),Input(1,1,2,3));
    invalid(Input(1,1,2,3),Input(2,1,2,3));
    invalid(Input(1,1,2,3),Input(1,2,2,3));
    const auto true_compiled = compile(true);
    const auto true_stats = ci::GetPrimitiveCacheStats();
    const runtime::RuntimeSession true_session(true_compiled.module(),true_compiled.plan());
    const Array<NDArray> inputs{Input(2,3,2,3),Input(2,3,2,3,37)};
    const auto outputs = true_session.Run(inputs, {{"stage","gqa_true_branch"},{"batch","2"},{"sequence","3"},
        {"plan_abi",api::BuildPlanAbiFingerprint(true_compiled).digest()}});
    for (size_t i=0;i<2;++i) CheckRepeated(inputs[i],outputs[i]);
    Check(SameStats(true_stats,ci::GetPrimitiveCacheStats()), "proved true branch compiled at Run");
    std::cout << "synthetic GQA: four shapes, 40 LLVM launches, exact repeated heads; "
                 "8 zero-allocation/launch rejections; proved true Where branch also executed\n";
}
void TestRejections(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,2,3},"float32"));
    const std::vector<restricted::InputAxisSymbol> axes{{0,0,"B",1,3,1},{0,1,"S",1,8,1}};
    const auto before = ci::GetPrimitiveCacheStats();
    const auto reject = [&](Expr body,const std::string& why) {
        Rejects([&]{ (void)Adapter::Prepare(Function({x},body),config,axes); },why);
        Check(SameStats(before,ci::GetPrimitiveCacheStats()),"GQA preparation rejection touched cache");
    };
    reject(Repeat(x,false,2),"equality cannot be proved");
    const Expr shape=Op("shape_of",{x});
    const Expr batch=Op("gather",{shape,Integer(0,true)},relay::GatherAttrs::Create(0));
    const Expr heads=Op("gather",{shape,Integer(2,true)},relay::GatherAttrs::Create(0));
    reject(Vectorize(Op("mul",{batch,Integer(2,true)})),"statically proved integer");
    reject(Vectorize(Op("mul",{heads,Integer(std::numeric_limits<int64_t>::max(),true)})),"overflow");
    reject(Vectorize(Op("mul",{heads,Integer(std::numeric_limits<int64_t>::min(),true)})),"magnitude");
    reject(Vectorize(Op("divide",{heads,Integer(0,true)})),"positive integer");
    reject(Op("reshape_dynamic",{shape,Integer(-2)}),"known element count");
    reject(Op("constant_of_shape",{Vectorize(batch)},relay::ConstantOfShapeAttrs::Create({},2,1)),"statically proved");
    reject(Op("constant_of_shape",{Integer(17)},relay::ConstantOfShapeAttrs::Create({},2,1)),"length cap");
    reject(Op("constant_of_shape",{Integer(5)},relay::ConstantOfShapeAttrs::Create({5},2,1)),"caller target");
    reject(Op("constant_of_shape",{Integer(5)},relay::ConstantOfShapeAttrs::Create({},3,1)),"requires int64");
    const Var condition("condition",TensorType({4},"bool"));
    Rejects([&]{ (void)Adapter::Prepare(Function({x,condition},Op("where",{condition,shape,shape})),config,axes); },
            "statically proved scalar/vector bool");
    Check(SameStats(before,ci::GetPrimitiveCacheStats()),"unknown predicate touched cache");
    std::cout << "GQA proof: 11 invalid/unknown controls rejected before compile/cache\n";
}
void TestAxisRoundTrip(const api::CompileConfig& config) {
    const Var x("x", TensorType({1,4,2,3}, "float32"));
    const Expr expanded = Op("unsqueeze", {x}, relay::UnsqueezeAttrs::Create({0,-2}));
    const Expr restored = Op("squeeze", {expanded}, relay::SqueezeAttrs::Create({0,4}));
    const auto prepared = Adapter::Prepare(Function({x},restored), config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1}});
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size() == 2, "axis edits must lower through their existing callbacks");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    for (const auto& [b,s] : std::vector<std::pair<int64_t,int64_t>>{{1,1},{3,8}}) {
        const auto input = Input(b,s,2,3);
        const auto output = session.Run({input}, {{"stage","gqa_axis_roundtrip"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(output.size() == 1 && Read(output[0]) == Read(input), "bounded axis roundtrip data mismatch");
        const auto shape = output[0].shape();
        Check(shape.size() == 4 && shape[0] == b && shape[1] == s && shape[2] == 2 && shape[3] == 3,
              "axis roundtrip froze B/S");
        Check(output[0]->dl_tensor.data != input->dl_tensor.data, "axis roundtrip must return fresh data");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()), "axis roundtrip Run touched cache");
    }
    std::cout << "bounded Unsqueeze/Squeeze: two shapes, four LLVM launches, exact roundtrip\n";
}
std::vector<float> ReadFile(const std::filesystem::path& path,size_t count) {
    std::ifstream in(path,std::ios::binary);
    Check(in.good(),"missing GQA fixture "+path.string());
    std::vector<float> values(count);
    in.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    Check(in.gcount()==static_cast<std::streamsize>(count*sizeof(float)) && in.peek()==std::char_traits<char>::eof(),
          "GQA fixture byte length mismatch");
    return values;
}
void TestActual(const api::CompileConfig& config) {
    const char* directory=std::getenv("KXC_MINIMIND_GQA_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] actual GQA: set KXC_MINIMIND_GQA_DIR\n"; return; }
    const std::filesystem::path root(directory);
    const auto source=frontend::LoadONNXShapeSource((root/"gqa.json").string(),(root/"gqa.params").string());
    Check(source.input_names==std::vector<std::string>{"present_k_0","present_v_0"},"actual GQA input order drifted");
    const auto prepared=Adapter::Prepare(source.function,config,Axes(),source.declared_output_types);
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size()==10 && compiled.plan().constant_value_ids().empty(),
          "actual GQA did not fold its shape controls and constants");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    std::ifstream receipt(root/"export_receipt.txt"),cases(root/"cases.txt");
    std::string sha; receipt>>sha;
    Check(sha.size()==64 && sha.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid GQA receipt");
    int64_t b=0,s=0; size_t index=0;
    while(cases>>b>>s) {
        Check(b>=1 && b<=3 && s>=1 && s<=8 && index<4,"invalid actual GQA case");
        Array<NDArray> inputs;
        for(size_t i=0;i<2;++i) {
            auto input=Input(b,s,4,96);
            const auto values=ReadFile(root/("input_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),b*s*4*96);
            input.CopyFromBytes(values.data(),input.NBytes()); inputs.push_back(input);
        }
        const auto output=session.Run(inputs,{{"stage","minimind_gqa"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"export_receipt",sha},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(output.size()==2,"actual GQA output count mismatch");
        for(size_t i=0;i<2;++i) {
            CheckRepeated(inputs[i],output[i]);
            Check(Read(output[i])==ReadFile(root/("ref_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),b*s*8*96),
                  "actual GQA differs from dynamic ONNX reference");
        }
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"actual GQA Run touched cache"); ++index;
    }
    Check(index==4 && cases.eof(),"actual GQA requires four cases");
    std::cout << "actual MiniMind K/V GQA: four B/S cases, 40 LLVM launches, max_abs_error=0\n";
}
}  // namespace
int main() {
    try {
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled=true; options.ir_capture_mode=profiling::IRCaptureMode::kDisabled; options.record_pass_ir=false;
        options.bundle_dir=(std::filesystem::current_path()/"out"/"bounded_gqa_profile").string();
        const auto context=profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context,"bounded_gqa");
        const auto config=api::CompileConfig::Create(BuildTarget(Device::CPU()),2,options);
        TestRejections(config); TestSynthetic(config,context); TestAxisRoundTrip(config); TestActual(config);
        return 0;
    } catch(const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
