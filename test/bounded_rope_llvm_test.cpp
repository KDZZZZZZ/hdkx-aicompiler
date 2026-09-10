// Real MiniMind RMSNorm/QKV/head/RoPE chain and an independent rotary reference.
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
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
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
Expr Half(Expr x) {
    auto extent = Op("gather", {Op("shape_of", {x}), Integer(-1, true)}, relay::GatherAttrs::Create(0));
    auto half = Op("divide", {extent, Integer(2, true)});
    half = Op("cast", {Op("cast", {half}, relay::CastAttrs::Create(2))}, relay::CastAttrs::Create(2));
    return Op("unsqueeze", {half}, relay::UnsqueezeAttrs::Create({0}));
}
Expr Slice(Expr x, Expr start, Expr end, int64_t axis = -1, int64_t step = 1) {
    return Op("slice", {x, start, end, Integer(axis), Integer(step)});
}
Expr Rotate(Expr x, Expr cos, Expr sin) {
    const auto half = Half(x);
    const auto tail = Slice(x, half, Integer(std::numeric_limits<int64_t>::max()));
    const auto head = Slice(x, Integer(0), half);
    const auto rotated = Op("concatenate", {Op("neg", {tail}), head}, relay::ConcatenateAttrs::Create(-1));
    return Op("cast", {Op("add", {Op("mul", {x, cos}), Op("mul", {rotated, sin})})},
              relay::CastAttrs::Create(0));
}
NDArray Input(Array<int64_t> shape, int seed = 0) {
    auto result = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), Device::CPU());
    std::vector<float> values(result.NBytes() / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(std::sin(static_cast<double>(i + seed) * .137));
    }
    if (!values.empty()) result.CopyFromBytes(values.data(), result.NBytes());
    return result;
}
std::vector<float> Read(const NDArray& data) {
    std::vector<float> values(data.NBytes() / sizeof(float));
    if (!values.empty()) data.CopyToBytes(values.data(), data.NBytes());
    return values;
}
bool SameShape(const Array<int64_t>& a, const Array<int64_t>& b) {
    return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());
}
double CheckRotated(const NDArray& input, const NDArray& cos, const NDArray& sin, const NDArray& output) {
    Check(SameShape(input.shape(),output.shape()), "rotary output shape mismatch");
    Check(input->dl_tensor.data != output->dl_tensor.data, "rotary output must be fresh");
    const auto x = Read(input), c = Read(cos), s = Read(sin), y = Read(output);
    const auto shape = input.shape();
    const size_t width = shape[3], heads = shape[2], sequence = shape[1];
    double worst = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const size_t dim = i % width, row = i / width, pos = (row / heads) % sequence;
        const size_t other = row * width + (dim + width / 2) % width;
        const double sign = dim < width / 2 ? -1.0 : 1.0;
        const double reference = double(x[i]) * c[pos * width + dim] + sign * x[other] * s[pos * width + dim];
        Check(std::isfinite(y[i]), "rotary output is nonfinite");
        worst = std::max(worst, std::abs(y[i] - reference));
    }
    Check(worst < 3e-7, "rotary output differs from independent double reference");
    return worst;
}
std::pair<size_t,size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(in.good(), "missing rotary profile");
    std::pair<size_t,size_t> counts{};
    std::string line;
    while (std::getline(in,line)) {
        if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++counts.second;
    }
    return counts;
}
void TestSynthetic(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context) {
    const Var q("q",TensorType({1,4,4,6},"float32")), k("k",TensorType({1,4,2,6},"float32"));
    const Var c("c",TensorType({4,6},"float32")), s("s",TensorType({4,6},"float32"));
    const auto cu = Op("unsqueeze", {c}, relay::UnsqueezeAttrs::Create({1}));
    const auto su = Op("unsqueeze", {s}, relay::UnsqueezeAttrs::Create({1}));
    const auto prepared = Adapter::Prepare(Function({q,k,c,s}, Tuple({Rotate(q,cu,su), Rotate(k,cu,su)})), config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"B",1,3,1},{1,1,"S",1,8,1},
         {2,0,"S",1,8,1},{3,0,"S",1,8,1}});
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size() == 18 && compiled.plan().constant_value_ids().empty(),
          "RoPE must fold head-dimension controls and leave 18 data calls, no constant pool");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto abi = api::BuildPlanAbiFingerprint(compiled).digest();
    double worst = 0;
    for (const auto& [b,n] : std::vector<std::pair<int64_t,int64_t>>{{1,1},{1,4},{2,3},{3,8}}) {
        const Array<NDArray> inputs{Input({b,n,4,6}),Input({b,n,2,6},17),Input({n,6},51),Input({n,6},79)};
        const auto out = session.Run(inputs, {{"stage","rope_synthetic"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(n)},{"plan_abi",abi}});
        Check(out.size()==2,"rotary output count mismatch");
        for (size_t i=0;i<2;++i) worst=std::max(worst,CheckRotated(inputs[i],inputs[2],inputs[3],out[i]));
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"rotary Run touched cache");
    }
    const auto counts = Counts(context);
    Check(counts.first == 72, "four rotary cases must launch 72 kernels");
    const Array<NDArray> valid{Input({1,4,4,6}),Input({1,4,2,6}),Input({4,6}),Input({4,6})};
    for (const auto& [argument, shape] : std::vector<std::pair<size_t,Array<int64_t>>>{
        {0,{0,4,4,6}},{0,{4,4,4,6}},{0,{1,0,4,6}},{0,{1,9,4,6}},
        {0,{1,4,3,6}},{0,{1,4,4,5}},{1,{2,4,2,6}},{1,{1,3,2,6}},
        {2,{3,6}},{3,{4,5}}}) {
        Array<NDArray> inputs;
        for (const auto& value : valid) inputs.push_back(value);
        inputs[argument]=Input(shape);
        Rejects([&]{(void)session.Run(inputs);}, "");
        Check(Counts(context)==counts && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "invalid rotary input allocated, launched, or touched cache");
    }
    std::cout << "synthetic Q/K RoPE: four B/S cases, 72 LLVM launches, max_abs_error=" << worst
              << "; 10 zero-allocation/launch rejections\n";
}
void TestRejections(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,2,6},"float32"));
    const auto before=ci::GetPrimitiveCacheStats();
    const auto reject=[&](Expr body,const std::string& why) {
        Rejects([&]{(void)Adapter::Prepare(Function({x},body),config,
            {{0,0,"B",1,3,1},{0,1,"S",1,8,1}});},why);
        Check(SameStats(before,ci::GetPrimitiveCacheStats()),"invalid rotary preparation touched cache");
    };
    reject(Slice(x,Integer(0),Integer(3),-1,0),"step");
    reject(Slice(x,Integer(0),Integer(3),-1,-1),"step");
    reject(Slice(x,Integer(0),Integer(3),4),"axis");
    reject(Slice(x,Integer(0),Integer(3),1),"static sliced axes");
    reject(Slice(x,Op("gather",{Op("shape_of",{x}),Integer(1)},relay::GatherAttrs::Create(0)),Integer(3)),
           "provably constant");
    reject(Slice(x,Integer(0,true),Integer(3)),"nonempty vectors");
    reject(Op("slice",{x,Integer(0)}),"arity");
    reject(Op("slice",{x,Integer(0),Integer(3),Integer(-1),Integer(1)},
              relay::SliceAttrs::Create({0},{3},{-1},{1})),"without attrs");
    reject(Op("cast",{Half(x)},relay::CastAttrs::Create(1)),"preserve int64");
    reject(Op("concatenate",{x,x},relay::ConcatenateAttrs::Create(1)),"static concatenation axis");
    const Var y("y",TensorType({1,4,2,6},"float32"));
    Rejects([&]{(void)Adapter::Prepare(Function({x,y},Op("concatenate",{x,y},relay::ConcatenateAttrs::Create(-1))),config,
        {{0,1,"S",1,8,1},{1,1,"T",1,8,1}});},"identical expressions");
    const Var controls("controls",TensorType({1},"int64"));
    Rejects([&]{(void)Adapter::Prepare(Function({x,controls},Slice(x,controls,Integer(3))),config,
        {{0,1,"S",1,8,1}});},"proved int64");
    Rejects([&]{(void)relay::InferTypePass(Function({x},Slice(x,Integer(0),Integer(3))));},"restricted preparation");
    Check(SameStats(before,ci::GetPrimitiveCacheStats()),"negative controls touched cache");
    std::cout << "RoPE proof: 12 preparation rejections and unresolved Slice InferType rejection before cache\n";
}
void TestConstantInputs(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,2},"float32")), y("y",TensorType({1,4,2},"float32")),
              z("z",TensorType({1,4,2},"float32"));
    auto constant=NDArray::Empty({},runtime::DataTypeFromString("float32"),Device::CPU());
    const float value=2; constant.CopyFromBytes(&value,sizeof(value));
    Expr body=x;
    for (int i=0;i<7;++i) body=Op("nn_relu",{body});
    body=Op("add",{Op("add",{Op("add",{body,Constant(constant)}),y}),z});
    const auto prepared=Adapter::Prepare(Function({x,y,z},body),config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"B",1,3,1},{1,1,"S",1,8,1},
         {2,0,"B",1,3,1},{2,1,"S",1,8,1}});
    const auto& inputs=prepared.graph_template().shape_program().inputs();
    Check(inputs.size()==4 && inputs[2].name=="value.10" && inputs[3].name=="value.2",
          "regression must interleave a lexical constant name before parameter value.2");
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size()==10 && compiled.plan().constant_value_ids().size()==1,
          "three-parameter regression must retain one immutable constant");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    for (const auto& [b,s] : std::vector<std::pair<int64_t,int64_t>>{{1,1},{3,8}}) {
        const Array<NDArray> in{Input({b,s,2}),Input({b,s,2},17),Input({b,s,2},31)};
        const auto out=session.Run(in,{{"stage","rope_parameter_constants"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        const auto a=Read(in[0]),c=Read(in[1]),d=Read(in[2]),result=Read(out[0]);
        Check(SameShape(out[0].shape(),{b,s,2}),"parameter/constant output shape mismatch");
        for(size_t i=0;i<result.size();++i) {
            Check(result[i]==((std::max(a[i],0.f)+value)+c[i])+d[i],"parameter/constant value mismatch");
        }
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"parameter/constant Run touched cache");
    }
    std::cout << "lexically interleaved parameter/constant contracts: two shapes, 20 LLVM launches, exact results\n";
}
void TestStaticSliceSnapshot(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,2,6},"float32"));
    auto attrs=relay::SliceAttrs::Create({-9,1},{std::numeric_limits<int64_t>::max(),99},{-1,-2},{1,1});
    const auto prepared=Adapter::Prepare(Function({x},Op("slice",{x},attrs)),config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1}});
    auto* caller=const_cast<relay::SliceAttrsNode*>(attrs.operator->());
    caller->starts[0]=5; caller->ends[0]=6; caller->axes[0]=0; caller->steps[0]=-1;
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size()==1,"static Slice must remain one data call");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    for (const auto& [b,s] : std::vector<std::pair<int64_t,int64_t>>{{1,1},{3,8}}) {
        const auto in=Input({b,s,2,6});
        const auto out=session.Run({in},{{"stage","rope_slice_snapshot"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(SameShape(out[0].shape(),{b,s,1,6}),"Slice snapshot output shape mismatch");
        const auto a=Read(in), result=Read(out[0]);
        for(size_t i=0;i<result.size();++i) Check(result[i]==a[(i/6)*12+6+i%6],"Slice snapshot/index mismatch");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"Slice Run touched cache");
    }
    std::cout << "Slice deep snapshot and signed endpoint clamping: two shapes, two LLVM launches, exact results\n";
}
std::vector<float> ReadFile(const std::filesystem::path& file,size_t count) {
    std::ifstream in(file,std::ios::binary); Check(in.good(),"missing fixture: "+file.string());
    std::vector<float> values(count);
    in.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    Check(in.gcount()==static_cast<std::streamsize>(count*sizeof(float)) && in.peek()==std::char_traits<char>::eof(),
          "rotary fixture byte length mismatch");
    return values;
}
void TestActual(const api::CompileConfig& config) {
    const char* directory=std::getenv("KXC_MINIMIND_ROPE_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] actual RoPE: set KXC_MINIMIND_ROPE_DIR\n"; return; }
    const std::filesystem::path root(directory);
    const auto source=frontend::LoadONNXShapeSource((root/"rope.json").string(),(root/"rope.params").string());
    Check(source.input_names.size()==3 && source.input_names[1]=="/model/model/Slice_output_0" &&
        source.input_names[2]=="/model/model/Slice_1_output_0","actual rotary input order drifted");
    const auto prepared=Adapter::Prepare(source.function,config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"S",1,8,1},{2,0,"S",1,8,1}},source.declared_output_types);
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size()==51,"actual rotary plan must have 51 calls after control folding");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    std::ifstream receipt(root/"export_receipt.txt"),cases(root/"cases.txt");
    std::string sha; receipt>>sha;
    Check(sha.size()==64 && sha.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid rotary receipt");
    int64_t b=0,s=0; size_t index=0; double worst=0;
    while(cases>>b>>s) {
        Check(b>=1 && b<=3 && s>=1 && s<=8 && index<4,"invalid actual rotary case");
        Array<NDArray> inputs;
        for(size_t i=0;i<3;++i) {
            auto input=Input(i==0 ? Array<int64_t>{b,s,768} : Array<int64_t>{s,96});
            const auto values=ReadFile(root/("input_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),input.NBytes()/4);
            input.CopyFromBytes(values.data(),input.NBytes()); inputs.push_back(input);
        }
        const auto out=session.Run(inputs,{{"stage","minimind_rope"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"export_receipt",sha},{"position_offset",std::to_string(index)},
            {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(out.size()==3,"actual rotary output count mismatch");
        for(size_t i=0;i<3;++i) {
            Check(SameShape(out[i].shape(),{b,s,i==0?8:4,96}),"actual rotary output shape mismatch");
            const auto actual=Read(out[i]);
            const auto ref=ReadFile(root/("ref_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),actual.size());
            for(size_t j=0;j<actual.size();++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(ref[j]),"nonfinite actual rotary output");
                worst=std::max(worst,std::abs(double(actual[j])-ref[j]));
            }
        }
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"actual rotary Run touched cache"); ++index;
    }
    Check(index==4 && cases.eof() && worst<2e-5,"actual rotary differs from dynamic ONNX reference");
    std::cout << "actual MiniMind RMSNorm/QKV/heads/RoPE: four B/S cases, 204 LLVM launches, max_abs_error=" << worst << '\n';
}
}  // namespace
int main() {
    try {
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled=true; options.ir_capture_mode=profiling::IRCaptureMode::kDisabled; options.record_pass_ir=false;
        options.bundle_dir=(std::filesystem::current_path()/"out"/"bounded_rope_profile").string();
        const auto context=profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context,"bounded_rope");
        const auto config=api::CompileConfig::Create(BuildTarget(Device::CPU()),2,options);
        TestRejections(config); TestSynthetic(config,context);
        TestConstantInputs(config); TestStaticSliceSnapshot(config); TestActual(config);
        return 0;
    } catch(const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
