// Bounded causal attention, including the actual first MiniMind layer.
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
Expr Integer(int64_t value) {
    auto data = NDArray::Empty({1},runtime::DataTypeFromString("int64"),Device::CPU());
    data.CopyFromBytes(&value,sizeof(value)); return Constant(data);
}
NDArray Input(Array<int64_t> shape, int seed=0) {
    auto input=NDArray::Empty(shape,runtime::DataTypeFromString("float32"),Device::CPU());
    std::vector<float> values(input.NBytes()/4);
    for (size_t i=0; i<values.size(); ++i) values[i]=std::sin(double(i+seed)*0.17);
    input.CopyFromBytes(values.data(),input.NBytes()); return input;
}
std::vector<float> Read(const NDArray& array) {
    std::vector<float> values(array.NBytes()/4);
    array.CopyToBytes(values.data(),array.NBytes()); return values;
}
bool SameShape(const Array<int64_t>& a, const Array<int64_t>& b) {
    return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());
}
std::pair<size_t,size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir())/"events.jsonl");
    Check(in.good(),"missing causal attention profile");
    std::pair<size_t,size_t> counts{}; std::string line;
    while (std::getline(in,line)) {
        if (line.find("\"event_type\":\"kernel_submit\"")!=std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"")!=std::string::npos) ++counts.second;
    }
    return counts;
}
Expr Target(Expr source, int axis) {
    const auto sequence=Op("gather",{Op("shape_of",{source}),Integer(axis)},relay::GatherAttrs::Create(0));
    return Op("concatenate",{sequence,sequence},relay::ConcatenateAttrs::Create(0));
}
Expr Fill(Expr target, double value=-std::numeric_limits<double>::infinity()) {
    return Op("constant_of_shape",{target},relay::ConstantOfShapeAttrs::Create({},0,value));
}
Function Attention() {
    const TensorType type({1,2,4,6},"float32");
    const Var q("q",type),k("k",type),v("v",type);
    auto scale=NDArray::Empty({},runtime::DataTypeFromString("float32"),Device::CPU());
    const float divisor=std::sqrt(6.f); scale.CopyFromBytes(&divisor,sizeof(divisor));
    const auto kt=Op("transpose",{k},relay::TransposeAttrs::Create({0,1,3,2}));
    const auto scores=Op("divide",{Op("matmul",{q,kt}),Constant(scale)});
    const auto mask=Op("trilu",{Fill(Target(q,2))},relay::TriluAttrs::Create(1,1));
    const auto weights=Op("softmax",{Op("add",{scores,mask})},relay::SoftmaxAttrs::Create(-1));
    return Function({q,k,v},Tuple({Op("matmul",{weights,v}),mask}));
}
std::vector<restricted::InputAxisSymbol> Axes() {
    return {{0,0,"B",1,3,1},{0,2,"S",1,8,1},{1,0,"B",1,3,1},{1,2,"S",1,8,1},
            {2,0,"B",1,3,1},{2,2,"S",1,8,1}};
}
double Verify(const Array<NDArray>& inputs,const Array<NDArray>& outputs) {
    const int64_t b=inputs[0].shape()[0],s=inputs[0].shape()[2],d=6,h=2;
    Check(outputs.size()==2 && SameShape(outputs[0].shape(),{b,h,s,d}) &&
        SameShape(outputs[1].shape(),{s,s}),"causal attention output shape mismatch");
    const auto q=Read(inputs[0]),k=Read(inputs[1]),v=Read(inputs[2]),out=Read(outputs[0]),mask=Read(outputs[1]);
    for(int64_t row=0;row<s;++row) for(int64_t col=0;col<s;++col) {
        Check(mask[row*s+col]==(col>row ? -std::numeric_limits<float>::infinity() : 0.f),
              "dynamic mask must preserve negative infinity above the diagonal");
    }
    double worst=0;
    for(int64_t bh=0;bh<b*h;++bh) for(int64_t row=0;row<s;++row) {
        std::vector<double> scores(row+1);
        for(int64_t col=0;col<=row;++col) {
            double dot=0;
            for(int64_t dim=0;dim<d;++dim) dot+=double(q[(bh*s+row)*d+dim])*k[(bh*s+col)*d+dim];
            scores[col]=dot/std::sqrt(6.0);
        }
        const double maximum=*std::max_element(scores.begin(),scores.end());
        double denominator=0;
        for(double& score:scores) denominator+=(score=std::exp(score-maximum));
        for(int64_t dim=0;dim<d;++dim) {
            double reference=0;
            for(int64_t col=0;col<=row;++col) reference+=scores[col]/denominator*v[(bh*s+col)*d+dim];
            const float value=out[(bh*s+row)*d+dim];
            Check(std::isfinite(value),"causal attention output must be finite");
            worst=std::max(worst,std::abs(value-reference));
        }
    }
    Check(worst<1e-6,"causal attention differs from independent double reference"); return worst;
}
void TestSynthetic(const api::CompileConfig& config,const std::shared_ptr<profiling::ProfileContext>& context) {
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
        Adapter::Prepare(Attention(),config,Axes())));
    Check(compiled.plan().calls().size()==9,"synthetic causal plan must contain nine calls");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats(); double worst=0;
    for(const auto& [b,s]:std::vector<std::pair<int64_t,int64_t>>{{1,1},{1,4},{2,3},{3,8}}) {
        const Array<NDArray> in{Input({b,2,s,6},1),Input({b,2,s,6},5),Input({b,2,s,6},13)};
        const auto out=session.Run(in,{{"stage","causal_synthetic"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        worst=std::max(worst,Verify(in,out));
        if(s==8) {
            // Alter every future K/V element while leaving the first four positions intact.
            Array<NDArray> changed{in[0],Input({b,2,s,6},5),Input({b,2,s,6},13)};
            for(size_t argument=1;argument<3;++argument) {
                auto values=Read(changed[argument]);
                for(size_t i=0;i<values.size();++i) if((i/6)%s>=4) values[i]+=17.f;
                changed[argument].CopyFromBytes(values.data(),changed[argument].NBytes());
            }
            const auto future=session.Run(changed,{{"stage","causal_future"},{"batch","3"},{"sequence","8"},
                {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
            const auto a=Read(out[0]),z=Read(future[0]);
            bool suffix_changed=false;
            for(size_t i=0;i<a.size();++i) {
                if((i/6)%s<4) Check(a[i]==z[i],"future K/V changed a causal prefix");
                else suffix_changed=suffix_changed || std::abs(a[i]-z[i])>1e-3;
            }
            Check(suffix_changed,"causality perturbation did not change the suffix");
        }
    }
    const auto counts=Counts(context);
    for(const auto& bad:std::vector<Array<NDArray>>{
        {Input({1,2,4,6}),Input({1,2,3,6}),Input({1,2,4,6})},
        {Input({1,2,4,6}),Input({2,2,4,6}),Input({1,2,4,6})},
        {Input({1,2,4,6}),Input({1,2,4,6}),Input({1,2,5,6})},
        {Input({1,2,9,6}),Input({1,2,9,6}),Input({1,2,9,6})},
        {Input({4,2,4,6}),Input({4,2,4,6}),Input({4,2,4,6})},
        {Input({1,3,4,6}),Input({1,2,4,6}),Input({1,2,4,6})}}) {
        Rejects([&]{(void)session.Run(bad);},"RuntimeSession");
        Check(Counts(context)==counts,"invalid attention inputs allocated or launched");
    }
    Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"causal Run touched primitive cache");
    std::cout<<"causal synthetic: four B/S references and future perturbation, 45 LLVM launches, max_abs_error="<<worst<<'\n';
}
void TestFixedInferredReshape(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,2,3},"float32")),y("y",TensorType({1,4,2,5},"float32"));
    const auto shape=Op("shape_of",{x});
    const auto batch=Op("gather",{shape,Integer(0)},relay::GatherAttrs::Create(0));
    const auto sequence=Op("gather",{shape,Integer(1)},relay::GatherAttrs::Create(0));
    const auto concat=[](Expr a,Expr b){return Op("concatenate",{a,b},relay::ConcatenateAttrs::Create(0));};
    const auto prefix=concat(batch,sequence),target=concat(prefix,Integer(-1));
    const auto reshape=[&](Expr data,Expr control){return Op("reshape_dynamic",{data,control});};
    // One signed target is shared by two tensors with different fixed widths.
    // Its normalization must remain local to each consuming Reshape.
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(
        Function({x,y},Tuple({reshape(x,target),reshape(y,target)})),config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"B",1,3,1},{1,1,"S",1,8,1}})));
    Check(compiled.plan().calls().size()==4,"inferred reshapes require separate normalized controls");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto before=ci::GetPrimitiveCacheStats();
    for(const auto& [b,t]:std::vector<std::pair<int64_t,int64_t>>{{1,1},{1,4},{2,3},{3,8}}) {
        const Array<NDArray> inputs{Input({b,t,2,3},4),Input({b,t,2,5},7)};
        const auto outputs=session.Run(inputs,{{"stage","causal_fixed_reshape"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(t)},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        for(size_t i=0;i<2;++i) Check(SameShape(outputs[i].shape(),{b,t,i==0?6:10}) &&
            Read(outputs[i])==Read(inputs[i]),"fixed -1 reshape differs or shares the wrong normalized target");
    }
    const auto axes=std::vector<restricted::InputAxisSymbol>{{0,0,"B",1,3,1},{0,1,"S",1,8,1}};
    const auto prepare=[&](Expr control){return Adapter::Prepare(Function({x},reshape(x,control)),config,axes);};
    Rejects([&]{(void)prepare(concat(Integer(-1),Integer(-1)));},"at most one inferred");
    Rejects([&]{(void)prepare(Integer(-1));},"static after cancellation");
    Rejects([&]{(void)prepare(concat(concat(prefix,Integer(4)),Integer(-1)));},"positive integral quotient");
    Rejects([&]{(void)prepare(concat(prefix,Integer(-2)));},"constants must be >= -1");
    Rejects([&]{(void)Adapter::Prepare(Function({x},reshape(x,target)),config,
        {{0,0,"B",1,3,1},{0,1,"S",0,8,1}});},"positive direct symbols");
    Check(SameStats(before,ci::GetPrimitiveCacheStats()),"reshape Run/rejections touched cache");
    std::cout<<"fixed inferred reshape: shared signed target, four B/S cases, 16 LLVM launches, exact; five proof rejections\n";
}
void TestRepeatedAxes(const api::CompileConfig& config,const std::shared_ptr<profiling::ProfileContext>& context) {
    const Var x("square",TensorType({4,4},"float32"));
    auto attrs=relay::TriluAttrs::Create(1,1);
    const auto prepared=Adapter::Prepare(Function({x},Op("trilu",{x},attrs)),config,
        {{0,0,"S",1,8,1},{0,1,"S",1,8,1}});
    auto* caller=const_cast<relay::TriluAttrsNode*>(attrs.operator->());
    caller->upper=0; caller->k=-5;  // The prepared snapshot must own its attrs.
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    for(int64_t s:{1,4,8}) {
        const auto input=Input({s,s},11);
        const auto output=session.Run({input},{{"stage","causal_square"},{"sequence",std::to_string(s)},
            {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        const auto a=Read(input),z=Read(output[0]);
        Check(SameShape(output[0].shape(),{s,s}),"same-input symbol output shape mismatch");
        for(size_t i=0;i<a.size();++i) Check(z[i]==(i%s>i/s ? a[i] : 0.f),"trilu snapshot changed");
    }
    const Array<NDArray> bad{Input({4,3})};
    const auto counts=Counts(context);
    Rejects([&]{(void)session.Run(bad);},"RuntimeSession");
    Rejects([&]{(void)compiled.module().Invoke(compiled.plan().calls()[0]->symbol,bad,
        DeviceStream::Default(Device::CPU()));},"input guard rejected before allocation");
    Check(Counts(context)==counts && SameStats(stats,ci::GetPrimitiveCacheStats()),
        "same-input axis rejection allocated, launched or touched cache");
    std::cout<<"same-input S/S guards: three LLVM shapes; session and direct module reject nonsquare before allocation\n";
}
void TestRejections(const api::CompileConfig& config) {
    const Var x("x",TensorType({1,4,6},"float32"));
    const auto axes=std::vector<restricted::InputAxisSymbol>{{0,1,"S",1,8,1}};
    const auto target=Target(x,1);
    const auto before=ci::GetPrimitiveCacheStats();
    const auto prepare=[&](Expr body){return Adapter::Prepare(Function({x},body),config,axes);};
    Rejects([&]{(void)prepare(Fill(target,std::numeric_limits<double>::quiet_NaN()));},"non-NaN");
    Rejects([&]{(void)prepare(Op("constant_of_shape",{target},relay::ConstantOfShapeAttrs::Create({},3,0)));},"control fill requires int64");
    Rejects([&]{(void)prepare(Op("constant_of_shape",{target},relay::ConstantOfShapeAttrs::Create({4,4},0,0)));},"caller target");
    Rejects([&]{(void)prepare(Op("constant_of_shape",{target},relay::ConstantOfShapeAttrs::Create({},0,0,{1,1},{0,0},{1,1})));},"caller target");
    Rejects([&]{(void)prepare(Op("constant_of_shape",{x,target},relay::ConstantOfShapeAttrs::Create({},0,0)));},"arity mismatch");
    Rejects([&]{(void)Adapter::Prepare(Function({x},Fill(target)),config,{{0,1,"S",1,8193,1}});},"byte cap");
    const Var control("control",TensorType({2},"int64"));
    Rejects([&]{(void)Adapter::Prepare(Function({x,control},Fill(control)),config,axes);},"restricted shape value");
    Rejects([&]{(void)prepare(Op("trilu",{x},relay::TriluAttrs::Create(2,0)));},"upper 0 or 1");
    Rejects([&]{(void)prepare(Op("trilu",{x}));},"upper 0 or 1");
    const Var vector("vector",TensorType({4},"float32"));
    Rejects([&]{(void)Adapter::Prepare(Function({vector},Op("trilu",{vector},relay::TriluAttrs::Create(1,1))),
        config,{{0,0,"S",1,8,1}});},"rank >= 2");
    const Var integer("integer",TensorType({4,4},"int64"));
    Rejects([&]{(void)Adapter::Prepare(Function({integer},Op("trilu",{integer},relay::TriluAttrs::Create(1,1))),
        config,{{0,0,"S",1,8,1},{0,1,"S",1,8,1}});},"float32");
    Rejects([&]{(void)relay::InferTypePass(Function({x},Fill(target)));},"target must be nonempty");
    Check(SameStats(before,ci::GetPrimitiveCacheStats()),"rejected mask sources touched cache");
    std::cout<<"mask proof: 11 preparation rejections plus raw-fill InferType rejection before cache\n";
}
std::vector<float> ReadFile(const std::filesystem::path& file,size_t count) {
    std::ifstream in(file,std::ios::binary); Check(in.good(),"missing fixture: "+file.string());
    std::vector<float> values(count);
    in.read(reinterpret_cast<char*>(values.data()),count*sizeof(float));
    Check(in.gcount()==static_cast<std::streamsize>(count*sizeof(float)) && in.peek()==std::char_traits<char>::eof(),
          "attention fixture byte length mismatch");
    return values;
}
void TestActual(const api::CompileConfig& config) {
    const char* directory=std::getenv("KXC_MINIMIND_ATTENTION_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] actual RoPE: set KXC_MINIMIND_ATTENTION_DIR\n"; return; }
    const std::filesystem::path root(directory);
    const auto source=frontend::LoadONNXShapeSource((root/"attention.json").string(),(root/"attention.params").string());
    Check(source.input_names.size()==3 && source.input_names[1]=="/model/model/Slice_output_0" &&
        source.input_names[2]=="/model/model/Slice_1_output_0","actual attention input order drifted");
    const auto prepared=Adapter::Prepare(source.function,config,
        {{0,0,"B",1,3,1},{0,1,"S",1,8,1},{1,0,"S",1,8,1},{2,0,"S",1,8,1}},source.declared_output_types);
    const auto compiled=api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size()==77,"actual attention plan must have 77 calls after control folding");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats=ci::GetPrimitiveCacheStats();
    std::ifstream receipt(root/"export_receipt.txt"),cases(root/"cases.txt");
    std::string sha; receipt>>sha;
    Check(sha.size()==64 && sha.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid attention receipt");
    int64_t b=0,s=0; size_t index=0; double worst=0;
    while(cases>>b>>s) {
        Check(b>=1 && b<=3 && s>=1 && s<=8 && index<4,"invalid actual attention case");
        Array<NDArray> inputs;
        for(size_t i=0;i<3;++i) {
            auto input=Input(i==0 ? Array<int64_t>{b,s,768} : Array<int64_t>{s,96});
            const auto values=ReadFile(root/("input_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),input.NBytes()/4);
            input.CopyFromBytes(values.data(),input.NBytes()); inputs.push_back(input);
        }
        const auto out=session.Run(inputs,{{"stage","minimind_attention"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"export_receipt",sha},{"position_offset",std::to_string(index)},
            {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(out.size()==3,"actual attention output count mismatch");
        for(size_t i=0;i<3;++i) {
            Check(SameShape(out[i].shape(),i==0 ? Array<int64_t>{b,s,768} : Array<int64_t>{b,s,4,96}),"actual attention output shape mismatch");
            const auto actual=Read(out[i]);
            const auto ref=ReadFile(root/("ref_"+std::to_string(index)+"_"+std::to_string(i)+".bin"),actual.size());
            for(size_t j=0;j<actual.size();++j) {
                Check(std::isfinite(actual[j]) && std::isfinite(ref[j]),"nonfinite actual attention output");
                worst=std::max(worst,std::abs(double(actual[j])-ref[j]));
            }
        }
        if (s==8) {
            Array<NDArray> changed{Input({b,s,768}),inputs[1],inputs[2]};
            auto values=Read(inputs[0]);
            for(size_t i=0;i<values.size();++i) if((i/768)%s>=4) values[i]+=5.f;
            changed[0].CopyFromBytes(values.data(),changed[0].NBytes());
            const auto future=session.Run(changed,{{"stage","minimind_attention_future"},{"batch",std::to_string(b)},
                {"sequence",std::to_string(s)},{"export_receipt",sha},{"position_offset",std::to_string(index)},
                {"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
            bool suffix_changed=false;
            for(size_t argument=0;argument<3;++argument) {
                const auto a=Read(out[argument]),z=Read(future[argument]);
                const size_t width=argument==0 ? 768 : 384;
                for(size_t i=0;i<a.size();++i) {
                    if((i/width)%s<4) Check(a[i]==z[i],"future activation changed an actual causal prefix");
                    else if(argument==0) suffix_changed=suffix_changed || std::abs(a[i]-z[i])>1e-3;
                }
            }
            Check(suffix_changed,"actual activation perturbation did not change the suffix");
            std::cout<<"actual causal prefix invariant under future activation perturbation: 77 additional LLVM launches\n";
        }
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"actual attention Run touched cache"); ++index;
    }
    Check(index==4 && cases.eof() && worst<3e-5,"actual attention differs from dynamic ONNX reference");
    std::cout << "actual MiniMind RMSNorm/QKV/RoPE/GQA/causal attention/output projection: four B/S cases, 308 LLVM launches, max_abs_error=" << worst << '\n';
}
}  // namespace
int main() {
    try {
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled=true; options.ir_capture_mode=profiling::IRCaptureMode::kDisabled; options.record_pass_ir=false;
        options.bundle_dir=(std::filesystem::current_path()/"out"/"bounded_causal_attention_profile").string();
        const auto context=profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context,"bounded_causal_attention");
        const auto config=api::CompileConfig::Create(BuildTarget(Device::CPU()),2,options);
        TestRejections(config); TestSynthetic(config,context); TestFixedInferredReshape(config); TestRepeatedAxes(config,context); TestActual(config);
        return 0;
    } catch(const std::exception& error) { std::cerr<<"[FAIL] "<<error.what()<<'\n'; return 1; }
}
