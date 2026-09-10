// Explicit all-masked semantics through production Relay -> TE -> LLVM.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "support/primitive_lowering.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/ffi/registry.h"
#include "kxc/runtime/session.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

namespace {
using namespace kxc;
using runtime::NDArray;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
void Check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void Rejects(F action, const std::string& expected) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(expected) != std::string::npos,"unexpected rejection: "+std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: "+expected);
}
Expr Op(const std::string& name,Array<Expr> inputs,relay::Attrs attrs = {}) {
    return Call(relay::Op::Get(name),std::move(inputs),std::move(attrs));
}
size_t Size(const Array<int64_t>& shape) {
    size_t size = 1; for (int64_t dim : shape) size *= static_cast<size_t>(dim); return size;
}
template<class T> NDArray Tensor(const Array<int64_t>& shape,const std::vector<T>& data,const std::string& dtype) {
    auto value = NDArray::Empty(shape,runtime::DataTypeFromString(dtype),Device::CPU());
    Check(value.NBytes() == data.size()*sizeof(T),"tensor data size mismatch");
    if (!data.empty()) value.CopyFromBytes(data.data(),value.NBytes());
    return value;
}
template<class T> std::vector<T> Read(const NDArray& value) {
    std::vector<T> result(value.NBytes()/sizeof(T));
    if (!result.empty()) value.CopyToBytes(result.data(),value.NBytes());
    return result;
}
size_t MaskIndex(size_t index,const Array<int64_t>& shape,const Array<int64_t>& mask_shape) {
    size_t offset = 0, stride = 1;
    for (size_t reverse = 0; reverse < shape.size(); ++reverse) {
        const size_t axis = shape.size()-1-reverse;
        const size_t coordinate = index % static_cast<size_t>(shape[axis]);
        index /= static_cast<size_t>(shape[axis]);
        if (reverse < mask_shape.size()) {
            const int64_t dim = mask_shape[mask_shape.size()-1-reverse];
            offset += (dim == 1 ? 0 : coordinate)*stride; stride *= static_cast<size_t>(dim);
        }
    }
    return offset;
}
template<class T> std::vector<double> Reference(const std::vector<T>& logits,
    const std::vector<uint8_t>& mask,const Array<int64_t>& shape,const Array<int64_t>& mask_shape,int axis) {
    if (axis < 0) axis += static_cast<int>(shape.size());
    size_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= static_cast<size_t>(shape[i]);
    for (size_t i = axis+1; i < shape.size(); ++i) inner *= static_cast<size_t>(shape[i]);
    const size_t count = shape[axis];
    std::vector<double> result(logits.size(),0);
    for (size_t row = 0; row < outer; ++row) for (size_t lane = 0; lane < inner; ++lane) {
        double maximum = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (size_t j = 0; j < count; ++j) {
            const size_t index = (row*count+j)*inner+lane;
            if (mask[MaskIndex(index,shape,mask_shape)]) { maximum = std::max(maximum,double(logits[index])); any = true; }
        }
        if (!any) continue;
        double sum = 0;
        for (size_t j = 0; j < count; ++j) {
            const size_t index = (row*count+j)*inner+lane;
            if (mask[MaskIndex(index,shape,mask_shape)]) { result[index] = std::exp(double(logits[index])-maximum); sum += result[index]; }
        }
        for (size_t j = 0; j < count; ++j) result[(row*count+j)*inner+lane] /= sum;
    }
    return result;
}
template<class T> void Close(const NDArray& output,const std::vector<double>& expected,double tolerance) {
    const auto actual = Read<T>(output);
    Check(actual.size() == expected.size(),"output element count mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        Check(std::isfinite(actual[i]) && std::abs(double(actual[i])-expected[i]) <= tolerance,"masked numeric mismatch at "+std::to_string(i));
        if (expected[i] == 0) Check(actual[i] == T(0) && !std::signbit(actual[i]),"masked element must be exact positive zero");
    }
}
bool SameStats(const ci::PrimitiveCacheStats& a,const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions && a.in_flight == b.in_flight &&
        a.merged_waiters == b.merged_waiters && a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
size_t Submits(const std::shared_ptr<profiling::ProfileContext>& profile) {
    profile->Flush(); size_t count = 0;
    std::ifstream input(std::filesystem::path(profile->bundle_dir())/"events.jsonl");
    Check(input.good(),"missing masked softmax bundle"); std::string line;
    while (std::getline(input,line)) count += line.find("\"event_type\":\"kernel_submit\"") != std::string::npos;
    return count;
}
Function FunctionFor(const Array<int64_t>& shape,const Array<int64_t>& mask_shape,const std::string& dtype,int axis) {
    const Var data("logits",TensorType(shape,dtype)), mask("mask",TensorType(mask_shape,"bool"));
    return Function({data,mask},Op("masked_softmax",{data,mask},relay::SoftmaxAttrs::Create(axis)));
}
void TestContract() {
    const Var data("logits",TensorType({2,3},"float32")), mask("mask",TensorType({3},"bool"));
    const auto maker = Registry::Global().Get("kxc.relay.op._make.masked_softmax");
    Check(maker.defined(),"masked_softmax generated registration/FFI missing");
    const Call call = CastTo<Call>(maker(data,mask,-1));
    Check(call->args.size() == 2 && call->attrs.As<relay::SoftmaxAttrsNode>()->axis == -1,"masked FFI boundary drift");
    const auto lowered = test_support::LowerFirstPrimitive(Function({data,mask},call));
    Check(lowered->prim_func.defined(),"masked_softmax production primitive lowering failed");
    auto* cuda = new TargetNode();
    cuda->kind = "cuda"; cuda->device_type = kCUDA; cuda->device_id = 0;
    cuda->attrs.exists = 1; cuda->attrs.max_threads_per_block = 128;
    cuda->attrs.max_shared_memory_per_block = 48*1024;
    const Target synthetic_cuda{ObjectRef(cuda)};
    const auto cuda_lowered = test_support::LowerPrimitivesForTest(Function({data,mask},call), synthetic_cuda);
    Check(tir::BindCudaThreads(cuda_lowered.lowered[0]->prim_func,synthetic_cuda).prim_func().defined(),
          "masked_softmax must use the production CUDA multi-stage schedule");
    const auto rejects = [&](Function function,const std::string& reason) {
        Rejects([&] { (void)relay::InferTypePass(function); },reason);
    };
    rejects(FunctionFor({}, {},"float32",0),"rank >= 1");
    rejects(FunctionFor({2,3},{3},"int32",-1),"float32 or float64");
    rejects(FunctionFor({2,0},{0},"float32",-1),"positive reduction");
    rejects(FunctionFor({2,3},{3},"float32",2),"axis out of range");
    rejects(FunctionFor({2,1},{3},"float32",-1),"cannot expand");
    rejects(FunctionFor({2,3},{2},"float32",-1),"broadcast");
    const Var integer_mask("int_mask",TensorType({3},"int32"));
    rejects(Function({data,integer_mask},Op("masked_softmax",{data,integer_mask})),"bool mask");
    rejects(Function({data,mask},Op("masked_softmax",{data,mask},relay::TransposeAttrs::Create({1,0}))),"SoftmaxAttrs");
    rejects(Function({data},Op("masked_softmax",{data})),"input");
}

template<class T> void TestStatic(const api::CompileConfig& config,
    const std::shared_ptr<profiling::ProfileContext>& profile,const Array<int64_t>& shape,
    const Array<int64_t>& mask_shape,const std::string& dtype,int axis) {
    auto graph = api::Compiler::Compile(FunctionFor(shape,mask_shape,dtype,axis),config);
    Check(graph.plan().calls().size() == 1,"masked softmax must remain one ordinary primitive");
    const runtime::RuntimeSession session(graph.module(),graph.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<uint8_t> mask(Size(mask_shape));
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = pattern == 1 || pattern == 3 || (pattern == 2 && i%5 != 0 && i >= (mask_shape.empty() ? 1 : static_cast<size_t>(mask_shape[mask_shape.size()-1])));
        std::vector<T> data(Size(shape));
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = T(std::sin(double(i)*.37)*1000);
            if (pattern == 3) data[i] = (i%3 ? T(-1) : T(1))*std::numeric_limits<T>::max();
            if (!mask[MaskIndex(i,shape,mask_shape)]) data[i] = i%2
                ? std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::infinity();
        }
        const auto expected = Reference(data,mask,shape,mask_shape,axis);
        const auto count = Submits(profile);
        const auto outputs = session.Run({Tensor(shape,data,dtype),Tensor(mask_shape,mask,"bool")},
            {{"stage","masked_softmax_static"},{"pattern",std::to_string(pattern)},{"dtype",dtype}});
        Check(outputs.size() == 1 && Submits(profile) == count+1,"missing masked LLVM invocation");
        Close<T>(outputs[0],expected,sizeof(T) == 4 ? 2e-6 : 1e-12);
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"masked runtime touched cache");
    }
    graph = {}; ci::ClearPrimitiveCacheForTesting();
    std::vector<uint8_t> empty(Size(mask_shape),0);
    std::vector<T> poison(Size(shape),std::numeric_limits<T>::quiet_NaN());
    Close<T>(session.Run({Tensor(shape,poison,dtype),Tensor(mask_shape,empty,"bool")})[0],
        std::vector<double>(poison.size(),0),0);
}

void TestCompatibilityAndIdentity(const api::CompileConfig& config) {
    const Var data("logits",TensorType({2,2},"float32"));
    const auto masked = [&](int axis,uint8_t enabled) {
        return Function({data},Op("masked_softmax",{data,Constant(Tensor<uint8_t>({2,2},
            std::vector<uint8_t>(4,enabled),"bool"))},relay::SoftmaxAttrs::Create(axis)));
    };
    ci::ClearPrimitiveCacheForTesting();
    auto rows = api::Compiler::Compile(masked(1,1),config);
    const auto before = ci::GetPrimitiveCacheStats();
    auto empty = api::Compiler::Compile(masked(1,0),config);
    auto columns = api::Compiler::Compile(masked(0,1),config);
    Check(rows.artifact_pins()[0].record().artifact_key == empty.artifact_pins()[0].record().artifact_key &&
          rows.artifact_pins()[0].record().artifact_key != columns.artifact_pins()[0].record().artifact_key &&
          ci::GetPrimitiveCacheStats().hits == before.hits+1,"mask payload relocation or axis identity drift");
    const runtime::RuntimeSession row_session(rows.module(),rows.plan()), empty_session(empty.module(),empty.plan()),
        column_session(columns.module(),columns.plan());
    rows = {}; empty = {}; columns = {}; ci::ClearPrimitiveCacheForTesting();
    const std::vector<float> logits{1,2,4,8};
    const auto input = Tensor<float>({2,2},logits,"float32");
    Close<float>(row_session.Run({input})[0],Reference(logits,{1,1,1,1},{2,2},{2,2},1),2e-6);
    Close<float>(column_session.Run({input})[0],Reference(logits,{1,1,1,1},{2,2},{2,2},0),2e-6);
    Close<float>(empty_session.Run({input})[0],{0,0,0,0},0);

    const auto ordinary = api::Compiler::Compile(Function({data},Op("softmax",{data},relay::SoftmaxAttrs::Create(-1))),config);
    const runtime::RuntimeSession ordinary_session(ordinary.module(),ordinary.plan());
    const auto result = Read<float>(ordinary_session.Run({Tensor<float>({2,2},
        std::vector<float>(4,-std::numeric_limits<float>::infinity()),"float32")})[0]);
    Check(std::all_of(result.begin(),result.end(),[](float value) { return std::isnan(value); }),
        "ordinary Softmax all-negative-infinity behavior changed");
    std::cout << "masked_softmax: axis identity, cached bool constant relocation and ordinary Softmax compatibility passed\n";
}

Function Attention() {
    const Var q("q",TensorType({1,2,3,4},"float32")), kt("kt",TensorType({1,2,4,5},"float32"));
    const Var v("v",TensorType({1,2,5,3},"float32")), mask("mask",TensorType({3,5},"bool"));
    const auto scale = Constant(Tensor<float>({}, {0.5f},"float32"));
    const auto scores = Op("mul",{Op("matmul",{q,kt}),scale});
    const auto weights = Op("masked_softmax",{scores,mask},relay::SoftmaxAttrs::Create(-1));
    return Function({q,kt,v,mask},Tuple({weights,Op("matmul",{weights,v})}));
}
void RunAttention(const api::CompiledGraph& graph,int64_t batch,int64_t queries,int64_t keys,
                   const std::shared_ptr<profiling::ProfileContext>& profile) {
    const Array<int64_t> qs{batch,2,queries,4},ks{batch,2,4,keys},vs{batch,2,keys,3},ms{queries,keys};
    const auto values = [](size_t size,double scale) {
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) data[i] = static_cast<float>(std::sin(double(i+1)*scale));
        return data;
    };
    const auto q = values(Size(qs),.21), k = values(Size(ks),.17), v = values(Size(vs),.39);
    std::vector<uint8_t> mask(Size(ms));
    for (int64_t row = 1; row < queries; ++row) for (int64_t col = 0; col < keys; ++col) mask[row*keys+col] = col%3 != 2;
    std::vector<float> scores(batch*2*queries*keys);
    for (int64_t bh = 0; bh < batch*2; ++bh) for (int64_t row = 0; row < queries; ++row) for (int64_t col = 0; col < keys; ++col) {
        double sum = 0;
        for (int64_t d = 0; d < 4; ++d) sum += double(q[(bh*queries+row)*4+d])*k[(bh*4+d)*keys+col];
        scores[(bh*queries+row)*keys+col] = static_cast<float>(sum*.5);
    }
    const auto weights = Reference(scores,mask,{batch,2,queries,keys},ms,-1);
    std::vector<double> output(batch*2*queries*3,0);
    for (int64_t bh = 0; bh < batch*2; ++bh) for (int64_t row = 0; row < queries; ++row) for (int64_t d = 0; d < 3; ++d) {
        for (int64_t col = 0; col < keys; ++col) output[(bh*queries+row)*3+d] += weights[(bh*queries+row)*keys+col]*v[(bh*keys+col)*3+d];
    }
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto count = Submits(profile);
    const runtime::RuntimeSession session(graph.module(),graph.plan());
    const auto result = session.Run({Tensor(qs,q,"float32"),Tensor(ks,k,"float32"),Tensor(vs,v,"float32"),Tensor(ms,mask,"bool")},
        {{"stage","masked_attention"},{"batch",std::to_string(batch)},{"queries",std::to_string(queries)},{"keys",std::to_string(keys)}});
    Check(result.size() == 2 && Submits(profile)-count == 4,"masked attention production call count drifted");
    Close<float>(result[0],weights,2e-6); Close<float>(result[1],output,3e-6);
    Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"attention runtime touched cache");
    const auto submitted = Submits(profile);
    Rejects([&] { (void)session.Run({Tensor(qs,q,"float32"),Tensor(ks,k,"float32"),Tensor(vs,v,"float32"),
        Tensor<uint8_t>({queries,keys+1},std::vector<uint8_t>(queries*(keys+1),1),"bool")}); },"");
    Check(Submits(profile) == submitted && SameStats(stats,ci::GetPrimitiveCacheStats()),
        "invalid mask shape reached a kernel or cache");
    std::cout << "masked attention LLVM: B=" << batch << " S=" << queries << " T=" << keys
              << ", exact-zero first query, probabilities and context match independent reference\n";
}
} // namespace

int main() {
    try {
        TestContract();
#if KXC_USE_LLVM
        profiling::ProfileOptions options; options.enabled = true;
        options.ir_capture_mode = profiling::IRCaptureMode::kDisabled; options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path()/"out"/"masked_softmax_profile").string();
        const auto profile = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(profile,"masked_softmax");
        const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()),2,options);
        TestCompatibilityAndIdentity(config);
        TestStatic<float>(config,profile,{2,3,5},{3,5},"float32",-1);
        TestStatic<double>(config,profile,{2,3,5},{1,3,5},"float64",-1);
        TestStatic<float>(config,profile,{2,3,4},{2,3,4},"float32",1);
        TestStatic<double>(config,profile,{3},{},"float64",0);
        TestStatic<float>(config,profile,{0,5},{5},"float32",1);
        TestStatic<float>(config,profile,{1,7},{1,7},"float32",-1);
        RunAttention(api::Compiler::Compile(Attention(),config),1,3,5,profile);
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
        using Adapter = restricted::RestrictedSymbolicShapeAdapter;
        const std::vector<restricted::InputAxisSymbol> axes{
            {0,0,"B",1,2,1},{0,2,"S",1,4,1},{1,0,"B",1,2,1},{1,3,"T",1,6,1},
            {2,0,"B",1,2,1},{2,2,"T",1,6,1},{3,0,"S",1,4,1},{3,1,"T",1,6,1}};
        const auto graph = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(Adapter::Prepare(Attention(),config,axes)));
        for (const auto& dims : std::vector<std::vector<int64_t>>{{1,1,1},{1,3,5},{2,4,6},{2,2,3}}) {
            RunAttention(graph,dims[0],dims[1],dims[2],profile);
        }
        auto invalid = axes; invalid[3].lower = 0; invalid[5].lower = 0; invalid[7].lower = 0;
        const auto stats = ci::GetPrimitiveCacheStats();
        Rejects([&] { (void)Adapter::Prepare(Attention(),config,invalid); },"positive");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"invalid masked reduction reached compilation");
#endif
        profile->Flush();
#endif
        std::cout << "masked_softmax: contracts, broadcast/axes, all-false/true/mixed masks and masked NaN/Inf passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
