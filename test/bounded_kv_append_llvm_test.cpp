// One compiled artifact appends a fixed-size fragment to variable-length KV.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/compiler/internal/te_to_tir.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;
void Check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Action> void Rejects(Action action, const std::string& text) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(text) != std::string::npos,
              "wrong rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("missing rejection: " + text);
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
Expr Concat(Expr a, Expr b) {
    return Op("concatenate", {a,b}, relay::ConcatenateAttrs::Create(1));
}
NDArray Input(Array<int64_t> shape, int seed = 0, const std::string& dtype = "float32") {
    auto value = NDArray::Empty(shape, runtime::DataTypeFromString(dtype), Device::CPU());
    std::vector<float> data(value.NBytes()/4);
    for (size_t i = 0; i < data.size(); ++i) data[i] = float(int(i % 29) - 14 + seed) / 8;
    if (!data.empty()) value.CopyFromBytes(data.data(), value.NBytes());
    return value;
}
template<class T> std::vector<T> Read(const NDArray& value) {
    std::vector<T> data(value.NBytes()/sizeof(T));
    if (!data.empty()) value.CopyToBytes(data.data(), value.NBytes());
    return data;
}
std::vector<restricted::InputAxisSymbol> Axes(int64_t upper = 8) {
    return {{0,0,"B",1,3,1},{0,1,"P",0,upper,1},{1,0,"B",1,3,1}};
}
std::pair<size_t,size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir())/"events.jsonl");
    Check(in.good(), "missing KV append profile");
    std::pair<size_t,size_t> result{};
    std::string line;
    while (std::getline(in,line)) {
        result.first += line.find("\"event_type\":\"kernel_submit\"") != std::string::npos;
        result.second += line.find("\"event_type\":\"alloc\"") != std::string::npos;
    }
    return result;
}
void TestAppend(const api::CompileConfig& config,
                const std::shared_ptr<profiling::ProfileContext>& context,
                int64_t count, bool prepend) {
    const Var past("past",TensorType({1,4,2,3},"float32"));
    const Var token("token",TensorType({1,count,2,3},"float32"));
    const Expr joined = prepend ? Concat(token,past) : Concat(past,token);
    const Function graph({past,token},Tuple({joined,Op("nn_relu",{joined}),Op("shape_of",{joined})}));
    const auto prepared = Adapter::Prepare(graph, config, Axes());
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size() == 3, "append must use three ordinary units");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto before = Counts(context);
    for (const auto& [batch,length] : std::vector<std::pair<int64_t,int64_t>>{{1,0},{1,1},{2,4},{3,8}}) {
        const auto a = Input({batch,length,2,3}), b = Input({batch,count,2,3},31);
        const auto outputs = session.Run({a,b},{{"stage","bounded_kv_append"},
            {"batch",std::to_string(batch)},{"past",std::to_string(length)},
            {"append_count",std::to_string(count)},{"prepend",prepend ? "true" : "false"}});
        const std::vector<int64_t> expected_shape{batch,length+count,2,3};
        Check(outputs.size() == 3 && outputs[0].shape().size() == 4 &&
            std::equal(expected_shape.begin(),expected_shape.end(),outputs[0].shape().begin()) &&
            Read<int64_t>(outputs[2]) == expected_shape, "derived output shape or shape value mismatch");
        const auto av = Read<float>(a), bv = Read<float>(b), actual = Read<float>(outputs[0]), relu = Read<float>(outputs[1]);
        std::vector<float> reference;
        for (int64_t row = 0; row < batch; ++row) {
            const auto append_past = [&] { reference.insert(reference.end(),av.begin()+row*length*6,av.begin()+(row+1)*length*6); };
            const auto append_token = [&] { reference.insert(reference.end(),bv.begin()+row*count*6,bv.begin()+(row+1)*count*6); };
            if (prepend) { append_token(); append_past(); } else { append_past(); append_token(); }
        }
        Check(actual == reference && relu.size() == reference.size(), "KV concatenate changed payload or batch stride");
        for (size_t i = 0; i < relu.size(); ++i) Check(relu[i] == std::max(0.f,reference[i]), "derived shape consumer used the wrong stride");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()), "Run changed primitive cache");
    }
    const auto after = Counts(context);
    Check(after.first == before.first + 12, "missing append LLVM calls");
    for (const auto& inputs : std::vector<Array<NDArray>>{
        {Input({1,9,2,3}),Input({1,count,2,3})},
        {Input({4,4,2,3}),Input({4,count,2,3})},
        {Input({1,4,2,3}),Input({2,count,2,3})},
        {Input({1,4,2,3}),Input({1,count+1,2,3})},
        {Input({1,4,6}),Input({1,count,2,3})},
        {Input({1,4,2,3},0,"int32"),Input({1,count,2,3})}}) {
        Rejects([&] { (void)session.Run(inputs); }, "RuntimeSession");
        Check(Counts(context) == after && SameStats(stats,ci::GetPrimitiveCacheStats()),
              "rejected append allocated, launched, or touched the cache");
    }
    std::cout << "bounded KV " << (prepend ? "prepend" : "append") << " C=" << count
              << ": four B/P shapes, 12 LLVM calls, exact payloads, six preflight rejections\n";
}
void TestProofs(const api::CompileConfig& config) {
    const Var a("a",TensorType({1,4,2,3},"float32")), b("b",TensorType({1,1,2,3},"float32"));
    const auto stats = ci::GetPrimitiveCacheStats();
    Rejects([&] { (void)Adapter::Prepare(Function({a,b},Concat(a,b)),config,
        {{0,1,"P",0,8,1},{1,1,"Q",0,8,1}}); },"static concatenation axis");
    Rejects([&] { (void)Adapter::Prepare(Function({a,b},Concat(a,b)),config,
        {{0,0,"B",1,3,1},{0,1,"P",0,8,1},{1,0,"C",1,3,1}}); },"identical expressions");
    Rejects([&] {
        const auto prepared = Adapter::Prepare(Function({a,b},Concat(a,b)),config,Axes(std::numeric_limits<int32_t>::max()));
        (void)api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    },"int32 loop domain");
    Check(SameStats(stats,ci::GetPrimitiveCacheStats()), "invalid append proof touched cache");
}

void TestChainedAppend(const api::CompileConfig& config,
                       const std::shared_ptr<profiling::ProfileContext>& context) {
    const Var past("past",TensorType({1,4,2,3},"float32")), token("token",TensorType({1,1,2,3},"float32"));
    const auto second = Concat(Concat(past,token),token);
    auto index = NDArray::Empty({1},runtime::DataTypeFromString("int64"),Device::CPU());
    const int64_t axis = 1;
    index.CopyFromBytes(&axis,sizeof(axis));
    const auto extent = Op("gather",{Op("shape_of",{second}),Constant(index)},relay::GatherAttrs::Create(0));
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
        Adapter::Prepare(Function({past,token},Tuple({second,extent})),config,Axes())));
    Check(compiled.plan().calls().size() == 3,"chained append must fold its shape control");
    const runtime::RuntimeSession session(compiled.module(),compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto before = Counts(context);
    for (const auto& [batch,length] : std::vector<std::pair<int64_t,int64_t>>{{1,0},{1,1},{2,4},{3,8}}) {
        const auto a = Input({batch,length,2,3}), b = Input({batch,1,2,3},31);
        const auto outputs = session.Run({a,b},{{"stage","bounded_kv_append_chain"},
            {"batch",std::to_string(batch)},{"past",std::to_string(length)},{"append_count","2"}});
        Check(Read<int64_t>(outputs[1]) == std::vector<int64_t>{length+2},"shape control lost cumulative offset");
        const auto av = Read<float>(a),bv = Read<float>(b);
        std::vector<float> reference;
        for (int64_t row = 0; row < batch; ++row) {
            reference.insert(reference.end(),av.begin()+row*length*6,av.begin()+(row+1)*length*6);
            for (int i = 0; i < 2; ++i) reference.insert(reference.end(),bv.begin()+row*6,bv.begin()+(row+1)*6);
        }
        Check(Read<float>(outputs[0]) == reference,"chained append changed batch strides");
        Check(SameStats(stats,ci::GetPrimitiveCacheStats()),"chained Run touched cache");
    }
    Check(Counts(context).first == before.first+12,"missing chained append LLVM calls");
    std::cout << "bounded chained append: four B/P shapes, P+2 payload and shape control exact, 12 LLVM calls\n";
}

void TestExtentRecognition() {
    const tir::Var buffer("extent",tir::DataType::UInt(64)), foreign("foreign",tir::DataType::UInt(64));
    const auto load = relay::internal::LoadRuntimeExtent(buffer);
    const auto one = tir::IntImm(1,tir::DataType::Int(64));
    size_t slot = 99;
    int64_t offset = -1;
    for (const auto& expression : Array<tir::PrimExpr>{load+one,one+load}) {
        Check(relay::internal::MatchRuntimeExtentOffset(expression,{buffer},&slot,&offset) &&
            slot == 0 && offset == 1,"extent offset recognition failed");
        Check(!relay::internal::MatchRuntimeExtentLoad(expression,{buffer},nullptr),"direct-load recognition accepted arithmetic");
    }
    for (const auto& expression : Array<tir::PrimExpr>{
        load + tir::IntImm(-1,tir::DataType::Int(64)), load+load, load*one,
        relay::internal::LoadRuntimeExtent(foreign)+one}) {
        Check(!relay::internal::MatchRuntimeExtentOffset(expression,{buffer},nullptr),"invalid extent arithmetic was accepted");
    }
}
}  // namespace

int main() {
    try {
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true; options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path()/"out"/"bounded_kv_append_profile").string();
        const auto context = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context,"bounded_kv_append");
        const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()),2,options);
        TestProofs(config);
        TestExtentRecognition();
        TestAppend(config,context,1,false);
        TestAppend(config,context,2,true);
        TestAppend(config,context,0,false);
        TestChainedAppend(config,context);
        return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
