/*! \file test/compiled_module_test.cpp
 * \brief Public CompiledModule invocation ABI, ownership, and dynamic bounds.
 */
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {
using namespace kxc;
using namespace kxc::api;
using namespace kxc::codegen;
using runtime::NDArray;
#define CHECK(x) do { if (!(x)) { std::cerr << __FUNCTION__ << ": " #x "\n"; return false; } } while (false)
DLDataType F32() { return runtime::DataTypeFromString("float32"); }
DLDataType U64() { return {kDLUInt, 64, 1}; }
bool SameDType(DLDataType lhs, DLDataType rhs) { return lhs.code==rhs.code && lhs.bits==rhs.bits && lhs.lanes==rhs.lanes; }
bool Throws(const std::function<void()>& f) { try { f(); } catch (const std::exception&) { return true; } return false; }

class Recorder final : public KernelLauncher {
public:
    explicit Recorder(bool invalid_completion = false, bool ready = true)
        : invalid_(invalid_completion), ready_(ready) {}
    bool IsReady() const noexcept override { return ready_; }
    AsyncOperation Launch(const Array<NDArray>& args, const DeviceStream& stream,
                          const ObjectRef&) const override {
        ++calls; seen=args;
        if (invalid_) return AsyncOperation();
        Array<Storage> keep; for (const auto& value : args) keep.push_back(value.storage());
        return AsyncOperation::Completed(stream, std::move(keep));
    }
    mutable int calls{0}; mutable Array<NDArray> seen;
private: bool invalid_; bool ready_;
};

internal::CompiledModuleEntry Entry(const KernelSignature& signature,
                                    const std::shared_ptr<Recorder>& launcher,
                                    std::shared_ptr<const ModuleInvocationContract> contract = {}) {
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    return {tir::PrimFunc(), signature, metadata, CompiledKernel(signature, metadata, launcher), std::move(contract)};
}
CompiledModule Build(const KernelSignature& signature, const std::shared_ptr<Recorder>& launcher,
                     std::shared_ptr<const ModuleInvocationContract> contract = {},
                     Map<String, NDArray> constants = {}) {
    return internal::BuildCompiledModule(BuildTarget(Device::CPU()), {Entry(signature, launcher, std::move(contract))}, std::move(constants));
}
std::shared_ptr<const ModuleInvocationContract> DynamicContract(bool scalar = true, size_t budget = 4096) {
    ModuleInputContract in{{{0, 0, 64, 1, std::nullopt, std::nullopt}}};
    const auto twice=ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0, 0), ModuleShapeExpr::Const(2));
    ModuleTensorContract out; out.max_bytes=4096;
    out.logical={twice}; out.physical={twice}; out.valid={twice};
    std::vector<ModuleRuntimeExtentScalar> scalars;
    if (scalar) scalars.push_back({twice});
    return std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{in}, std::vector<ModuleTensorContract>{out}, std::move(scalars), budget);
}
CompiledModule Dynamic(const std::shared_ptr<Recorder>& launcher, bool scalar = true, size_t budget = 4096) {
    Array<KernelArgSpec> args{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU())};
    if (scalar) args.push_back(KernelArgSpec("extent",KernelArgRole::kRuntimeExtent,U64(),{1},Device::CPU(),8));
    args.push_back(KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true));
    return Build(KernelSignature("dynamic",args),launcher,DynamicContract(scalar,budget));
}

bool StaticPublicInvokeAndConstants() {
    auto launcher=std::make_shared<Recorder>();
    KernelSignature sig("static", {KernelArgSpec("x",KernelArgRole::kInput,F32(),{2,3},Device::CPU(),4),
        KernelArgSpec("c",KernelArgRole::kConstant,F32(),{3},Device::CPU(),4,false,"c"),
        KernelArgSpec("y",KernelArgRole::kOutput,F32(),{2,3},Device::CPU(),64,true)});
    NDArray source=NDArray::Zeros({3},F32(),Device::CPU()); Map<String,NDArray> constants; constants.Set("c",source);
    auto module=Build(sig,launcher,{},constants); auto result=module.Invoke("static",{NDArray::Zeros({2,3},F32(),Device::CPU(),4)},DeviceStream::Default(Device::CPU()));
    CHECK(result.operation.IsReady() && launcher->calls==1 && result.outputs.size()==1);
    CHECK(result.operation->retained_storage.size()==3);
    CHECK(result.outputs[0].physical==std::vector<ModuleExtent>({2,3}) &&
          internal::BorrowCompiledModuleInvocationContract(module, "static").outputs()[0].max_bytes==24);
    CHECK(SameDType(result.outputs[0].storage.dtype(),F32()) && result.outputs[0].storage.device()==Device::CPU() && result.outputs[0].storage.storage()->alignment>=64);
    CHECK(launcher->seen.size()==3 && launcher->seen[1].get()!=source.get());
    auto copy=module.constants(); const float changed[3]={1,2,3}; copy.at("c").CopyFromBytes(changed,sizeof(changed));
    std::vector<float> actual(3); module.constants().at("c").CopyToBytes(actual.data(),sizeof(changed)); CHECK(actual==std::vector<float>({0,0,0}));
    CHECK(module.HasFunction("static") && module.entry_count()==1 && module.symbols().size()==1);
    return true;
}
bool PublicSourceValidation() {
    auto launcher=std::make_shared<Recorder>();
    KernelSignature sig("source",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{2,3},Device::CPU(),4),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{2,3},Device::CPU(),4,true)});
    auto module=Build(sig,launcher); auto stream=DeviceStream::Default(Device::CPU());
    CHECK(Throws([&]{module.Invoke("source",{},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray(),NDArray()},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray()},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Zeros({2,3},runtime::DataTypeFromString("int32"),Device::CPU())},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Empty({2,3},DLDataType{kDLFloat,32,2},Device::CPU())},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Zeros({6},F32(),Device::CPU())},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Zeros({2,4},F32(),Device::CPU())},stream);}) && launcher->calls==0);
    NDArray backing=NDArray::Empty({7},F32(),Device::CPU()); auto bad=backing.CreateView({2,3},{3,1},1);
    CHECK(Throws([&]{module.Invoke("source",{bad},stream);}) && launcher->calls==0);
    auto* range=const_cast<runtime::NDArrayNode*>(bad.As<runtime::NDArrayNode>()); range->byte_offset=range->storage.capacity_bytes();
    CHECK(Throws([&]{module.Invoke("source",{bad},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Zeros({2,3},F32(),Device::CPU(),4)},DeviceStream());}) && launcher->calls==0);
    return true;
}
bool ObjectReadinessZeroByteAndSymbols() {
    CHECK(Throws([&]{ CompiledModule invalid(ObjectRef(Device::CPU())); }));
    KernelSignature one("one",{KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),4,true)});
    KernelLaunchMetadata metadata(Device::CPU(),CodeGenBackend::kLLVM);
    auto not_ready=std::make_shared<Recorder>(false,false);
    CHECK(Throws([&]{ internal::BuildCompiledModule(BuildTarget(Device::CPU()),
        {{tir::PrimFunc(),one,metadata,CompiledKernel(one,metadata,not_ready),{}}},{}); }));

    auto first=std::make_shared<Recorder>(); auto second=std::make_shared<Recorder>();
    KernelSignature a("entry_a",{KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)});
    KernelSignature b("entry_b",{KernelArgSpec("y",KernelArgRole::kOutput,F32(),{2},Device::CPU(),1,true)});
    auto module=internal::BuildCompiledModule(BuildTarget(Device::CPU()),
        {Entry(a,first),Entry(b,second)},{});
    const auto stream=DeviceStream::Default(Device::CPU());
    CHECK(module.entry_count()==2 && module.HasFunction("entry_a") && module.HasFunction("entry_b") && !module.HasFunction("missing"));
    CHECK(Throws([&]{module.Invoke("missing",{},stream);}) && Throws([&]{module.signature("missing");}));
    CHECK(Throws([&]{module.Invoke("entry_b",{},stream,1);}) && first->calls==0 && second->calls==0);
    CHECK(module.Invoke("entry_b",{},stream).outputs[0].physical==std::vector<ModuleExtent>({2}) && first->calls==0 && second->calls==1);
    CHECK(Throws([&]{internal::BuildCompiledModule(BuildTarget(Device::CPU()),{Entry(a,first),Entry(a,second)},{});}));

    KernelSignature empty("empty",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{0},Device::CPU(),64),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{0},Device::CPU(),64,true)});
    auto empty_launcher=std::make_shared<Recorder>(); auto empty_module=Build(empty,empty_launcher);
    CHECK(empty_module.Invoke("empty",{NDArray::Empty({0},F32(),Device::CPU())},stream).operation.IsReady() && empty_launcher->calls==1);
    NDArray backing=NDArray::Empty({2},F32(),Device::CPU(),4);
    KernelSignature offset("offset",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{1},Device::CPU(),4),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),4,true)});
    auto offset_launcher=std::make_shared<Recorder>(); auto offset_module=Build(offset,offset_launcher);
    CHECK(offset_module.Invoke("offset",{backing.CreateView({1},{1},sizeof(float))},stream).operation.IsReady() && offset_launcher->calls==1);
    return true;
}
bool DynamicGateGuardsScalarsAndPreallocation() {
    auto launcher=std::make_shared<Recorder>(); auto module=Dynamic(launcher); auto stream=DeviceStream::Default(Device::CPU());
#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    for (int64_t s : {0,2,3}) { auto result=module.Invoke("dynamic",{NDArray::Zeros({s},F32(),Device::CPU())},stream); CHECK(result.outputs[0].logical[0]==static_cast<ModuleExtent>(2*s)); CHECK(result.operation.IsReady()); }
    CHECK(launcher->calls==3 && launcher->seen.size()==3);
    std::uint64_t extent=0; launcher->seen[1].CopyToBytes(&extent,sizeof(extent)); CHECK(extent==6);
    CHECK(Throws([&]{module.Invoke("dynamic",{NDArray::Zeros({65},F32(),Device::CPU())},stream);}) && launcher->calls==3);
    CHECK(Throws([&]{internal::InvokeCompiledModuleWithOutputs(module,"dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},{NDArray::Zeros({3},F32(),Device::CPU())},stream);}) && launcher->calls==3);
    CHECK(internal::InvokeCompiledModuleWithOutputs(module,"dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},{NDArray::Zeros({4},F32(),Device::CPU())},stream).IsReady());
#else
    CHECK(Throws([&]{module.Invoke("dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},stream);}) && launcher->calls==0);
#endif
    return true;
}
bool MultipleRuntimeExtentOrdering() {
    auto launcher=std::make_shared<Recorder>(); const auto extent=ModuleShapeExpr::InputAxis(0,0); const auto twice=ModuleShapeExpr::Mul(extent,ModuleShapeExpr::Const(2));
    ModuleInputContract input{{{0,0,64,1,std::nullopt,std::nullopt}}}; ModuleTensorContract output; output.logical={twice}; output.physical={twice}; output.valid={twice}; output.max_bytes=4096;
    KernelSignature signature("two_extents",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU()),KernelArgSpec("twice",KernelArgRole::kRuntimeExtent,U64(),{1},Device::CPU(),8),KernelArgSpec("extent",KernelArgRole::kRuntimeExtent,U64(),{1},Device::CPU(),8),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true)});
    auto contract=std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{input},std::vector<ModuleTensorContract>{output},std::vector<ModuleRuntimeExtentScalar>{{twice},{extent}});
    auto module=Build(signature,launcher,contract); const auto stream=DeviceStream::Default(Device::CPU());
#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    CHECK(module.Invoke("two_extents",{NDArray::Zeros({3},F32(),Device::CPU())},stream).operation.IsReady() && launcher->seen.size()==4);
    std::uint64_t first=0, second=0; launcher->seen[1].CopyToBytes(&first,sizeof(first)); launcher->seen[2].CopyToBytes(&second,sizeof(second)); CHECK(first==6 && second==3);
#else
    CHECK(Throws([&]{module.Invoke("two_extents",{NDArray::Zeros({3},F32(),Device::CPU())},stream);}) && launcher->calls==0);
#endif
    return true;
}
bool BudgetAndExtentsBeforeAllocationOrLaunch() {
    auto launcher=std::make_shared<Recorder>(); auto module=Dynamic(launcher,true,8); auto stream=DeviceStream::Default(Device::CPU());
#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    CHECK(Throws([&]{module.Invoke("dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},stream,4096);}) && launcher->calls==0);
    auto contract=DynamicContract(true); const auto& output=contract->outputs()[0]; CHECK(!output.logical.empty());
#endif
    return true;
}
bool MultiEntryConstantsAndCanonicalIdentity() {
    const String key("shared");
    KernelSignature a("constant_a",{KernelArgSpec("c",KernelArgRole::kConstant,F32(),{1},Device::CPU(),64,false,key),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)});
    KernelSignature b("constant_b",{KernelArgSpec("c",KernelArgRole::kConstant,F32(),{1},Device::CPU(),4,false,key),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)});
    NDArray source=NDArray::Zeros({1},F32(),Device::CPU()); Map<String,NDArray> constants; constants.Set(key,source);
    auto la=std::make_shared<Recorder>(); auto lb=std::make_shared<Recorder>();
    auto module=internal::BuildCompiledModule(BuildTarget(Device::CPU()),{Entry(a,la),Entry(b,lb)},constants);
    CHECK(module.constants().at(key).storage()->alignment>=64);
    CHECK(module.Invoke("constant_a",{},DeviceStream::Default(Device::CPU())).operation.IsReady() && la->seen[0].get()!=source.get());
    CHECK(module.Invoke("constant_b",{},DeviceStream::Default(Device::CPU())).operation.IsReady() && lb->seen[0].get()!=source.get());
    KernelSignature conflict("conflict",{KernelArgSpec("c",KernelArgRole::kConstant,F32(),{2},Device::CPU(),1,false,key),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{2},Device::CPU(),1,true)});
    CHECK(Throws([&]{internal::BuildCompiledModule(BuildTarget(Device::CPU()),{Entry(a,la),Entry(conflict,lb)},constants);}));
    return true;
}
bool ContractAssemblyAndExpressionLimits() {
    ModuleInputContract first{{{0,1,8,2,2,std::nullopt}}};
    ModuleInputContract second{{{0,1,8,1,std::nullopt,ModuleAxisReference{0,0}}}};
    ModuleTensorContract out; out.max_bytes=64; out.logical={ModuleShapeExpr::Const(1)}; out.physical=out.logical; out.valid=out.logical;
    const auto canonical=ModuleInvocationContract({first,second},{out},{}).CanonicalBytes();
    CHECK(ModuleInvocationContract({first,second},{out},{}).abi_version()==2 && canonical.rfind("KXC_MODULE_INVOKE_V2;",0)==0);
    CHECK(canonical==ModuleInvocationContract({first,second},{out},{}).CanonicalBytes());
    auto altered=out; altered.logical={ModuleShapeExpr::Const(2)}; altered.physical=altered.logical; altered.valid=altered.logical;
    CHECK(canonical!=ModuleInvocationContract({first,second},{altered},{}).CanonicalBytes());
    CHECK(Throws([&]{ ModuleInvocationContract({first},{altered},{}).Validate(KernelSignature("bad_output", {KernelArgSpec("x",KernelArgRole::kInput,F32(),{2},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)})); }));
    auto guard_changed=first; guard_changed.axis_guards[0].upper=7;
    CHECK(canonical!=ModuleInvocationContract({guard_changed,second},{out},{}).CanonicalBytes());
    auto unordered_second=second; unordered_second.axis_guards.clear();
    auto reordered=ModuleInvocationContract({unordered_second,first},{out},{}).CanonicalBytes();
    CHECK(canonical!=reordered);
    CHECK(canonical!=ModuleInvocationContract({first,second},{out},{{ModuleShapeExpr::Const(1)}}).CanonicalBytes());
    CHECK(Throws([&]{
        ModuleInvocationContract({first},{out},{{ModuleShapeExpr::Const(1)}}).Validate(
            KernelSignature("bad_scalar", {KernelArgSpec("x", KernelArgRole::kInput,
                F32(), {2}, Device::CPU()), KernelArgSpec("extent", KernelArgRole::kInput,
                U64(), {1}, Device::CPU()), KernelArgSpec("y", KernelArgRole::kOutput,
                F32(), {1}, Device::CPU(), 1, true)}));
    }));
    auto unordered=first; unordered.axis_guards.push_back({0,1,8,1,std::nullopt,std::nullopt}); CHECK(Throws([&]{ModuleInvocationContract({unordered},{out},{},0).Validate(KernelSignature("guards",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{2},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)}));}));
    auto forward=first; forward.axis_guards[0].equal_to=ModuleAxisReference{0,0}; CHECK(Throws([&]{ModuleInvocationContract({forward},{out},{},0).Validate(KernelSignature("forward",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{2},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)}));}));
    auto expr=ModuleShapeExpr::Const(1); for(size_t i=0;i<=ModuleShapeExpr::kMaxDepth;++i) expr=ModuleShapeExpr::Add(expr,ModuleShapeExpr::Const(1)); out.logical={expr}; out.physical={expr}; out.valid={expr}; CHECK(Throws([&]{ModuleInvocationContract({first},{out},{},0).Validate(KernelSignature("depth",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{2},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)}));}));
    CHECK(Throws([&]{ModuleShapeExpr::FloorDiv(ModuleShapeExpr::Const(1),ModuleShapeExpr::Const(0)).Evaluate({});}));
    ModuleInputContract unbounded{{{0,0,64,1,std::nullopt,std::nullopt}}}; ModuleTensorContract dynamic_output; dynamic_output.max_bytes=4096;
    const auto rejects_divisor=[&](ModuleShapeExpr divisor) { const auto expr=ModuleShapeExpr::FloorDiv(ModuleShapeExpr::Const(8),divisor); dynamic_output.logical={expr}; dynamic_output.physical={expr}; dynamic_output.valid={expr}; return Throws([&]{ Build(KernelSignature("bad_divisor",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true)}),std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{unbounded},std::vector<ModuleTensorContract>{dynamic_output},std::vector<ModuleRuntimeExtentScalar>{})); }); };
    CHECK(rejects_divisor(ModuleShapeExpr::Const(0)) && rejects_divisor(ModuleShapeExpr::InputAxis(0,0)) && rejects_divisor(ModuleShapeExpr::Add(ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()),ModuleShapeExpr::Const(1))));
    const auto rejects_expression=[&](ModuleShapeExpr expr) { dynamic_output.logical={expr}; dynamic_output.physical={expr}; dynamic_output.valid={expr}; return Throws([&]{ Build(KernelSignature("bad_range",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true)}),std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{unbounded},std::vector<ModuleTensorContract>{dynamic_output},std::vector<ModuleRuntimeExtentScalar>{})); }); };
    CHECK(rejects_expression(ModuleShapeExpr::Add(ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()),ModuleShapeExpr::Const(1))) && rejects_expression(ModuleShapeExpr::Mul(ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()),ModuleShapeExpr::Const(2))) && rejects_expression(ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0,0),ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()))));
    ModuleInputContract bounded{{{0,0,3,1,std::nullopt,std::nullopt}}}; const auto bounded_expr=ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0,0),ModuleShapeExpr::Const(2)); dynamic_output.logical={bounded_expr}; dynamic_output.physical={bounded_expr}; dynamic_output.valid={bounded_expr};
    const KernelSignature dynamic_signature("bounded_range",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true)});
    CHECK(!Throws([&]{ Build(dynamic_signature,std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{bounded},std::vector<ModuleTensorContract>{dynamic_output},std::vector<ModuleRuntimeExtentScalar>{})); }));
    ModuleTensorContract one; one.logical={ModuleShapeExpr::Const(1)}; one.physical=one.logical; one.valid=one.logical; one.max_bytes=4096;
    ModuleInputContract partial{{{0,0,4,1,std::nullopt,std::nullopt}}};
    CHECK(Throws([&]{ Build(KernelSignature("partial_guard",{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1,-1},Device::CPU()),KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true)}),std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{partial},std::vector<ModuleTensorContract>{one},std::vector<ModuleRuntimeExtentScalar>{})); }));
    ModuleInputContract outside_domain{{{0,0,static_cast<ModuleExtent>(std::numeric_limits<int64_t>::max())+1,1,std::nullopt,std::nullopt}}};
    CHECK(Throws([&]{ Build(dynamic_signature,std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{outside_domain},std::vector<ModuleTensorContract>{one},std::vector<ModuleRuntimeExtentScalar>{})); }));
    ModuleInputContract ordered_input{{{0,2,4,1,std::nullopt,std::nullopt}}}; const auto axis=ModuleShapeExpr::InputAxis(0,0);
    const auto admits_output=[&](ModuleTensorContract candidate) { return !Throws([&]{ Build(dynamic_signature,std::make_shared<Recorder>(),std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{ordered_input},std::vector<ModuleTensorContract>{candidate},std::vector<ModuleRuntimeExtentScalar>{})); }); };
    ModuleTensorContract ordered; ordered.valid={ModuleShapeExpr::Const(1)}; ordered.logical={axis}; ordered.physical={ModuleShapeExpr::Const(5)}; ordered.max_bytes=4096;
    ModuleTensorContract equal; equal.logical={axis}; equal.physical={axis}; equal.valid={axis}; equal.max_bytes=4096;
    ModuleTensorContract invalid_constants; invalid_constants.valid={ModuleShapeExpr::Const(2)}; invalid_constants.logical={ModuleShapeExpr::Const(1)}; invalid_constants.physical={ModuleShapeExpr::Const(1)}; invalid_constants.max_bytes=4096;
    ModuleTensorContract crossing; crossing.valid={ModuleShapeExpr::Const(1)}; crossing.logical={axis}; crossing.physical={ModuleShapeExpr::Const(3)}; crossing.max_bytes=4096;
    CHECK(admits_output(ordered) && admits_output(equal) && !admits_output(invalid_constants) && !admits_output(crossing));
    CHECK(Throws([&]{ModuleShapeExpr::Mul(ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()),ModuleShapeExpr::Const(2)).Evaluate({});}));
    return true;
}
bool InvalidCompletionIsRejected() {
    auto launcher=std::make_shared<Recorder>(true); KernelSignature sig("bad",{KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)}); auto module=Build(sig,launcher);
    CHECK(Throws([&]{module.Invoke("bad",{},DeviceStream::Default(Device::CPU()));}) && launcher->calls==1); return true;
}
}  // namespace
int main() {
    const std::vector<std::pair<const char*,bool(*)()>> tests={{"static",StaticPublicInvokeAndConstants},{"source",PublicSourceValidation},{"object_symbols",ObjectReadinessZeroByteAndSymbols},{"dynamic",DynamicGateGuardsScalarsAndPreallocation},{"multiple_scalars",MultipleRuntimeExtentOrdering},{"budget",BudgetAndExtentsBeforeAllocationOrLaunch},{"constants",MultiEntryConstantsAndCanonicalIdentity},{"contract",ContractAssemblyAndExpressionLimits},{"completion",InvalidCompletionIsRejected}};
    int failed=0; for(const auto& test:tests) try { if(!test.second()) ++failed; else std::cout<<"[PASS] "<<test.first<<"\n"; } catch(const std::exception& e) { std::cerr<<"[FAIL] "<<test.first<<": "<<e.what()<<"\n"; ++failed; } return failed?1:0;
}
