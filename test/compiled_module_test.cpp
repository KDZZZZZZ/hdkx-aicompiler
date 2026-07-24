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
bool Throws(const std::function<void()>& f) { try { f(); } catch (const std::exception&) { return true; } return false; }

class Recorder final : public KernelLauncher {
public:
    explicit Recorder(bool invalid_completion = false) : invalid_(invalid_completion) {}
    bool IsReady() const noexcept override { return true; }
    AsyncOperation Launch(const Array<NDArray>& args, const DeviceStream& stream,
                          const ObjectRef&) const override {
        ++calls; seen=args;
        if (invalid_) return AsyncOperation();
        Array<Storage> keep; for (const auto& value : args) keep.push_back(value.storage());
        return AsyncOperation::Completed(stream, std::move(keep));
    }
    mutable int calls{0}; mutable Array<NDArray> seen;
private: bool invalid_;
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
    ModuleInputContract in{F32(), Device::CPU(), 1, {{0, 0, 64, 1, std::nullopt, std::nullopt}}};
    const auto twice=ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0, 0), ModuleShapeExpr::Const(2));
    ModuleTensorContract out; out.dtype=F32(); out.device=Device::CPU(); out.max_bytes=4096;
    out.logical={twice}; out.physical={twice}; out.valid={twice};
    std::vector<ModuleRuntimeExtentScalar> scalars;
    if (scalar) scalars.push_back({twice, U64(), Device::CPU(), 8});
    return std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{in}, std::vector<ModuleTensorContract>{out}, std::move(scalars), budget);
}
CompiledModule Dynamic(const std::shared_ptr<Recorder>& launcher, bool scalar = true, size_t budget = 4096) {
    Array<KernelArgSpec> args{KernelArgSpec("x",KernelArgRole::kInput,F32(),{-1},Device::CPU())};
    if (scalar) args.push_back(KernelArgSpec("extent",KernelArgRole::kInput,U64(),{1},Device::CPU(),8));
    args.push_back(KernelArgSpec("y",KernelArgRole::kOutput,F32(),{-1},Device::CPU(),1,true));
    return Build(KernelSignature("dynamic",args),launcher,DynamicContract(scalar,budget));
}

bool StaticPublicInvokeAndConstants() {
    auto launcher=std::make_shared<Recorder>();
    KernelSignature sig("static", {KernelArgSpec("x",KernelArgRole::kInput,F32(),{2,3},Device::CPU(),4),
        KernelArgSpec("c",KernelArgRole::kConstant,F32(),{3},Device::CPU(),4,false,"c"),
        KernelArgSpec("y",KernelArgRole::kOutput,F32(),{2,3},Device::CPU(),4,true)});
    NDArray source=NDArray::Zeros({3},F32(),Device::CPU()); Map<String,NDArray> constants; constants.Set("c",source);
    auto module=Build(sig,launcher,{},constants); auto result=module.Invoke("static",{NDArray::Zeros({2,3},F32(),Device::CPU(),4)},DeviceStream::Default(Device::CPU()));
    CHECK(result.operation.IsReady() && launcher->calls==1 && result.outputs.size()==1);
    CHECK(result.outputs[0].physical==std::vector<ModuleExtent>({2,3}) && module.invocation_contract("static").outputs()[0].max_bytes==24);
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
    CHECK(Throws([&]{module.Invoke("source",{NDArray()},stream);}) && launcher->calls==0);
    NDArray backing=NDArray::Empty({7},F32(),Device::CPU()); auto bad=backing.CreateView({2,3},{3,1},1);
    CHECK(Throws([&]{module.Invoke("source",{bad},stream);}) && launcher->calls==0);
    auto* range=const_cast<runtime::NDArrayNode*>(bad.As<runtime::NDArrayNode>()); range->byte_offset=range->storage.capacity_bytes();
    CHECK(Throws([&]{module.Invoke("source",{bad},stream);}) && launcher->calls==0);
    CHECK(Throws([&]{module.Invoke("source",{NDArray::Zeros({2,3},F32(),Device::CPU(),4)},DeviceStream());}) && launcher->calls==0);
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
bool BudgetAndExtentsBeforeAllocationOrLaunch() {
    auto launcher=std::make_shared<Recorder>(); auto module=Dynamic(launcher,true,8); auto stream=DeviceStream::Default(Device::CPU());
#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    CHECK(Throws([&]{module.Invoke("dynamic",{NDArray::Zeros({2},F32(),Device::CPU())},stream);}) && launcher->calls==0);
    auto contract=DynamicContract(true); const auto& output=contract->outputs()[0]; CHECK(!output.logical.empty());
#endif
    return true;
}
bool ContractAssemblyAndExpressionLimits() {
    ModuleInputContract first{F32(),Device::CPU(),1,{{0,1,8,2,2,std::nullopt}}};
    ModuleInputContract second{F32(),Device::CPU(),1,{{0,1,8,1,std::nullopt,ModuleAxisReference{0,0}}}};
    ModuleTensorContract out; out.dtype=F32(); out.device=Device::CPU(); out.max_bytes=64; out.logical={ModuleShapeExpr::Const(1)}; out.physical=out.logical; out.valid=out.logical;
    CHECK(ModuleInvocationContract({first,second},{out}).CanonicalBytes()==ModuleInvocationContract({first,second},{out}).CanonicalBytes());
    auto unordered=first; unordered.axis_guards.push_back({0,1,8,1,std::nullopt,std::nullopt}); CHECK(Throws([&]{ModuleInvocationContract({unordered},{out});}));
    auto forward=first; forward.axis_guards[0].equal_to=ModuleAxisReference{0,0}; CHECK(Throws([&]{ModuleInvocationContract({forward},{out});}));
    auto expr=ModuleShapeExpr::Const(1); for(size_t i=0;i<=ModuleShapeExpr::kMaxDepth;++i) expr=ModuleShapeExpr::Add(expr,ModuleShapeExpr::Const(1)); out.logical={expr}; out.physical={expr}; out.valid={expr}; CHECK(Throws([&]{ModuleInvocationContract({first},{out});}));
    CHECK(Throws([&]{ModuleShapeExpr::FloorDiv(ModuleShapeExpr::Const(1),ModuleShapeExpr::Const(0)).Evaluate({});}));
    CHECK(Throws([&]{ModuleShapeExpr::Mul(ModuleShapeExpr::Const(std::numeric_limits<ModuleExtent>::max()),ModuleShapeExpr::Const(2)).Evaluate({});}));
    return true;
}
bool InvalidCompletionIsRejected() {
    auto launcher=std::make_shared<Recorder>(true); KernelSignature sig("bad",{KernelArgSpec("y",KernelArgRole::kOutput,F32(),{1},Device::CPU(),1,true)}); auto module=Build(sig,launcher);
    CHECK(Throws([&]{module.Invoke("bad",{},DeviceStream::Default(Device::CPU()));}) && launcher->calls==1); return true;
}
}  // namespace
int main() {
    const std::vector<std::pair<const char*,bool(*)()>> tests={{"static",StaticPublicInvokeAndConstants},{"source",PublicSourceValidation},{"dynamic",DynamicGateGuardsScalarsAndPreallocation},{"budget",BudgetAndExtentsBeforeAllocationOrLaunch},{"contract",ContractAssemblyAndExpressionLimits},{"completion",InvalidCompletionIsRejected}};
    int failed=0; for(const auto& test:tests) try { if(!test.second()) ++failed; else std::cout<<"[PASS] "<<test.first<<"\n"; } catch(const std::exception& e) { std::cerr<<"[FAIL] "<<test.first<<": "<<e.what()<<"\n"; ++failed; } return failed?1:0;
}
