#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {
using namespace kxc;
using namespace kxc::api;
using namespace kxc::codegen;
using runtime::NDArray;

#define CHECK(x) do { if (!(x)) { std::cerr << #x << " failed\n"; return false; } } while (false)
DLDataType F32() { return runtime::DataTypeFromString("float32"); }
class Launcher final : public KernelLauncher {
public:
 bool IsReady() const noexcept override { return true; }
 AsyncOperation Launch(const Array<NDArray>& args, const DeviceStream& stream, const ObjectRef&) const override {
   ++calls; seen=args; Array<Storage> keep; for(const auto& a:args) keep.push_back(a.storage()); return AsyncOperation::Completed(stream, std::move(keep));
 }
 mutable int calls{0}; mutable Array<NDArray> seen;
};
CompiledModule Make(bool dynamic, std::shared_ptr<Launcher>* launcher) {
 const Device cpu=Device::CPU();
 KernelArgSpec in("x",KernelArgRole::kInput,F32(),{codegen::kDynamicDimension},cpu,1,false);
 KernelArgSpec out("y",KernelArgRole::kOutput,F32(),{dynamic?codegen::kDynamicDimension:4},cpu,1,true);
 KernelSignature sig("entry",{in,out}); KernelLaunchMetadata md(cpu,CodeGenBackend::kLLVM);
 *launcher=std::make_shared<Launcher>(); CompiledKernel kernel(sig,md,*launcher);
 internal::CompiledModuleEntry entry{tir::PrimFunc(),sig,md,kernel};
 if(dynamic) {
   ModuleInputContract input{F32(),cpu,1,{ModuleAxisGuard{0,0,1024,2,std::nullopt,std::nullopt}}};
   ModuleTensorContract output; output.dtype=F32(); output.device=cpu; output.max_bytes=4096;
   output.logical={ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0,0),ModuleShapeExpr::Const(2))};
   output.physical=output.logical; output.valid=output.logical;
   entry.invocation_contract=std::make_shared<ModuleInvocationContract>(std::vector<ModuleInputContract>{input},std::vector<ModuleTensorContract>{output},4096);
 }
 return internal::BuildCompiledModule(BuildTarget(cpu),{entry},{});
}
bool StaticAutoContractAndInvoke() {
 std::shared_ptr<Launcher> launcher; CompiledModule module=Make(false,&launcher);
 CHECK(module.invocation_contract("entry").IsConstantShape());
 NDArray input=NDArray::Zeros({4},F32(),Device::CPU());
 auto result=module.Invoke("entry",{input},DeviceStream::Default(Device::CPU()));
 CHECK(result.outputs.size()==1 && result.outputs[0].physical==std::vector<ModuleExtent>{4});
 CHECK(launcher->calls==1 && launcher->seen.size()==2);
 return true;
}
bool DynamicGateAndGuards() {
 std::shared_ptr<Launcher> launcher; CompiledModule module=Make(true,&launcher);
 NDArray good=NDArray::Zeros({2},F32(),Device::CPU());
#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
 auto result=module.Invoke("entry",{good},DeviceStream::Default(Device::CPU()));
 CHECK(result.outputs[0].logical==std::vector<ModuleExtent>{4} && launcher->calls==1);
 NDArray bad=NDArray::Zeros({3},F32(),Device::CPU());
 try { module.Invoke("entry",{bad},DeviceStream::Default(Device::CPU())); return false; } catch(const ModuleInvocationError& e) { CHECK(e.kind()==ModuleInvocationFailureKind::kGuard); }
#else
 try { module.Invoke("entry",{good},DeviceStream::Default(Device::CPU())); return false; } catch(const ModuleInvocationError& e) { CHECK(e.kind()==ModuleInvocationFailureKind::kDisabled); }
 CHECK(launcher->calls==0);
#endif
 return true;
}
}
int main() { return StaticAutoContractAndInvoke() && DynamicGateAndGuards() ? 0 : 1; }
