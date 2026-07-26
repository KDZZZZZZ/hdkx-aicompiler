/*! \file test/compiled_module_dynamic_llvm_test.cpp
 * \brief Real LLVM/JIT dynamic CompiledModule invocation ABI regression.
 */

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "../src/codegen/llvm/internal/codegen_llvm.h"
#include "../src/codegen/llvm/internal/llvm_jit.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {
using namespace kxc;
using namespace kxc::api;
using namespace kxc::codegen;
using runtime::NDArray;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

DLDataType F32() { return runtime::DataTypeFromString("float32"); }
DLDataType U64() { return {kDLUInt, 64, 1}; }

NDArray Floats(std::int64_t extent, const std::vector<float>& values = {}) {
    NDArray result = NDArray::Empty({extent}, F32(), Device::CPU(), 8);
    if (!values.empty()) result.CopyFromBytes(values.data(), result.NBytes());
    return result;
}

std::vector<float> ReadFloats(const NDArray& value) {
    std::vector<float> result(value.NBytes() / sizeof(float));
    value.CopyToBytes(result.data(), value.NBytes());
    return result;
}

/* The physical ABI is caller input, generated uint64[1] extent, output.
 * The scalar Load is the actual loop bound; each trip writes two outputs. */
tir::PrimFunc DynamicTwicePrimFunc(const String& symbol) {
    using namespace tir;
    const DataType f32 = DataType::Float(32);
    const DataType u64 = DataType::UInt(64);
    Var input("input", f32);
    Var extent("extent", u64);
    Var output("output", f32);
    Var i("i", u64);
    const PrimExpr scalar = Load(extent, IntImm(0, u64));
    const PrimExpr first = PrimExpr(i) * IntImm(2, u64);
    const PrimExpr input_value = Load(input, PrimExpr(i));
    const Stmt body = SeqStmt({
        Store(output, input_value, first),
        Store(output, input_value + FloatImm(1.0, f32),
              first + IntImm(1, u64)),
    });
    const Stmt loop = For(i, IntImm(0, u64), scalar, ForType::Serial, body);
    Array<Var> params{input, extent, output};
    Map<Var, Buffer> buffers;
    buffers.Set(input, Buffer(input, f32, {IntImm(3, DataType::Int(64))}, {},
                              IntImm(0), "input", 0, 0));
    buffers.Set(extent, Buffer(extent, u64, {IntImm(1, DataType::Int(64))}, {},
                               IntImm(0), "extent", 0, 0));
    buffers.Set(output, Buffer(output, f32, {IntImm(6, DataType::Int(64))}, {},
                               IntImm(0), "output", 0, 0));
    Map<String, ObjectRef> attrs;
    attrs.Set("global_symbol", symbol);
    return PrimFunc(params, loop, buffers, attrs);
}

CompiledModule BuildDynamicLLVMModule() {
    const String symbol("llvm_dynamic_twice");
    const KernelSignature signature(symbol, {
        KernelArgSpec("input", KernelArgRole::kInput, F32(), {-1}, Device::CPU(), 8),
        KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, U64(), {1}, Device::CPU(), 8),
        KernelArgSpec("output", KernelArgRole::kOutput, F32(), {-1}, Device::CPU(), 8,
                      true),
    });
    const KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    auto context = std::make_unique<llvm::LLVMContext>();
    CodeGenLLVM codegen(*context);
    const tir::PrimFunc function = DynamicTwicePrimFunc(symbol);
    codegen.AddFunction(function, symbol);
    const CompiledKernel kernel = LLVMJITEngine().Compile(
        codegen.TakeModule(), std::move(context), signature, metadata, 0);

    const ModuleShapeExpr twice = ModuleShapeExpr::Mul(
        ModuleShapeExpr::InputAxis(0, 0), ModuleShapeExpr::Const(2));
    ModuleInputContract input{{{0, 0, 3, 1, std::nullopt, std::nullopt}}};
    ModuleTensorContract output;
    output.logical = {twice};
    output.physical = {twice};
    output.valid = {twice};
    output.max_bytes = 6 * sizeof(float);
    auto contract = std::make_shared<ModuleInvocationContract>(
        std::vector<ModuleInputContract>{input},
        std::vector<ModuleTensorContract>{output},
        std::vector<ModuleRuntimeExtentScalar>{{
            ModuleShapeExpr::InputAxis(0, 0)}});
    return internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {{signature, metadata, kernel, std::move(contract)}}, {});
}

void TestDynamicInvokeUsesGeneratedScalarInLLVM() {
    const CompiledModule module = BuildDynamicLLVMModule();
    const String symbol("llvm_dynamic_twice");
    const DeviceStream stream = DeviceStream::Default(Device::CPU());
    Require(module.IsReady(), "real LLVM module is not ready");

    for (const std::int64_t extent : {0, 2, 3}) {
        std::vector<float> input;
        for (std::int64_t i = 0; i < extent; ++i) input.push_back(10.0F + i);
        const ModuleInvocationResult result = module.Invoke(symbol, {Floats(extent, input)}, stream);
        result.operation.Wait();
        Require(result.outputs.size() == 1, "dynamic invocation returned wrong output count");
        const auto& output = result.outputs[0];
        const std::vector<ModuleExtent> expected_extent{static_cast<ModuleExtent>(2 * extent)};
        Require(output.logical == expected_extent && output.physical == expected_extent &&
                    output.valid == expected_extent,
                "dynamic invocation extent descriptors are incorrect");
        std::vector<float> expected;
        for (float value : input) {
            expected.push_back(value);
            expected.push_back(value + 1.0F);
        }
        const std::vector<float> actual = ReadFloats(output.storage);
        Require(actual == expected, "LLVM output did not consume generated scalar as loop bound");
    }

    bool guard_rejected = false;
    try {
        (void)module.Invoke(symbol, {Floats(4, {1, 2, 3, 4})}, stream);
    } catch (const ModuleInvocationError& error) {
        guard_rejected = error.kind() == ModuleInvocationFailureKind::kGuard;
    }
    Require(guard_rejected, "guard miss was not rejected before LLVM launch/allocation");
}
}  // namespace

int main() {
    try {
        TestDynamicInvokeUsesGeneratedScalarInLLVM();
        std::cout << "[PASS] dynamic LLVM CompiledModule Invoke\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
