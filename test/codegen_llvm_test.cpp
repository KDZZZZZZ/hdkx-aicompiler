/*! \file test/codegen_llvm_test.cpp
 * \brief 验证 TIR/Relay/Compiler 到 LLVM NDArray Launch 的核心路径。
 */

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api/compiler.h"
#include "base/ndarray.h"
#include "codegen/codegen_c.h"
#include "relay/relay.h"
#include "relay/transforms/lower.h"
#include "runtime/runtime_session.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/llvm_jit.h"
#endif

namespace {

// 失败条件统一抛异常，main 会把任一失败转换为非零退出码。
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 测试统一使用 CPU float32 NDArray，避免重新引入裸主机参数 ABI。
kxc::runtime::NDArray FloatArray(kxc::Array<int64_t> shape,
                                 const std::vector<float>& values = {}) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        std::move(shape), kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    if (!values.empty()) {
        Require(values.size() * sizeof(float) == array.NBytes(),
                "host value count does not match NDArray shape");
        array.CopyFromBytes(values.data(), array.NBytes());
    }
    return array;
}

// 把 CPU NDArray 拷回独立主机向量供数值断言使用。
std::vector<float> ReadFloats(const kxc::runtime::NDArray& array) {
    std::vector<float> values(array.NBytes() / sizeof(float));
    array.CopyToBytes(values.data(), array.NBytes());
    return values;
}

// 逐元素比较结果，错误信息保留第一个失败位置。
void ExpectNear(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tolerance = 1e-5f) {
    Require(actual.size() == expected.size(), "result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("value mismatch at " + std::to_string(i));
        }
    }
}

// 手工构造 c[i] = a[i] + b[i] 的 TIR PrimFunc。
kxc::tir::PrimFunc MakeAddPrimFunc(const std::string& symbol) {
    using namespace kxc;
    using namespace kxc::tir;
    tir::Var a("a", DataType::Float(32));
    tir::Var b("b", DataType::Float(32));
    tir::Var c("c", DataType::Float(32));
    tir::Var i("i", DataType::Int(32));
    Stmt body = For(i, IntImm(0, DataType::Int(32)),
                    IntImm(8, DataType::Int(32)), ForType::Serial,
                    Store(c, Load(a, PrimExpr(i)) + Load(b, PrimExpr(i)), PrimExpr(i)));
    Array<tir::Var> params{a, b, c};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : params) {
        buffers.Set(parameter, Buffer(parameter, DataType::Float(32),
                                      {IntImm(8, DataType::Int(64))}, {},
                                      IntImm(0), parameter->name_hint, 0, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String(symbol));
    return PrimFunc(params, body, buffers, attrs);
}

#if KXC_USE_LLVM
// 编译手工 TIR，并通过强类型 CompiledKernel 启动 NDArray 参数。
void TestDirectLLVM() {
    using namespace kxc;
    using namespace kxc::codegen;
    const String symbol("direct_add");
    tir::PrimFunc function = MakeAddPrimFunc(symbol);
    auto context = std::make_unique<llvm::LLVMContext>();
    CodeGenLLVM codegen(*context);
    codegen.AddFunction(function, symbol);

    KernelArgSpec a("a", KernelArgRole::kInput,
                    runtime::DataTypeFromString("float32"), {8}, Device::CPU());
    KernelArgSpec b("b", KernelArgRole::kInput,
                    runtime::DataTypeFromString("float32"), {8}, Device::CPU());
    KernelArgSpec c("c", KernelArgRole::kOutput,
                    runtime::DataTypeFromString("float32"), {8}, Device::CPU(), 4, true);
    KernelSignature signature(symbol, {a, b, c});
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel kernel = LLVMJITEngine().Compile(
        codegen.TakeModule(), std::move(context), signature, metadata, 2);

    Array<runtime::NDArray> arguments{
        FloatArray({8}, {1, 2, 3, 4, 5, 6, 7, 8}),
        FloatArray({8}, {10, 20, 30, 40, 50, 60, 70, 80}),
        FloatArray({8}),
    };
    AsyncOperation operation = kernel.Launch(
        arguments, DeviceStream::Default(Device::CPU()));
    Require(operation.IsReady(), "LLVM CPU launch should complete inline");
    ExpectNear(ReadFloats(arguments[2]), {11, 22, 33, 44, 55, 66, 77, 88});
}

// Relay lowering 的签名必须与 JIT launcher 使用同一个 ObjectRef 契约。
void TestRelayLLVM() {
    using namespace kxc;
    using namespace kxc::codegen;
    Var x("x", TensorType({8}, "float32"));
    Var y("y", TensorType({8}, "float32"));
    Function relay_function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    relay::LoweredFunction lowered = relay::LowerToTIR(relay_function);
    const String symbol("relay_add");
    Map<String, runtime::NDArray> constants;
    Target target = BuildTarget(Device::CPU());
    KernelSignature signature = BuildKernelSignature(
        lowered->prim_func, constants, target, symbol);
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);

    auto context = std::make_unique<llvm::LLVMContext>();
    CodeGenLLVM codegen(*context);
    codegen.AddFunction(lowered->prim_func, symbol);
    CompiledKernel kernel = LLVMJITEngine().Compile(
        codegen.TakeModule(), std::move(context), signature, metadata, 1);
    Array<runtime::NDArray> arguments{
        FloatArray({8}, {1, 2, 3, 4, 5, 6, 7, 8}),
        FloatArray({8}, {1, 1, 1, 1, 1, 1, 1, 1}),
        FloatArray({8}),
    };
    kernel.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
    ExpectNear(ReadFloats(arguments[2]), {2, 3, 4, 5, 6, 7, 8, 9});
}

// Compiler 公共入口必须返回可通过 NDArray + DeviceStream 启动的模块。
void TestCompilerLLVM() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Function function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    api::CompiledModule module = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    Array<runtime::NDArray> arguments{
        FloatArray({4}, {100, 200, 300, 400}),
        FloatArray({4}, {1, 2, 3, 4}),
        FloatArray({4}),
    };
    module.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
    ExpectNear(ReadFloats(arguments[2]), {101, 202, 303, 404});
}

// 两级 Relay 计算验证 LLVM Allocate 中间存储仍保持正确。
void TestCompilerIntermediateAllocate() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Call first(relay::Op::Get("add"), {x, y});
    Function function({x, y}, Call(relay::Op::Get("add"), {first, y}));
    api::CompiledModule module = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    Array<runtime::NDArray> arguments{
        FloatArray({4}, {1, 2, 3, 4}),
        FloatArray({4}, {10, 20, 30, 40}),
        FloatArray({4}),
    };
    module.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
    ExpectNear(ReadFloats(arguments[2]), {21, 42, 63, 84});
}

/*! \brief RuntimeSession 只接收 inputs，并自动分配 add 输出。 */
void TestRuntimeSessionLLVM() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Function function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    runtime::RuntimeSession session(api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2)));
    Array<runtime::NDArray> inputs{
        FloatArray({4}, {100, 200, 300, 400}),
        FloatArray({4}, {1, 2, 3, 4}),
    };

    Array<runtime::NDArray> outputs = session.Run(inputs);
    Require(outputs.size() == 1,
            "RuntimeSession LLVM should allocate one output");
    ExpectNear(ReadFloats(outputs[0]), {101, 202, 303, 404});

    runtime::RunAsyncResult async_result = session.RunAsync(
        inputs, DeviceStream::Create(Device::CPU()));
    Require(async_result.outputs.size() == 1 &&
                async_result.completion.IsReady(),
            "LLVM RunAsync should return a completed operation and one output");
    async_result.completion.Wait();
    ExpectNear(ReadFloats(async_result.outputs[0]), {101, 202, 303, 404});
}

/*! \brief RuntimeSession 必须按 constant_key 自动插入 Compiler 持有的常量。 */
void TestRuntimeSessionLLVMConstant() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    runtime::NDArray constant_data =
        FloatArray({4}, {10, 20, 30, 40});
    Function function(
        {x}, Call(relay::Op::Get("add"), {x, Constant(constant_data)}));
    runtime::RuntimeSession session(api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2)));
    Array<runtime::NDArray> outputs =
        session.Run({FloatArray({4}, {1, 2, 3, 4})});
    Require(outputs.size() == 1,
            "constant RuntimeSession LLVM should allocate one output");
    ExpectNear(ReadFloats(outputs[0]), {11, 22, 33, 44});
}
#endif

// C emitter 仅作为诊断源码工具，测试不把它声明为可执行 backend。
void TestDiagnosticCSource() {
    kxc::codegen::CSourceEmitter emitter;
    const std::string source = emitter.Generate(MakeAddPrimFunc("diagnostic_add"),
                                                "diagnostic_add");
    Require(source.find("diagnostic_add") != std::string::npos &&
                source.find("return 0") != std::string::npos,
            "diagnostic C source is missing its entry contract");
}

}  // namespace

// 顺序执行 LLVM/C 诊断测试，并把任一异常转换为可靠失败退出码。
int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests = {
#if KXC_USE_LLVM
        {"direct_llvm", TestDirectLLVM},
        {"relay_llvm", TestRelayLLVM},
        {"compiler_llvm", TestCompilerLLVM},
        {"compiler_intermediate_allocate", TestCompilerIntermediateAllocate},
        {"runtime_session_llvm", TestRuntimeSessionLLVM},
        {"runtime_session_llvm_constant", TestRuntimeSessionLLVMConstant},
#endif
        {"diagnostic_c_source", TestDiagnosticCSource},
    };
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
