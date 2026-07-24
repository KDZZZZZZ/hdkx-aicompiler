/*! \file test/codegen_llvm_test.cpp
 * \brief 验证 TIR/Relay/Compiler 到 LLVM NDArray Launch 的核心路径。
 */

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/runtime/ndarray.h"
#include "../src/codegen/c/internal/codegen_c.h"
#include "kxc/relay/relay.h"
#include "kxc/relay/op.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/runtime/session.h"

#if KXC_USE_LLVM
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include "../src/codegen/llvm/internal/codegen_llvm.h"
#include "../src/compiler/internal/kernel_abi_builder.h"
#include "../src/codegen/llvm/internal/llvm_jit.h"
#endif

namespace {

// 失败条件统一抛异常，main 会把任一失败转换为非零退出码。
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 负例同时校验异常和关键诊断，避免错误路径因其他原因“碰巧失败”。
template <typename Fn>
void RequireThrowsContaining(Fn&& function, const std::string& expected) {
    try {
        function();
    } catch (const std::exception& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos,
                "unexpected diagnostic: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected exception containing: " + expected);
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

// int32 ABI 使用独立辅助函数，避免通过 float 转换掩盖整数 load/store 错误。
kxc::runtime::NDArray IntArray(kxc::Array<int64_t> shape,
                               const std::vector<int32_t>& values = {}) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        std::move(shape), kxc::runtime::DataTypeFromString("int32"),
        kxc::Device::CPU());
    if (!values.empty()) {
        Require(values.size() * sizeof(int32_t) == array.NBytes(),
                "host int value count does not match NDArray shape");
        array.CopyFromBytes(values.data(), array.NBytes());
    }
    return array;
}

// bool NDArray 的宿主表示是一字节 0/1，不能使用没有连续 data() 的 vector<bool>。
kxc::runtime::NDArray BoolArray(kxc::Array<int64_t> shape,
                                const std::vector<uint8_t>& values = {}) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        std::move(shape), kxc::runtime::DataTypeFromString("bool"),
        kxc::Device::CPU());
    if (!values.empty()) {
        Require(values.size() == array.NBytes(),
                "host bool value count does not match NDArray shape");
        array.CopyFromBytes(values.data(), array.NBytes());
    }
    return array;
}

// 从 CPU NDArray 读取任意平凡标量序列，供 int32/bool ABI 断言复用。
template <typename T>
std::vector<T> ReadScalars(const kxc::runtime::NDArray& array) {
    Require(array.NBytes() % sizeof(T) == 0,
            "NDArray byte size is not divisible by scalar size");
    std::vector<T> values(array.NBytes() / sizeof(T));
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
kxc::tir::PrimFunc MakeAddPrimFunc(const std::string& symbol, int64_t extent = 8) {
    using namespace kxc;
    using namespace kxc::tir;
    tir::Var a("a", DataType::Float(32));
    tir::Var b("b", DataType::Float(32));
    tir::Var c("c", DataType::Float(32));
    tir::Var i("i", DataType::Int(32));
    Stmt body = For(i, IntImm(0, DataType::Int(32)),
                    IntImm(extent, DataType::Int(32)), ForType::Serial,
                    Store(c, Load(a, PrimExpr(i)) + Load(b, PrimExpr(i)), PrimExpr(i)));
    Array<tir::Var> params{a, b, c};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : params) {
        buffers.Set(parameter, Buffer(parameter, DataType::Float(32),
                                      {IntImm(extent, DataType::Int(64))}, {},
                                      IntImm(0), parameter->name_hint, 0, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String(symbol));
    return PrimFunc(params, body, buffers, attrs);
}

// 构造同一循环写两个输出的 TIR，用于验证 LLVM call frame 的完整参数顺序。
kxc::tir::PrimFunc MakeTwoOutputPrimFunc(const std::string& symbol,
                                         int64_t extent) {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType f32 = DataType::Float(32);
    const DataType i32 = DataType::Int(32);
    tir::Var a("a", f32);
    tir::Var b("b", f32);
    tir::Var sum("sum", f32);
    tir::Var difference("difference", f32);
    tir::Var i("i", i32);
    Stmt body = For(
        i, IntImm(0, i32), IntImm(extent, i32), ForType::Serial,
        SeqStmt({Store(sum, Load(a, i) + Load(b, i), i),
                 Store(difference, Load(a, i) - Load(b, i), i)}));
    Array<tir::Var> params{a, b, sum, difference};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : params) {
        buffers.Set(parameter, Buffer(parameter, f32,
                                      {IntImm(extent, DataType::Int(64))}, {},
                                      IntImm(0), parameter->name_hint, 0, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String(symbol));
    return PrimFunc(params, body, buffers, attrs);
}

// 构造保持 dtype 不变的逐元素复制，覆盖 float 之外的标量 ABI 和零尺寸循环。
kxc::tir::PrimFunc MakeCopyPrimFunc(const std::string& symbol,
                                    kxc::tir::DataType dtype,
                                    int64_t extent) {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType i32 = DataType::Int(32);
    tir::Var input("input", dtype);
    tir::Var output("output", dtype);
    tir::Var i("i", i32);
    Stmt body = For(i, IntImm(0, i32), IntImm(extent, i32), ForType::Serial,
                    Store(output, Load(input, i), i));
    Array<tir::Var> params{input, output};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : params) {
        buffers.Set(parameter, Buffer(parameter, dtype,
                                      {IntImm(extent, DataType::Int(64))}, {},
                                      IntImm(0), parameter->name_hint, 0, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String(symbol));
    return PrimFunc(params, body, buffers, attrs);
}

#if KXC_USE_LLVM
// 把手工 TIR 编译为独占 ORC 资源的内核，调用方只接触强类型句柄。
kxc::codegen::CompiledKernel CompileLLVM(
    const kxc::tir::PrimFunc& function,
    const kxc::codegen::KernelSignature& signature,
    int opt_level = 2) {
    auto context = std::make_unique<llvm::LLVMContext>();
    kxc::codegen::CodeGenLLVM codegen(*context);
    codegen.AddFunction(function, signature->symbol);
    return kxc::codegen::LLVMJITEngine().Compile(
        codegen.TakeModule(), std::move(context), signature,
        kxc::codegen::KernelLaunchMetadata(
            kxc::Device::CPU(), kxc::codegen::CodeGenBackend::kLLVM),
        opt_level);
}

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

// 两个输出必须各自占用稳定 ABI slot，不能被错误别名为第一个输出。
void TestLLVMMultipleOutputs() {
    using namespace kxc;
    using namespace kxc::codegen;
    constexpr int64_t kExtent = 4;
    const String symbol("two_outputs");
    const DLDataType f32 = runtime::DataTypeFromString("float32");
    KernelSignature signature(
        symbol,
        {KernelArgSpec("a", KernelArgRole::kInput, f32, {kExtent}, Device::CPU()),
         KernelArgSpec("b", KernelArgRole::kInput, f32, {kExtent}, Device::CPU()),
         KernelArgSpec("sum", KernelArgRole::kOutput, f32, {kExtent},
                       Device::CPU(), 4, true),
         KernelArgSpec("difference", KernelArgRole::kOutput, f32, {kExtent},
                       Device::CPU(), 4, true)});
    CompiledKernel kernel = CompileLLVM(
        MakeTwoOutputPrimFunc(symbol, kExtent), signature);
    Array<runtime::NDArray> arguments{
        FloatArray({kExtent}, {5, 7, 11, 13}),
        FloatArray({kExtent}, {1, 2, 3, 4}),
        FloatArray({kExtent}),
        FloatArray({kExtent}),
    };
    kernel.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
    ExpectNear(ReadFloats(arguments[2]), {6, 9, 14, 17});
    ExpectNear(ReadFloats(arguments[3]), {4, 5, 8, 9});
}

// LLVM 后端必须正确解释 int32、bool 和零尺寸 NDArray 的实际字节布局。
void TestLLVMScalarDTypesAndZeroSize() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();

    const DLDataType i32 = runtime::DataTypeFromString("int32");
    KernelSignature int_signature(
        "copy_i32",
        {KernelArgSpec("input", KernelArgRole::kInput, i32, {4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, i32, {4}, cpu, 4, true)});
    CompiledKernel int_kernel = CompileLLVM(
        MakeCopyPrimFunc("copy_i32", tir::DataType::Int(32), 4), int_signature);
    Array<runtime::NDArray> int_arguments{
        IntArray({4}, {-7, 0, 42, 100000}), IntArray({4})};
    int_kernel.Launch(int_arguments, DeviceStream::Default(cpu)).Wait();
    Require(ReadScalars<int32_t>(int_arguments[1]) ==
                std::vector<int32_t>({-7, 0, 42, 100000}),
            "LLVM int32 copy result mismatch");

    const DLDataType boolean = runtime::DataTypeFromString("bool");
    KernelSignature bool_signature(
        "copy_bool",
        {KernelArgSpec("input", KernelArgRole::kInput, boolean, {4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, boolean, {4}, cpu, 1, true)});
    CompiledKernel bool_kernel = CompileLLVM(
        MakeCopyPrimFunc("copy_bool", tir::DataType::Bool(), 4), bool_signature);
    Array<runtime::NDArray> bool_arguments{
        BoolArray({4}, {1, 0, 1, 1}), BoolArray({4})};
    bool_kernel.Launch(bool_arguments, DeviceStream::Default(cpu)).Wait();
    Require(ReadScalars<uint8_t>(bool_arguments[1]) ==
                std::vector<uint8_t>({1, 0, 1, 1}),
            "LLVM bool copy result mismatch");

    const DLDataType f32 = runtime::DataTypeFromString("float32");
    KernelSignature zero_signature(
        "copy_zero",
        {KernelArgSpec("input", KernelArgRole::kInput, f32, {0}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, f32, {0}, cpu, 4, true)});
    CompiledKernel zero_kernel = CompileLLVM(
        MakeCopyPrimFunc("copy_zero", tir::DataType::Float(32), 0), zero_signature,
        0);
    Array<runtime::NDArray> zero_arguments{FloatArray({0}), FloatArray({0})};
    AsyncOperation zero_operation = zero_kernel.Launch(
        zero_arguments, DeviceStream::Default(cpu));
    Require(zero_operation.IsReady() && zero_arguments[1].NBytes() == 0,
            "LLVM zero-size launch did not complete without dereferencing null data");
}

// launcher 形成参数地址时必须叠加 NDArray byte_offset，并保持视图外哨兵不变。
void TestLLVMByteOffset() {
    using namespace kxc;
    using namespace kxc::codegen;
    constexpr int64_t kExtent = 4;
    const DLDataType f32 = runtime::DataTypeFromString("float32");
    KernelSignature signature(
        "offset_add",
        {KernelArgSpec("a", KernelArgRole::kInput, f32, {kExtent}, Device::CPU()),
         KernelArgSpec("b", KernelArgRole::kInput, f32, {kExtent}, Device::CPU()),
         KernelArgSpec("output", KernelArgRole::kOutput, f32, {kExtent},
                       Device::CPU(), 4, true)});
    CompiledKernel kernel = CompileLLVM(
        MakeAddPrimFunc("offset_add", kExtent), signature);

    runtime::NDArray a_backing = FloatArray({6}, {-100, 1, 2, 3, 4, -101});
    runtime::NDArray b_backing = FloatArray({6}, {-200, 10, 20, 30, 40, -201});
    runtime::NDArray output_backing = FloatArray({6}, {-300, 0, 0, 0, 0, -301});
    Array<runtime::NDArray> arguments{
        a_backing.CreateView({kExtent}, {1}, sizeof(float)),
        b_backing.CreateView({kExtent}, {1}, sizeof(float)),
        output_backing.CreateView({kExtent}, {1}, sizeof(float)),
    };
    kernel.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
    ExpectNear(ReadFloats(arguments[2]), {11, 22, 33, 44});
    ExpectNear(ReadFloats(output_backing), {-300, 11, 22, 33, 44, -301});
}

// verifier、symbol lookup 和 opt-level 边界必须分别给出可区分的失败原因。
void TestLLVMValidationAndLookupErrors() {
    using namespace kxc;
    using namespace kxc::codegen;
    LLVMJITEngine engine;
    for (int opt_level = 0; opt_level <= 3; ++opt_level) {
        auto context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>("opt_level", *context);
        engine.Optimize(module.get(), opt_level);
    }
    auto invalid_opt = [&](int opt_level) {
        auto context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>("invalid_opt", *context);
        engine.Optimize(module.get(), opt_level);
    };
    RequireThrowsContaining([&] { invalid_opt(-1); }, "opt_level");
    RequireThrowsContaining([&] { invalid_opt(4); }, "opt_level");

    const DLDataType f32 = runtime::DataTypeFromString("float32");
    KernelSignature invalid_signature(
        "invalid_module",
        {KernelArgSpec("output", KernelArgRole::kOutput, f32, {1},
                       Device::CPU(), 4, true)});
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    auto invalid_context = std::make_unique<llvm::LLVMContext>();
    auto invalid_module = std::make_unique<llvm::Module>(
        "invalid_module", *invalid_context);
    llvm::FunctionType* function_type = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(*invalid_context), false);
    llvm::Function* invalid_function = llvm::Function::Create(
        function_type, llvm::Function::ExternalLinkage, "invalid_module",
        invalid_module.get());
    (void)llvm::BasicBlock::Create(*invalid_context, "entry", invalid_function);
    RequireThrowsContaining(
        [&] {
            (void)engine.Compile(std::move(invalid_module),
                                 std::move(invalid_context), invalid_signature,
                                 metadata, 0);
        },
        "verification failed before optimization");

    const String actual_symbol("actual_symbol");
    auto lookup_context = std::make_unique<llvm::LLVMContext>();
    CodeGenLLVM codegen(*lookup_context);
    codegen.AddFunction(MakeAddPrimFunc(actual_symbol), actual_symbol);
    KernelSignature missing_signature(
        "missing_symbol",
        {KernelArgSpec("a", KernelArgRole::kInput, f32, {8}, Device::CPU()),
         KernelArgSpec("b", KernelArgRole::kInput, f32, {8}, Device::CPU()),
         KernelArgSpec("output", KernelArgRole::kOutput, f32, {8},
                       Device::CPU(), 4, true)});
    RequireThrowsContaining(
        [&] {
            (void)engine.Compile(codegen.TakeModule(), std::move(lookup_context),
                                 missing_signature, metadata, 0);
        },
        "Failed to lookup LLVM function 'missing_symbol'");
}

// 最后一个 CompiledKernel 引用释放后，持有 ORC LLJIT 的 launcher 必须同步析构。
void TestLLVMJITLifetime() {
    using namespace kxc;
    using namespace kxc::codegen;
    std::weak_ptr<const KernelLauncher> launcher;
    {
        const DLDataType f32 = runtime::DataTypeFromString("float32");
        KernelSignature signature(
            "lifetime_add",
            {KernelArgSpec("a", KernelArgRole::kInput, f32, {2}, Device::CPU()),
             KernelArgSpec("b", KernelArgRole::kInput, f32, {2}, Device::CPU()),
             KernelArgSpec("output", KernelArgRole::kOutput, f32, {2},
                           Device::CPU(), 4, true)});
        CompiledKernel kernel = CompileLLVM(
            MakeAddPrimFunc("lifetime_add", 2), signature);
        launcher = kernel->launcher;
        Array<runtime::NDArray> arguments{
            FloatArray({2}, {1, 2}), FloatArray({2}, {3, 4}), FloatArray({2})};
        kernel.Launch(arguments, DeviceStream::Default(Device::CPU())).Wait();
        ExpectNear(ReadFloats(arguments[2]), {4, 6});
        Require(!launcher.expired(), "LLVM launcher expired while kernel was alive");
    }
    Require(launcher.expired(),
            "LLVM launcher retained ORC JIT after the last kernel reference");
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

void TestLLVMBatchModule() {
    using namespace kxc;
    using namespace kxc::codegen;
    const DLDataType f32 = runtime::DataTypeFromString("float32");
    const auto make_signature = [&](const char* symbol) {
        return KernelSignature(
            symbol,
            {KernelArgSpec("a", KernelArgRole::kInput, f32, {4},
                           Device::CPU()),
             KernelArgSpec("b", KernelArgRole::kInput, f32, {4},
                           Device::CPU()),
             KernelArgSpec("out", KernelArgRole::kOutput, f32, {4},
                           Device::CPU(), 4, true)});
    };
    std::vector<KernelSignature> signatures{
        make_signature("batch_add_a"), make_signature("batch_add_b")};
    std::vector<KernelLaunchMetadata> metadata{
        KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kLLVM),
        KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kLLVM)};
    auto context = std::make_unique<llvm::LLVMContext>();
    CodeGenLLVM codegen(*context);
    codegen.AddFunctions({{MakeAddPrimFunc("batch_add_a", 4), "batch_add_a"},
                          {MakeAddPrimFunc("batch_add_b", 4), "batch_add_b"}});
    std::vector<CompiledKernel> kernels = LLVMJITEngine().CompileMany(
        codegen.TakeModule(), std::move(context), signatures, metadata, 2);
    Require(kernels.size() == 2 && kernels[0].IsReady() &&
                kernels[1].IsReady(),
            "LLVM batch compilation must return one kernel per symbol");
    kernels.erase(kernels.begin());
    Array<runtime::NDArray> arguments{
        FloatArray({4}, {1, 2, 3, 4}),
        FloatArray({4}, {10, 20, 30, 40}), FloatArray({4})};
    kernels[0]
        .Launch(arguments, DeviceStream::Default(Device::CPU()))
        .Wait();
    ExpectNear(ReadFloats(arguments[2]), {11, 22, 33, 44});
}

// Compiler 公共入口必须返回可通过 NDArray + DeviceStream 启动的模块。
void TestCompilerLLVM() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Function function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    api::CompiledGraph compiled = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    Array<runtime::NDArray> arguments{
        FloatArray({4}, {100, 200, 300, 400}),
        FloatArray({4}, {1, 2, 3, 4}),
        FloatArray({4}),
    };
    Require(compiled.plan.calls().size() == 1 &&
                compiled.module.entry_count() == 1,
            "single Relay add must compile to one explicit entry");
    auto result = compiled.module.Invoke(
        compiled.plan.calls()[0]->symbol, {arguments[0], arguments[1]},
        DeviceStream::Default(Device::CPU()));
    result.operation.Wait();
    ExpectNear(ReadFloats(result.outputs[0].storage), {101, 202, 303, 404});
}

// 两级 Relay 计算验证 LLVM Allocate 中间存储仍保持正确。
void TestCompilerIntermediateAllocate() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Call first(relay::Op::Get("add"), {x, y});
    Function function({x, y}, Call(relay::Op::Get("add"), {first, y}));
    api::CompiledGraph compiled = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    Array<runtime::NDArray> outputs = session.Run({
        FloatArray({4}, {1, 2, 3, 4}),
        FloatArray({4}, {10, 20, 30, 40}),
    });
    Require(compiled.module.entry_count() == 2 &&
                compiled.plan.calls().size() == 2 && outputs.size() == 1,
            "two Relay calls must compile and execute as two entries");
    ExpectNear(ReadFloats(outputs[0]), {21, 42, 63, 84});
}

/*! \brief RuntimeSession 只接收 inputs，并自动分配 add 输出。 */
void TestRuntimeSessionLLVM() {
    using namespace kxc;
    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Function function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    api::CompiledGraph compiled = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    runtime::RuntimeSession session(compiled.module, compiled.plan);
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
    api::CompiledGraph compiled = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    runtime::RuntimeSession session(compiled.module, compiled.plan);
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
        {"llvm_multiple_outputs", TestLLVMMultipleOutputs},
        {"llvm_scalar_dtypes_and_zero_size", TestLLVMScalarDTypesAndZeroSize},
        {"llvm_byte_offset", TestLLVMByteOffset},
        {"llvm_validation_and_lookup_errors", TestLLVMValidationAndLookupErrors},
        {"llvm_jit_lifetime", TestLLVMJITLifetime},
        {"relay_llvm", TestRelayLLVM},
        {"llvm_batch_module", TestLLVMBatchModule},
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
