/*! \file test/codegen_cuda_test.cpp
 * \brief 验证线程绑定 CUDA 源码、NVRTC/Driver module 和异步 NDArray 启动契约。
 */

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/ndarray.h"
#include "../src/codegen/cuda/internal/codegen_cuda.h"
#include "../src/codegen/cuda/internal/cuda_module.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

namespace {

/*! \brief 将任意失败条件转换为带上下文的测试异常。 */
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

/*! \brief 构造 threadIdx.x 驱动的 c[i] = a[i] + b[i] CUDA TIR。 */
kxc::tir::PrimFunc MakeBoundAdd(const std::string& symbol, int64_t extent = 8) {
    using namespace kxc;
    using namespace kxc::tir;
    tir::Var a("a", DataType::Float(32));
    tir::Var b("b", DataType::Float(32));
    tir::Var c("c", DataType::Float(32));
    tir::Var thread("tx", DataType::Int(32));
    Stmt body = ThreadBinding(
        thread, ThreadIndexKind::kThreadIdxX, IntImm(extent, DataType::Int(32)),
        Store(c, Load(a, PrimExpr(thread)) + Load(b, PrimExpr(thread)),
              PrimExpr(thread)));
    Array<tir::Var> parameters{a, b, c};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : parameters) {
        buffers.Set(parameter,
                    Buffer(parameter, DataType::Float(32),
                           {IntImm(extent, DataType::Int(64))}, {}, IntImm(0),
                           parameter->name_hint, 0, 0));
    }
    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String(symbol));
    return PrimFunc(parameters, body, buffers, attrs);
}

/*! \brief 为三个 float32 CUDA buffer 构造与 TIR 参数顺序一致的签名。 */
kxc::codegen::KernelSignature AddSignature(const std::string& symbol,
                                            const kxc::Device& device,
                                            int64_t extent = 8) {
    using namespace kxc;
    using namespace kxc::codegen;
    const DLDataType dtype = runtime::DataTypeFromString("float32");
    return KernelSignature(
        String(symbol),
        {KernelArgSpec("a", KernelArgRole::kInput, dtype, {extent}, device),
         KernelArgSpec("b", KernelArgRole::kInput, dtype, {extent}, device),
         KernelArgSpec("c", KernelArgRole::kOutput, dtype, {extent}, device,
                       4, true)});
}

/*! \brief 源码发射必须保留入口 ABI、typed pointer 和结构化线程 builtin。 */
void TestSourceEmission() {
    kxc::codegen::CodeGenCUDA emitter;
    const std::string source = emitter.Generate(MakeBoundAdd("bound_add"),
                                                "bound_add");
    Require(source.find("extern \"C\" __global__ void bound_add") !=
                std::string::npos,
            "CUDA source is missing global entry");
    Require(source.find("float* __restrict__ a") != std::string::npos,
            "CUDA source is missing typed buffer parameter");
    Require(source.find("threadIdx.x") != std::string::npos,
            "CUDA source is missing structured thread binding");
    Require(source.find("c[tx] = (a[tx] + b[tx])") != std::string::npos,
            "CUDA source is missing add store");
}

kxc::tir::PrimFunc MakeSpecialValues(kxc::tir::DataType dtype) {
    using namespace kxc;
    using namespace kxc::tir;
    tir::Var output("output", dtype), thread("tx", DataType::Int(32));
    const PrimExpr value = Select(EQ(thread, IntImm(0)),
        FloatImm(std::numeric_limits<double>::quiet_NaN(), dtype),
        Select(EQ(thread, IntImm(1)), FloatImm(std::numeric_limits<double>::infinity(), dtype),
               Select(EQ(thread, IntImm(2)), FloatImm(-std::numeric_limits<double>::infinity(), dtype),
                      FloatImm(-0.0, dtype))));
    Map<tir::Var, Buffer> buffers;
    buffers.Set(output, Buffer(output, dtype, {IntImm(4)}, {}, IntImm(0), "output", 0, 0));
    return PrimFunc({output}, ThreadBinding(thread, ThreadIndexKind::kThreadIdxX, IntImm(4),
                                           Store(output, value, thread)), buffers);
}

void TestSpecialValueSource() {
    for (int bits : {32, 64}) {
        const auto source = kxc::codegen::CodeGenCUDA().Generate(
            MakeSpecialValues(kxc::tir::DataType::Float(bits)), "special_values");
        Require(source.find("INFINITY") == std::string::npos && source.find("NAN") == std::string::npos &&
                source.find(bits == 32 ? "__int_as_float" : "__longlong_as_double") != std::string::npos,
                "special literals must be self-contained CUDA expressions");
    }
}

void TestMathCallSource() {
    using namespace kxc;
    using namespace kxc::tir;
    for (int bits : {32, 64}) {
        const auto dtype = DataType::Float(bits);
        const auto base = MakeSpecialValues(dtype);
        tir::Var thread("tx", DataType::Int(32));
        const auto emit = [&](const PrimExpr& value) {
            return codegen::CodeGenCUDA().Generate(PrimFunc(base->params,
                ThreadBinding(thread, ThreadIndexKind::kThreadIdxX, IntImm(4),
                    Store(base->params[0], value, thread)), base->buffer_map), "math_calls");
        };
        for (const std::string& name : {"exp", "sqrt", "pow"}) {
            for (const std::string& prefix : {"", "tir."}) {
                Array<PrimExpr> arguments{FloatImm(1.0, dtype)};
                if (name == "pow") arguments.push_back(FloatImm(2.0, dtype));
                const auto source = emit(tir::Call(dtype, prefix + name, arguments));
                Require(source.find(name + (bits == 32 ? "f(" : "(")) != std::string::npos,
                        "CUDA must accept the existing TE and TIR math spellings");
            }
        }
        bool rejected = false;
        try { (void)emit(tir::Call(dtype, "exp", {})); }
        catch (const std::runtime_error&) { rejected = true; }
        Require(rejected, "CUDA math call with invalid arity must be rejected");
        for (const auto& arguments : {Array<PrimExpr>{FloatImm(1.0, dtype)},
                Array<PrimExpr>{FloatImm(1.0, dtype), IntImm(2)}}) {
            rejected = false;
            try { (void)emit(tir::Call(dtype, "pow", arguments)); }
            catch (const std::runtime_error&) { rejected = true; }
            Require(rejected, "CUDA pow requires exactly two matching floating-point arguments");
        }
    }
}

/*! \brief 未支持的 Block 不能降级为注释占位或静默空内核。 */
void TestUnsupportedTIRRejected() {
    using namespace kxc;
    using namespace kxc::tir;
    const Stmt unsupported = Block({}, {}, {}, "unsupported", Evaluate(IntImm(0)));
    const PrimFunc function({}, unsupported);
    bool rejected = false;
    try {
        (void)codegen::CodeGenCUDA().Generate(function, "unsupported");
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("unsupported statement") !=
                   std::string::npos;
    }
    Require(rejected, "CodeGenCUDA accepted an unsupported TIR node");
}

/*! \brief CUDA emitter 不能绕过调度直接接受仍含 serial loop 的 PrimFunc。 */
void TestUnboundTIRRejected() {
    using namespace kxc;
    using namespace kxc::tir;
    tir::Var value("value", DataType::Float(32));
    tir::Var output("output", DataType::Float(32));
    tir::Var index("i", DataType::Int(32));
    Stmt body = For(index, IntImm(0, DataType::Int(32)),
                    IntImm(8, DataType::Int(32)), ForType::Serial,
                    Store(output, Load(value, PrimExpr(index)), PrimExpr(index)));
    Array<tir::Var> parameters{value, output};
    Map<tir::Var, Buffer> buffers;
    for (const auto& parameter : parameters) {
        buffers.Set(parameter,
                    Buffer(parameter, DataType::Float(32),
                           {IntImm(8, DataType::Int(64))}, {}, IntImm(0),
                           parameter->name_hint, 0, 0));
    }
    bool rejected = false;
    try {
        (void)codegen::CodeGenCUDA().Generate(
            PrimFunc(parameters, body, buffers), "unbound");
    } catch (const std::exception& error) {
        rejected = std::string(error.what()).find("thread-bound") !=
                   std::string::npos;
    }
    Require(rejected, "CodeGenCUDA accepted an unbound serial PrimFunc");
}

#if KXC_USE_CUDA

/*! \brief 查询当前 GPU 对应的 NVRTC virtual architecture，例如 compute_75。 */
kxc::codegen::CUDACompileOptions CompileOptions(const kxc::Device& device) {
    const kxc::DeviceAttributes attrs = kxc::CollectDeviceAttributes(device);
    Require(attrs.exists != 0, "CUDA device is unavailable");
    kxc::codegen::CUDACompileOptions options;
    options.architecture = "compute_" +
                           std::to_string(attrs.compute_version_major) +
                           std::to_string(attrs.compute_version_minor);
    options.source_name = "codegen_cuda_test.cu";
    return options;
}

template <typename Scalar>
void TestSpecialValues(const kxc::Device& device,
                       const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::codegen;
    const std::string symbol = "special_values";
    const auto dtype = runtime::DataTypeFromString(sizeof(Scalar) == 8 ? "float64" : "float32");
    const auto kernel = CUDAModule::Compile(
        CodeGenCUDA().Generate(MakeSpecialValues(tir::DataType::Float(sizeof(Scalar) * 8)), symbol),
        KernelSignature(String(symbol), {KernelArgSpec("output", KernelArgRole::kOutput,
            dtype, {4}, device, sizeof(Scalar), true)}),
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1}, {4, 1, 1}), options);
    auto output = runtime::NDArray::Empty({4}, dtype, device);
    kernel.Launch({output}, DeviceStream::Create(device)).Wait();
    Scalar actual[4] = {};
    output.CopyToBytes(actual, sizeof(actual));
    Require(std::isnan(actual[0]) && std::isinf(actual[1]) && !std::signbit(actual[1]) &&
                std::isinf(actual[2]) && std::signbit(actual[2]) && actual[3] == 0 && std::signbit(actual[3]),
            "CUDA special values lost category or sign");
}

/*! \brief 非法 CUDA 源码的异常必须同时含 NVRTC 操作名和编译器诊断日志。 */
void TestNVRTCLog(const kxc::codegen::CUDACompileOptions& options) {
    bool rejected = false;
    try {
        (void)kxc::codegen::CUDAModule::CompileToPTX(
            "extern \"C\" __global__ void broken( {", options);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        rejected = message.find("nvrtcCompileProgram") != std::string::npos &&
                   message.find("codegen_cuda_test.cu") != std::string::npos;
    }
    Require(rejected, "NVRTC failure did not preserve its compiler log");
}

/*! \brief PTX 能加载但缺少 signature.symbol 时必须在 Driver lookup 阶段失败。 */
void TestMissingSymbol(const kxc::Device& device,
                       const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::codegen;
    const std::string source = CodeGenCUDA().Generate(
        MakeBoundAdd("actual_symbol"), "actual_symbol");
    bool rejected = false;
    std::string diagnostic;
    try {
        (void)CUDAModule::Compile(
            source, AddSignature("missing_symbol", device),
            KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1},
                                 {8, 1, 1}),
            options);
    } catch (const std::runtime_error& error) {
        diagnostic = error.what();
        rejected = diagnostic.find("cuModuleGetFunction") !=
                   std::string::npos;
    }
    Require(rejected, "CUDA missing-symbol diagnostic mismatch: " + diagnostic);
}

void TestBatchModule(const kxc::Device& device,
                     const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::codegen;
    constexpr int64_t kExtent = 8;
    const std::string source = CodeGenCUDA().GenerateModule(
        {{MakeBoundAdd("batch_add_a", kExtent), "batch_add_a"},
         {MakeBoundAdd("batch_add_b", kExtent), "batch_add_b"}});
    std::vector<KernelSignature> signatures{
        AddSignature("batch_add_a", device, kExtent),
        AddSignature("batch_add_b", device, kExtent)};
    std::vector<KernelLaunchMetadata> metadata{
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1},
                             {static_cast<uint32_t>(kExtent), 1, 1}),
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1},
                             {static_cast<uint32_t>(kExtent), 1, 1})};
    std::vector<CompiledKernel> kernels = CUDAModule::CompileMany(
        source, signatures, metadata, options);
    Require(kernels.size() == 2 && kernels[0].IsReady() &&
                kernels[1].IsReady(),
            "CUDA batch compilation must return one kernel per symbol");
    kernels.erase(kernels.begin());
    const DLDataType dtype = runtime::DataTypeFromString("float32");
    runtime::NDArray lhs = runtime::NDArray::Empty({kExtent}, dtype, device);
    runtime::NDArray rhs = runtime::NDArray::Empty({kExtent}, dtype, device);
    runtime::NDArray output = runtime::NDArray::Empty({kExtent}, dtype, device);
    const std::vector<float> lhs_values(kExtent, 2.0f);
    const std::vector<float> rhs_values(kExtent, 3.0f);
    lhs.CopyFromBytes(lhs_values.data(), lhs.NBytes());
    rhs.CopyFromBytes(rhs_values.data(), rhs.NBytes());
    kernels[0]
        .Launch({lhs, rhs, output}, DeviceStream::Create(device))
        .Wait();
    std::vector<float> actual(kExtent);
    output.CopyToBytes(actual.data(), output.NBytes());
    for (float value : actual) {
        Require(std::fabs(value - 5.0f) < 1e-5f,
                "CUDA batch module result mismatch");
    }
}

/*! \brief 在 GPU 上执行 add，并验证 operation 独立保活输入 Storage 和 module。 */
void TestAsyncLaunchLifetime(
    const kxc::Device& device,
    const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::codegen;
    using runtime::NDArray;
    constexpr int64_t kExtent = 8;
    const std::string symbol = "async_add";
    const std::string source = CodeGenCUDA().Generate(
        MakeBoundAdd(symbol, kExtent), symbol);
    const DLDataType dtype = runtime::DataTypeFromString("float32");
    const DeviceStream stream = DeviceStream::Create(device);
    NDArray output;
    AsyncOperation operation;
    std::weak_ptr<const KernelLauncher> launcher;
    {
        CompiledKernel kernel = CUDAModule::Compile(
            source, AddSignature(symbol, device, kExtent),
            KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1},
                                  {static_cast<uint32_t>(kExtent), 1, 1}),
            options);
        launcher = kernel->launcher;
        NDArray a = NDArray::Empty({kExtent}, dtype, device);
        NDArray b = NDArray::Empty({kExtent}, dtype, device);
        output = NDArray::Empty({kExtent}, dtype, device);
        const std::vector<float> a_values{1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<float> b_values{10, 20, 30, 40, 50, 60, 70, 80};
        a.CopyFromBytes(a_values.data(), a_values.size() * sizeof(float));
        b.CopyFromBytes(b_values.data(), b_values.size() * sizeof(float));
        Array<NDArray> arguments{a, b, output};

        bool mismatch_rejected = false;
        try {
            (void)kernel.Launch(arguments, DeviceStream::Default(Device::CPU()));
        } catch (const std::invalid_argument&) {
            mismatch_rejected = true;
        }
        Require(mismatch_rejected, "CUDA launch accepted a CPU stream");
        operation = kernel.Launch(arguments, stream);
        // 离开作用域后 kernel、arguments、a、b 的外部引用全部释放；operation 必须
        // 独立持有这些异步依赖，直到 event 完成。
    }
    Require(!launcher.expired(),
            "pending CUDA operation did not retain its CUmodule launcher");
    operation.Wait();
    std::vector<float> actual(kExtent);
    output.CopyToBytes(actual.data(), actual.size() * sizeof(float));
    const std::vector<float> expected{11, 22, 33, 44, 55, 66, 77, 88};
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::fabs(actual[i] - expected[i]) < 1e-5f,
                "CUDA add result mismatch at " + std::to_string(i));
    }
    Require(!launcher.expired(),
            "CUDA operation released its launcher before the completion handle");
    // completion 仍统一持有 executable；释放最后一个句柄后必须触发 launcher 析构和 module unload。
    operation = AsyncOperation();
    Require(launcher.expired(),
            "CUDA launcher survived after the final completion reference was released");
}

/*! \brief 非零 byte_offset 必须传递逻辑首元素地址，且不能覆盖相邻哨兵。 */
void TestByteOffsetLaunch(
    const kxc::Device& device,
    const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::codegen;
    using runtime::NDArray;
    constexpr int64_t kExtent = 8;
    const std::string symbol = "offset_add";
    CompiledKernel kernel = CUDAModule::Compile(
        CodeGenCUDA().Generate(MakeBoundAdd(symbol, kExtent), symbol),
        AddSignature(symbol, device, kExtent),
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1},
                             {static_cast<uint32_t>(kExtent), 1, 1}),
        options);
    const DLDataType dtype = runtime::DataTypeFromString("float32");
    NDArray a_backing = NDArray::Empty({kExtent + 2}, dtype, device);
    NDArray b_backing = NDArray::Empty({kExtent + 2}, dtype, device);
    NDArray out_backing = NDArray::Zeros({kExtent + 2}, dtype, device);
    NDArray a = a_backing.CreateView({kExtent}, {1}, sizeof(float));
    NDArray b = b_backing.CreateView({kExtent}, {1}, sizeof(float));
    NDArray output = out_backing.CreateView({kExtent}, {1}, sizeof(float));
    const std::vector<float> a_values{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> b_values{8, 7, 6, 5, 4, 3, 2, 1};
    a.CopyFromBytes(a_values.data(), a.NBytes());
    b.CopyFromBytes(b_values.data(), b.NBytes());
    kernel.Launch({a, b, output}, DeviceStream::Create(device)).Wait();

    std::vector<float> actual(kExtent + 2);
    out_backing.CopyToBytes(actual.data(), out_backing.NBytes());
    Require(actual.front() == 0.0f && actual.back() == 0.0f,
            "CUDA offset launch overwrote an adjacent sentinel");
    for (int64_t i = 1; i <= kExtent; ++i) {
        Require(std::fabs(actual[static_cast<size_t>(i)] - 9.0f) < 1e-5f,
                "CUDA offset launch used the Storage base address");
    }
}

/*!
 * \brief 编译 Relay 后只构造 inputs，由 RuntimeSession 分配输出并异步执行。
 *
 * H2D 与 kernel 使用同一 stream，依靠 CUDA stream 顺序而不在二者之间同步。
 * 离开内部作用域后只保留 outputs 和 completion，验证它们能独立覆盖执行期。
 */
template <typename Scalar = float>
std::vector<Scalar> CompileAndRunRelay(
    const kxc::Function& function, const kxc::Device& device,
    const std::vector<std::vector<Scalar>>& inputs, size_t output_elements) {
    using namespace kxc;
    runtime::RunAsyncResult run_result;
    {
        api::CompiledGraph compiled = api::Compiler::Compile(
            function, api::CompileConfig::Create(BuildTarget(device), 3));
        runtime::RuntimeSession session(compiled.module(), compiled.plan());
        const Array<runtime::ValueSpec> value_specs = compiled.plan().values();
        const auto find_value = [&](int64_t value_id) {
            for (const auto& value : value_specs) {
                if (value->value_id == value_id) return value;
            }
            throw std::runtime_error(
                "Compiler CUDA plan references an unknown input value");
        };
        const DeviceStream stream = DeviceStream::Create(device);
        Array<runtime::NDArray> device_inputs;
        Array<runtime::NDArray> host_inputs;
        Array<AsyncOperation> uploads;
        size_t input_index = 0;
        for (int64_t value_id : compiled.plan().input_value_ids()) {
            const runtime::ValueSpec spec = find_value(value_id);
            Require(input_index < inputs.size(),
                    "Compiler CUDA plan has too many inputs");
            const auto& values = inputs[input_index++];
            runtime::NDArray host = runtime::NDArray::Empty(
                spec.shape(), spec->dtype, Device::CPU());
            Require(values.size() * sizeof(Scalar) == host.NBytes(),
                    "Compiler CUDA input byte count mismatch");
            host.CopyFromBytes(values.data(), host.NBytes());
            runtime::NDArray input = runtime::NDArray::Empty(
                spec.shape(), spec->dtype, spec->device);
            uploads.push_back(input.CopyFromAsync(host, stream));
            host_inputs.push_back(std::move(host));
            device_inputs.push_back(std::move(input));
        }
        Require(input_index == inputs.size(),
                "Compiler CUDA signature omitted an input");
        // RunAsync 在同一 stream 上排在全部 H2D 之后；此处没有显式 Wait。
        run_result = session.RunAsync(device_inputs, stream);
    }
    run_result.completion.Wait();
    Require(run_result.outputs.size() == 1,
            "RuntimeSession CUDA should allocate one output");
    std::vector<Scalar> result(output_elements);
    Require(result.size() * sizeof(Scalar) == run_result.outputs[0].NBytes(),
            "Compiler CUDA output byte count mismatch");
    run_result.outputs[0].CopyToBytes(result.data(),
                                      run_result.outputs[0].NBytes());
    return result;
}

/*! \brief Compiler 拒绝超出私有临时存储预算的图。 */
void TestCompilerRejectsUnsupportedTransformerGraphs(const kxc::Device& device) {
    using namespace kxc;
    const auto expect_rejection = [&device](const Function& function,
                                            const std::string& name,
                                            const std::string& required_detail = "") {
        std::string diagnostic;
        try {
            (void)api::Compiler::Compile(
                function, api::CompileConfig::Create(BuildTarget(device), 3));
        } catch (const std::exception& error) {
            diagnostic = error.what();
        }
        Require(diagnostic.find("BindCudaThreads") != std::string::npos &&
                    (required_detail.empty() ||
                     diagnostic.find(required_detail) != std::string::npos),
                "Compiler CUDA did not reject unsupported " + name +
                    " at the target schedule gate: " + diagnostic);
    };

    Var logits("logits", TensorType({2, 10000}, "float32"));
    expect_rejection(
        Function({logits}, Call(relay::Op::Get("softmax"), {logits},
                                relay::SoftmaxAttrs::Create(-1))),
        "oversized softmax scratch", "64 KiB");

}

/*! \brief 验证 CPU Relay Constant 放置到 CUDA 后由 RuntimeSession 自动注入。 */
void TestCompilerConstant(const kxc::Device& device) {
    using namespace kxc;
    runtime::NDArray data = runtime::NDArray::Empty(
        {8}, runtime::DataTypeFromString("float32"), Device::CPU());
    const std::vector<float> constant_values{10, 20, 30, 40, 50, 60, 70, 80};
    data.CopyFromBytes(constant_values.data(), data.NBytes());
    Var x("x", TensorType({8}, "float32"));
    Function function(
        {x}, Call(relay::Op::Get("add"), {x, Constant(data)}));
    const std::vector<float> actual = CompileAndRunRelay(
        function, device, {{1, 2, 3, 4, 5, 6, 7, 8}}, 8);
    const std::vector<float> expected{11, 22, 33, 44, 55, 66, 77, 88};
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::fabs(actual[i] - expected[i]) < 1e-5f,
                "Compiler CUDA constant mismatch at " + std::to_string(i));
    }
}

/*! \brief 验证 Relay add 经 Compiler 和 RuntimeSession 自动输出路径得到正确结果。 */
void TestCompilerAdd(const kxc::Device& device) {
    using namespace kxc;
    Var x("x", TensorType({8}, "float32"));
    Var y("y", TensorType({8}, "float32"));
    Function function({x, y}, Call(relay::Op::Get("add"), {x, y}));
    const std::vector<float> actual = CompileAndRunRelay(
        function, device,
        {{1, 2, 3, 4, 5, 6, 7, 8}, {10, 20, 30, 40, 50, 60, 70, 80}}, 8);
    const std::vector<float> expected{11, 22, 33, 44, 55, 66, 77, 88};
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::fabs(actual[i] - expected[i]) < 1e-5f,
                "Compiler CUDA add mismatch at " + std::to_string(i));
    }
}

/*! \brief 仅验证非空 1-D Where、Slice、Concatenate 的本地 CUDA 证据。 */
void TestCompilerTransformerInjectiveOps1DNonEmpty(const kxc::Device& device) {
    using namespace kxc;
    const auto require_equal = [](const std::vector<float>& actual,
                                  const std::vector<float>& expected,
                                  const std::string& name) {
        Require(actual.size() == expected.size(), name + " CUDA result size mismatch");
        for (size_t i = 0; i < actual.size(); ++i) {
            Require(std::fabs(actual[i] - expected[i]) < 1e-5f,
                    name + " CUDA result mismatch at " + std::to_string(i));
        }
    };

    runtime::NDArray condition = runtime::NDArray::Empty(
        {4}, runtime::DataTypeFromString("bool"), Device::CPU());
    const std::vector<uint8_t> condition_values{1, 0, 1, 0};
    condition.CopyFromBytes(condition_values.data(), condition_values.size());
    Var x("x", TensorType({4}, "float32"));
    Var fallback("fallback", TensorType({4}, "float32"));
    Function where_function(
        {x, fallback},
        Call(relay::Op::Get("where"), {Constant(condition), x, fallback}));
    require_equal(CompileAndRunRelay(
                      where_function, device,
                      {{1, 2, 3, 4}, {10, 20, 30, 40}}, 4),
                  {1, 20, 3, 40}, "Where");

    Function slice_function(
        {x}, Call(relay::Op::Get("slice"), {x},
                  relay::SliceAttrs::Create({1}, {3}, {0}, {1})));
    require_equal(CompileAndRunRelay(slice_function, device, {{1, 2, 3, 4}}, 2),
                  {2, 3}, "Slice");

    Var lhs("lhs", TensorType({2}, "float32"));
    Var rhs("rhs", TensorType({2}, "float32"));
    Function concatenate_function(
        {lhs, rhs}, Call(relay::Op::Get("concatenate"), {lhs, rhs},
                         relay::ConcatenateAttrs::Create(0)));
    require_equal(CompileAndRunRelay(concatenate_function, device, {{2, 3}, {7, 8}}, 4),
                  {2, 3, 7, 8}, "Concatenate");
}

template <typename Scalar>
void ExpectNear(const std::vector<Scalar>& actual,
                const std::vector<Scalar>& expected, const std::string& name,
                double tolerance = 1e-5) {
    Require(actual.size() == expected.size(), name + " size mismatch");
    double max_error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        if (!std::isfinite(actual[i]) || error > tolerance * (1 + std::fabs(expected[i]))) {
            std::ostringstream detail;
            detail << std::setprecision(17) << name << " mismatch at " << i
                   << ": actual=" << actual[i] << " expected=" << expected[i] << " tolerance=" << tolerance;
            throw std::runtime_error(detail.str());
        }
        if (error > max_error) max_error = error;
    }
    std::cout << "[NUMERIC] " << name << " elements=" << actual.size()
              << " max_abs_error=" << max_error << '\n';
}

template <typename Scalar>
std::vector<Scalar> InputValues(size_t count, int salt) {
    std::vector<Scalar> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<Scalar>((static_cast<int>(i % 29) * salt) % 31 - 15) /
                    static_cast<Scalar>(37);
    }
    return values;
}

// Independent host loops verify the transpose contract, signed accumulation and tail.
template <typename Scalar>
void TestCompilerMatmul(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    constexpr int64_t m = 17, n = 19, k = 33;
    Var a("a", TensorType({m, k}, dtype));
    Var b("b", TensorType({k, n}, dtype));
    const auto av = InputValues<Scalar>(m * k, 7);
    const auto bv = InputValues<Scalar>(k * n, 11);
    std::vector<Scalar> expected(m * n);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sum = 0;
            for (int64_t r = 0; r < k; ++r) sum += static_cast<double>(av[i * k + r]) * bv[r * n + j];
            expected[i * n + j] = static_cast<Scalar>(sum);
        }
    }
    const double tolerance = sizeof(Scalar) == 8 ? 1e-12 : 1e-5;
    ExpectNear(CompileAndRunRelay<Scalar>(
        Function({a, b}, Call(relay::Op::Get("matmul"), {a, b})),
        device, {av, bv}, expected.size()), expected, "matmul_17x33x19_" + dtype, tolerance);

    Var weight("weight", TensorType({n, k}, dtype));
    std::vector<Scalar> transposed(bv.size());
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t r = 0; r < k; ++r) transposed[j * k + r] = bv[r * n + j];
    }
    ExpectNear(CompileAndRunRelay<Scalar>(
        Function({a, weight}, Call(relay::Op::Get("nn_dense"), {a, weight},
                                  relay::DenseAttrs::Create(n, dtype))),
        device, {av, transposed}, expected.size()), expected, "dense_17x33x19_" + dtype, tolerance);
}

void TestCompilerBatchedMatmul(const kxc::Device& device) {
    using namespace kxc;
    Var a("a", TensorType({2, 1, 5, 9}, "float32"));
    Var b("b", TensorType({1, 4, 9, 7}, "float32"));
    const auto av = InputValues<float>(2 * 5 * 9, 3);
    const auto bv = InputValues<float>(4 * 9 * 7, 13);
    std::vector<float> expected(2 * 4 * 5 * 7);
    for (int batch_a = 0; batch_a < 2; ++batch_a) {
        for (int batch_b = 0; batch_b < 4; ++batch_b) {
            for (int row = 0; row < 5; ++row) {
                for (int col = 0; col < 7; ++col) {
                    double sum = 0;
                    for (int r = 0; r < 9; ++r) {
                        sum += static_cast<double>(av[(batch_a * 5 + row) * 9 + r]) *
                               bv[(batch_b * 9 + r) * 7 + col];
                    }
                    expected[((batch_a * 4 + batch_b) * 5 + row) * 7 + col] = static_cast<float>(sum);
                }
            }
        }
    }
    ExpectNear(CompileAndRunRelay(
        Function({a, b}, Call(relay::Op::Get("matmul"), {a, b})), device, {av, bv},
        expected.size()), expected, "batched_matmul_2x1x5x9_1x4x9x7_float32");
}

void TestCompilerTransformerInjectiveOpsND(const kxc::Device& device) {
    using namespace kxc;
    runtime::NDArray condition = runtime::NDArray::Empty(
        {1, 3, 1}, runtime::DataTypeFromString("bool"), Device::CPU());
    const uint8_t condition_values[] = {1, 0, 1};
    condition.CopyFromBytes(condition_values, sizeof(condition_values));
    Var x("x", TensorType({2, 3, 5}, "float32"));
    Var fallback("fallback", TensorType({1, 1, 5}, "float32"));
    const auto xv = InputValues<float>(30, 7);
    const auto fv = InputValues<float>(5, 11);
    std::vector<float> expected = xv;
    for (int batch = 0; batch < 2; ++batch) {
        for (int col = 0; col < 5; ++col) expected[(batch * 3 + 1) * 5 + col] = fv[col];
    }
    ExpectNear(CompileAndRunRelay(
        Function({x, fallback}, Call(relay::Op::Get("where"), {Constant(condition), x, fallback})),
        device, {xv, fv}, expected.size()), expected, "where_3d_broadcast_float32");

    expected.clear();
    for (int row = 0; row < 6; ++row) {
        for (int col = 1; col < 4; ++col) expected.push_back(xv[row * 5 + col]);
    }
    ExpectNear(CompileAndRunRelay(
        Function({x}, Call(relay::Op::Get("slice"), {x},
                          relay::SliceAttrs::Create({1}, {4}, {-1}, {1}))),
        device, {xv}, expected.size()), expected, "slice_3d_step1_float32");

    Var y("y", TensorType({2, 1, 5}, "float32"));
    const auto yv = InputValues<float>(10, 5);
    expected.clear();
    for (int batch = 0; batch < 2; ++batch) {
        expected.insert(expected.end(), xv.begin() + batch * 15, xv.begin() + (batch + 1) * 15);
        expected.insert(expected.end(), yv.begin() + batch * 5, yv.begin() + (batch + 1) * 5);
    }
    ExpectNear(CompileAndRunRelay(
        Function({x, y}, Call(relay::Op::Get("concatenate"), {x, y},
                             relay::ConcatenateAttrs::Create(-2))),
        device, {xv, yv}, expected.size()), expected, "concatenate_3d_float32");

    Var scalar("scalar", TensorType({}, "float32"));
    ExpectNear(CompileAndRunRelay(Function({scalar}, Call(relay::Op::Get("add"), {scalar, scalar})),
                                  device, {{-3.5f}}, 1), std::vector<float>{-7}, "scalar_add_float32");
}

// Direct TIR complements production MatMul with scalar/multiple-axis sum and max.
template <typename Scalar>
void TestOwnedReductionKernel(const kxc::Device& device,
                              const kxc::codegen::CUDACompileOptions& options,
                              bool scalar) {
    using namespace kxc;
    using namespace kxc::tir;
    using namespace kxc::codegen;
    const int64_t rows = scalar ? 1 : 17 * 19;
    constexpr int64_t reduction = 33;
    const DataType dtype = DataType::Float(sizeof(Scalar) * 8);
    const DataType i64 = DataType::Int(64);
    tir::Var input("input", dtype), sum("sum", dtype), maximum("maximum", dtype);
    tir::Var i("i", i64), j("j", i64), r("r", i64);
    const int64_t reduction_min = scalar ? 2147483647 : 0;
    const PrimExpr index = scalar ? PrimExpr(IntImm(0, i64)) : i * IntImm(19, i64) + j;
    const PrimExpr value = Load(input, index * IntImm(reduction, i64) +
                                      (r - IntImm(reduction_min, i64)));
    Stmt body = SeqStmt({
        Store(sum, FloatImm(0, dtype), index),
        Store(maximum, FloatImm(-std::numeric_limits<double>::infinity(), dtype), index),
        For(r, IntImm(reduction_min, DataType::Int(32)), IntImm(reduction, DataType::Int(32)), ForType::Serial,
            SeqStmt({Store(sum, Load(sum, index) + value, index),
                     Store(maximum, Max(Load(maximum, index), value), index)}))});
    if (!scalar) {
        body = For(i, IntImm(0, i64), IntImm(17, i64), ForType::Serial,
                   For(j, IntImm(0, i64), IntImm(19, i64), ForType::Serial, body));
    }
    Map<tir::Var, Buffer> buffers;
    buffers.Set(input, Buffer(input, dtype, {IntImm(rows * reduction, i64)}, {}, IntImm(0), "input", 0, 0));
    for (const auto& output : {sum, maximum}) {
        buffers.Set(output, Buffer(output, dtype, {IntImm(rows, i64)}, {}, IntImm(0), output->name_hint, 0, 0));
    }
    const auto scheduled = BindCudaThreads(PrimFunc({input, sum, maximum}, body, buffers), BuildTarget(device));
    const std::string symbol = "owned_sum_max";
    const DLDataType runtime_dtype = runtime::DataTypeFromString(sizeof(Scalar) == 8 ? "float64" : "float32");
    const auto launch = scheduled.launch_config();
    const auto kernel = CUDAModule::Compile(CodeGenCUDA().Generate(scheduled.prim_func(), symbol),
        KernelSignature(String(symbol), {
            KernelArgSpec("input", KernelArgRole::kInput, runtime_dtype, {rows * reduction}, device),
            KernelArgSpec("sum", KernelArgRole::kOutput, runtime_dtype, {rows}, device, sizeof(Scalar), true),
            KernelArgSpec("maximum", KernelArgRole::kOutput, runtime_dtype, {rows}, device, sizeof(Scalar), true)}),
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {launch.grid_x, 1, 1}, {launch.block_x, 1, 1}), options);
    const auto values = InputValues<Scalar>(rows * reduction, 7);
    auto gpu_input = runtime::NDArray::Empty({rows * reduction}, runtime_dtype, device);
    auto gpu_sum = runtime::NDArray::Empty({rows}, runtime_dtype, device);
    auto gpu_max = runtime::NDArray::Empty({rows}, runtime_dtype, device);
    gpu_input.CopyFromBytes(values.data(), gpu_input.NBytes());
    kernel.Launch({gpu_input, gpu_sum, gpu_max}, DeviceStream::Create(device)).Wait();
    std::vector<Scalar> actual_sum(rows), actual_max(rows), expected_sum(rows), expected_max(rows);
    gpu_sum.CopyToBytes(actual_sum.data(), gpu_sum.NBytes());
    gpu_max.CopyToBytes(actual_max.data(), gpu_max.NBytes());
    for (int64_t row = 0; row < rows; ++row) {
        double total = 0;
        Scalar largest = -std::numeric_limits<Scalar>::infinity();
        for (int64_t k = 0; k < reduction; ++k) {
            const Scalar item = values[row * reduction + k];
            total += item;
            if (item > largest) largest = item;
        }
        expected_sum[row] = static_cast<Scalar>(total);
        expected_max[row] = largest;
    }
    const std::string name = std::string(scalar ? "scalar" : "2d") + "_float" + std::to_string(sizeof(Scalar) * 8);
    ExpectNear(actual_sum, expected_sum, "owned_sum_" + name, sizeof(Scalar) == 8 ? 1e-12 : 1e-5);
    ExpectNear(actual_max, expected_max, "owned_max_" + name, 0);
}

size_t ShapeSize(const kxc::Array<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dimension : shape) count *= static_cast<size_t>(dimension);
    return count;
}

size_t BroadcastMaskIndex(size_t index, const kxc::Array<int64_t>& shape,
                          const kxc::Array<int64_t>& mask_shape) {
    size_t result = 0, stride = 1;
    for (size_t reverse = shape.size(); reverse != 0; --reverse) {
        const size_t axis = reverse - 1, coordinate = index % shape[axis];
        index /= shape[axis];
        if (axis + mask_shape.size() >= shape.size()) {
            const auto dimension = mask_shape[axis + mask_shape.size() - shape.size()];
            if (dimension != 1) result += coordinate * stride;
            stride *= dimension;
        }
    }
    return result;
}

// Independent host reference: each logical reduction row is traversed directly.
template <typename Scalar>
std::vector<Scalar> SoftmaxReference(const std::vector<Scalar>& values,
    const kxc::Array<int64_t>& shape, int axis,
    const std::vector<uint8_t>& mask = {}, const kxc::Array<int64_t>& mask_shape = {}) {
    size_t inner = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
    const size_t width = shape[axis], outer = values.size() / (inner * width);
    std::vector<Scalar> expected(values.size(), Scalar(0));
    const auto active = [&](size_t index) {
        return mask.empty() || mask[BroadcastMaskIndex(index, shape, mask_shape)] != 0;
    };
    for (size_t row = 0; row < outer; ++row) {
        for (size_t tail = 0; tail < inner; ++tail) {
            double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
            for (size_t k = 0; k < width; ++k) {
                const size_t index = (row * width + k) * inner + tail;
                if (active(index) && values[index] > maximum) maximum = values[index];
            }
            for (size_t k = 0; k < width; ++k) {
                const size_t index = (row * width + k) * inner + tail;
                if (active(index)) sum += std::exp(static_cast<double>(values[index]) - maximum);
            }
            if (sum == 0) continue;
            for (size_t k = 0; k < width; ++k) {
                const size_t index = (row * width + k) * inner + tail;
                if (active(index)) expected[index] = static_cast<Scalar>(
                    std::exp(static_cast<double>(values[index]) - maximum) / sum);
            }
        }
    }
    return expected;
}

template <typename Scalar>
void TestCompilerSoftmax(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    const std::vector<std::pair<Array<int64_t>, int>> cases{
        {{257, 7}, 1}, {{3, 4, 5}, 0}, {{3, 4, 5}, 1}, {{7}, 0}, {{1, 257}, 1}};
    for (size_t test = 0; test < cases.size(); ++test) {
        const auto& shape = cases[test].first;
        const int axis = cases[test].second;
        for (int pattern = 0; pattern < 2; ++pattern) {
            auto values = InputValues<Scalar>(ShapeSize(shape), 7);
            if (pattern == 1) {
                for (auto& value : values) value *= Scalar(1000);
                values[0] = std::numeric_limits<Scalar>::max();
                values.back() = -std::numeric_limits<Scalar>::max();
            }
            Var x("logits", TensorType(shape, dtype));
            ExpectNear(CompileAndRunRelay<Scalar>(Function({x}, Call(relay::Op::Get("softmax"), {x},
                    relay::SoftmaxAttrs::Create(axis))), device, {values}, values.size()),
                SoftmaxReference(values, shape, axis), "softmax_case" + std::to_string(test) +
                "_pattern" + std::to_string(pattern) + "_" + dtype, sizeof(Scalar) == 8 ? 1e-12 : 2e-5);
        }
    }
}

template <typename Scalar>
void TestCompilerMaskedSoftmax(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    struct Case { Array<int64_t> shape, mask_shape; int axis; };
    const std::vector<Case> cases{{{257, 7}, {257, 7}, 1}, {{2, 3, 5}, {1, 3, 1}, 1}, {{3, 5}, {}, 1}};
    for (size_t test = 0; test < cases.size(); ++test) {
        const auto& item = cases[test];
        Var x("logits", TensorType(item.shape, dtype)), m("mask", TensorType(item.mask_shape, "bool"));
        const auto graph = api::Compiler::Compile(Function({x, m}, Call(relay::Op::Get("masked_softmax"),
            {x, m}, relay::SoftmaxAttrs::Create(item.axis))), api::CompileConfig::Create(BuildTarget(device), 3));
        Require(graph.plan().calls().size() == 1, "masked softmax must remain one primitive");
        runtime::RuntimeSession session(graph.module(), graph.plan());
        const auto value_dtype = runtime::DataTypeFromString(dtype);
        for (int pattern = 0; pattern < 4; ++pattern) {
            auto values = InputValues<Scalar>(ShapeSize(item.shape), 11);
            std::vector<uint8_t> mask(ShapeSize(item.mask_shape));
            for (size_t i = 0; i < mask.size(); ++i) {
                mask[i] = pattern == 1 || pattern == 3 ||
                    (pattern == 2 && i % 3 != 0 && (mask.size() <= 7 || i >= 7));
            }
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] *= Scalar(pattern == 1 ? 1 : 5);
                if (pattern == 3) values[i] = (i % 3 ? Scalar(-1) : Scalar(1)) * std::numeric_limits<Scalar>::max();
                if (!mask[BroadcastMaskIndex(i, item.shape, item.mask_shape)]) values[i] = i % 2
                    ? std::numeric_limits<Scalar>::quiet_NaN() : std::numeric_limits<Scalar>::infinity();
            }
            auto gpu_values = runtime::NDArray::Empty(item.shape, value_dtype, device);
            auto gpu_mask = runtime::NDArray::Empty(item.mask_shape, runtime::DataTypeFromString("bool"), device);
            gpu_values.CopyFromBytes(values.data(), gpu_values.NBytes());
            gpu_mask.CopyFromBytes(mask.data(), gpu_mask.NBytes());
            const auto run = session.RunAsync({gpu_values, gpu_mask}, DeviceStream::Create(device));
            run.completion.Wait();
            std::vector<Scalar> actual(values.size());
            run.outputs[0].CopyToBytes(actual.data(), run.outputs[0].NBytes());
            const auto expected = SoftmaxReference(values, item.shape, item.axis, mask, item.mask_shape);
            ExpectNear(actual, expected, "masked_softmax_case" + std::to_string(test) + "_pattern" +
                std::to_string(pattern) + "_" + dtype, sizeof(Scalar) == 8 ? 1e-12 : 2e-5);
            for (size_t i = 0; i < actual.size(); ++i) {
                if (expected[i] == 0) Require(actual[i] == Scalar(0) && !std::signbit(actual[i]),
                    "masked/all-false result must be exact positive zero");
            }
        }
    }
}

// Host reference indexes the logical table directly; runtime indices change on
// the same compiled session, including signed extremes and empty tensors.
template <typename Scalar, typename Index>
void TestCompilerGather(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    struct Case { Array<int64_t> table, indices; int axis; };
    const std::vector<Case> cases{{{2, 7, 5}, {3, 4}, 1}, {{7, 5}, {257}, 0},
        {{7}, {}, -1}, {{0, 5}, {9}, 0}, {{7, 5}, {0}, 0}, {{2, 3, 7}, {3, 4}, -1}};
    const std::string index_dtype = sizeof(Index) == 8 ? "int64" : "int32";
    for (size_t test = 0; test < cases.size(); ++test) {
        const auto& item = cases[test];
        const size_t axis = item.axis < 0 ? item.table.size() + item.axis : item.axis;
        const int64_t width = item.table[axis];
        size_t outer = 1, inner = 1;
        for (size_t i = 0; i < axis; ++i) outer *= item.table[i];
        for (size_t i = axis + 1; i < item.table.size(); ++i) inner *= item.table[i];
        Array<int64_t> output_shape;
        for (size_t i = 0; i < axis; ++i) output_shape.push_back(item.table[i]);
        for (auto extent : item.indices) output_shape.push_back(extent);
        for (size_t i = axis + 1; i < item.table.size(); ++i) output_shape.push_back(item.table[i]);
        Var x("table", TensorType(item.table, dtype)), idx("indices", TensorType(item.indices, index_dtype));
        const auto graph = api::Compiler::Compile(Function({x, idx}, Call(relay::Op::Get("gather"),
            {x, idx}, relay::GatherAttrs::Create(item.axis))),
            api::CompileConfig::Create(BuildTarget(device), 3));
        Require(graph.plan().calls().size() == 1, "Gather must use the production primitive");
        runtime::RuntimeSession session(graph.module(), graph.plan());
        std::vector<Scalar> values(ShapeSize(item.table));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<Scalar>(dtype == "bool" ? i % 2 : 1 + i % 37);
        }
        auto table = runtime::NDArray::Empty(item.table, runtime::DataTypeFromString(dtype), device);
        table.CopyFromBytes(values.data(), table.NBytes());
        const std::vector<Index> pattern{0, static_cast<Index>(width - 1), -1,
            static_cast<Index>(-width), static_cast<Index>(width), static_cast<Index>(-width - 1),
            std::numeric_limits<Index>::min(), std::numeric_limits<Index>::max(), 1};
        const auto stream = DeviceStream::Create(device);
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            std::vector<Index> indices(ShapeSize(item.indices));
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = pattern[(i + repeat * 4) % pattern.size()];
            auto gpu_indices = runtime::NDArray::Empty(item.indices, runtime::DataTypeFromString(index_dtype), device);
            gpu_indices.CopyFromBytes(indices.data(), gpu_indices.NBytes());
            std::vector<Scalar> expected(outer * indices.size() * inner, Scalar(0));
            for (size_t row = 0; row < outer; ++row) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    int64_t index = indices[i];
                    if (index < -width || index >= width) continue;
                    if (index < 0) index += width;
                    for (size_t tail = 0; tail < inner; ++tail) {
                        expected[(row * indices.size() + i) * inner + tail] =
                            values[(row * width + index) * inner + tail];
                    }
                }
            }
            const auto run = session.RunAsync({table, gpu_indices}, stream);
            run.completion.Wait();
            Require(run.outputs.size() == 1 && run.outputs[0].device() == device &&
                    run.outputs[0].shape().size() == output_shape.size() &&
                    std::equal(output_shape.begin(), output_shape.end(), run.outputs[0].shape().begin()) &&
                    run.outputs[0].NBytes() == expected.size() * sizeof(Scalar),
                    "Gather must preserve CUDA device, shape and dtype byte count");
            std::vector<Scalar> actual(expected.size());
            run.outputs[0].CopyToBytes(actual.data(), run.outputs[0].NBytes());
            const auto name = "gather_case" + std::to_string(test) + "_repeat" + std::to_string(repeat) +
                "_" + dtype + "_" + index_dtype;
            Require(actual == expected, name + " exact indexing/typed-zero mismatch");
            std::cout << "[NUMERIC] " << name << " elements=" << actual.size() << " max_abs_error=0\n";
        }
    }
}

template <typename Scalar>
void TestCompilerPow(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    const std::vector<Scalar> bases{-3, -2, -1, 0, 1, 2, 4, 9, Scalar(0.25), 16, Scalar(0.5), 3};
    const std::vector<Scalar> powers{3, 2, -3, 2, 0, -2, Scalar(0.5), Scalar(1.5), Scalar(-0.5), Scalar(0.25), 3, -1};
    Var x("base", TensorType({3, 4}, dtype));
    for (bool scalar_exponent : {false, true}) {
        Var y("exponent", TensorType(scalar_exponent ? Array<int64_t>{} : Array<int64_t>{3, 4}, dtype));
        std::vector<Scalar> expected(bases.size()), exponents = scalar_exponent ? std::vector<Scalar>{2} : powers;
        for (size_t i = 0; i < bases.size(); ++i) expected[i] = std::pow(bases[i], exponents[scalar_exponent ? 0 : i]);
        ExpectNear(CompileAndRunRelay<Scalar>(Function({x, y}, Call(relay::Op::Get("pow"), {x, y})),
            device, {bases, exponents}, bases.size()), expected,
            "pow_" + dtype + (scalar_exponent ? "_scalar" : "_tensor"), sizeof(Scalar) == 8 ? 1e-12 : 1e-6);
    }
}

// Relay Pow deliberately remains float32; float64 is a TIR/backend capability.
void TestPowFloat64Backend(const kxc::Device& device,
                          const kxc::codegen::CUDACompileOptions& options) {
    using namespace kxc;
    using namespace kxc::tir;
    using namespace kxc::codegen;
    const auto dtype = DataType::Float(64);
    const auto base = MakeSpecialValues(dtype);
    tir::Var thread("tx", DataType::Int(32));
    const auto power = [&](double a, double b) {
        return tir::Call(dtype, "pow", {FloatImm(a, dtype), FloatImm(b, dtype)});
    };
    const auto value = Select(EQ(thread, IntImm(0)), power(-3, 3),
        Select(EQ(thread, IntImm(1)), power(4, 0.5),
            Select(EQ(thread, IntImm(2)), power(2, -3), power(0.25, 0.5))));
    const PrimFunc function(base->params, ThreadBinding(thread, ThreadIndexKind::kThreadIdxX,
        IntImm(4), Store(base->params[0], value, thread)), base->buffer_map);
    const auto dltype = runtime::DataTypeFromString("float64");
    const auto kernel = CUDAModule::Compile(CodeGenCUDA().Generate(function, "pow_f64"),
        KernelSignature(String("pow_f64"), {KernelArgSpec("output", KernelArgRole::kOutput,
            dltype, {4}, device, 8, true)}),
        KernelLaunchMetadata(device, CodeGenBackend::kCUDA, {1, 1, 1}, {4, 1, 1}), options);
    auto output = runtime::NDArray::Empty({4}, dltype, device);
    kernel.Launch({output}, DeviceStream::Create(device)).Wait();
    std::vector<double> actual(4);
    output.CopyToBytes(actual.data(), output.NBytes());
    ExpectNear(actual, std::vector<double>{-27, 2, 0.125, 0.5}, "pow_float64_backend", 1e-12);
}

void TestCompilerLayerNorm(const kxc::Device& device) {
    using namespace kxc;
    const std::vector<std::pair<Array<int64_t>, int>> cases{{{257, 7}, 1}, {{2, 3, 5}, 1}, {{7}, 0}};
    for (size_t test = 0; test < cases.size(); ++test) {
        const auto& shape = cases[test].first;
        Array<int64_t> affine_shape;
        for (size_t i = cases[test].second; i < shape.size(); ++i) affine_shape.push_back(shape[i]);
        const size_t width = ShapeSize(affine_shape), count = ShapeSize(shape);
        auto values = InputValues<float>(count, 7), scale = InputValues<float>(width, 5), bias = InputValues<float>(width, 3);
        for (size_t i = 0; i < count; ++i) values[i] = 10000.0f + (i < width ? 0.0f : values[i]);
        const double epsilon = test == 1 ? 1e-3 : 1e-5;
        std::vector<float> expected(count);
        for (size_t row = 0; row < count / width; ++row) {
            double mean = 0, variance = 0;
            for (size_t k = 0; k < width; ++k) mean += values[row * width + k];
            mean /= width;
            for (size_t k = 0; k < width; ++k) {
                const double centered = static_cast<double>(values[row * width + k]) - mean;
                variance += centered * centered;
            }
            for (size_t k = 0; k < width; ++k) expected[row * width + k] = static_cast<float>(
                (static_cast<double>(values[row * width + k]) - mean) / std::sqrt(variance / width + epsilon) * scale[k] + bias[k]);
        }
        Var x("data", TensorType(shape, "float32")), s("scale", TensorType(affine_shape, "float32")),
            b("bias", TensorType(affine_shape, "float32"));
        ExpectNear(CompileAndRunRelay(Function({x, s, b}, Call(relay::Op::Get("nn_layer_norm"), {x, s, b},
            relay::LayerNormAttrs::Create(cases[test].second, epsilon, "float64"))), device,
            {values, scale, bias}, count), expected, "layer_norm_case" + std::to_string(test) + "_float32_f64acc", 1e-5);
    }
}

template <typename Scalar>
void TestCompilerReduceMean(const kxc::Device& device, const std::string& dtype) {
    using namespace kxc;
    const Array<int64_t> shape{3, 4, 5};
    const auto values = InputValues<Scalar>(60, 13);
    const std::vector<Array<int64_t>> axes{{0, 2}, {1}, {0, 1, 2}};
    for (size_t test = 0; test < axes.size(); ++test) {
        for (bool keep : {false, true}) {
            bool reduced[3] = {false, false, false};
            size_t denominator = 1;
            for (int64_t axis : axes[test]) { reduced[axis] = true; denominator *= shape[axis]; }
            std::vector<double> sums(values.size() / denominator);
            for (size_t index = 0; index < values.size(); ++index) {
                size_t output = 0, remaining = index;
                const size_t coordinates[] = {remaining / 20, (remaining / 5) % 4, remaining % 5};
                for (size_t axis = 0; axis < 3; ++axis) if (!reduced[axis]) output = output * shape[axis] + coordinates[axis];
                sums[output] += values[index];
            }
            std::vector<Scalar> expected(sums.size());
            for (size_t i = 0; i < sums.size(); ++i) expected[i] = static_cast<Scalar>(sums[i] / denominator);
            Var x("data", TensorType(shape, dtype));
            ExpectNear(CompileAndRunRelay<Scalar>(Function({x}, Call(relay::Op::Get("reduce_mean"), {x},
                relay::ReduceMeanAttrs::Create(axes[test], keep))), device, {values}, expected.size()), expected,
                "reduce_mean_case" + std::to_string(test) + "_keep" + std::to_string(keep) + "_" + dtype,
                sizeof(Scalar) == 8 ? 1e-12 : 1e-5);
        }
    }
}

void TestCompilerAttentionComposition(const kxc::Device& device) {
    using namespace kxc;
    const auto qv = InputValues<float>(3 * 5 * 7, 3), kv = InputValues<float>(3 * 7 * 11, 7),
               vv = InputValues<float>(3 * 11 * 4, 11);
    std::vector<float> scores(3 * 5 * 11), expected(3 * 5 * 4, 0);
    for (int h = 0; h < 3; ++h) for (int i = 0; i < 5; ++i) for (int j = 0; j < 11; ++j) {
        double sum = 0;
        for (int k = 0; k < 7; ++k) sum += static_cast<double>(qv[(h * 5 + i) * 7 + k]) * kv[(h * 7 + k) * 11 + j];
        scores[(h * 5 + i) * 11 + j] = static_cast<float>(sum);
    }
    const auto probabilities = SoftmaxReference(scores, {3, 5, 11}, 2);
    for (int h = 0; h < 3; ++h) for (int i = 0; i < 5; ++i) for (int j = 0; j < 4; ++j) {
        double sum = 0;
        for (int k = 0; k < 11; ++k) sum += static_cast<double>(probabilities[(h * 5 + i) * 11 + k]) * vv[(h * 11 + k) * 4 + j];
        expected[(h * 5 + i) * 4 + j] = static_cast<float>(sum);
    }
    Var q("q", TensorType({3, 5, 7}, "float32")), k("k", TensorType({3, 7, 11}, "float32")),
        v("v", TensorType({3, 11, 4}, "float32"));
    const auto logits = Call(relay::Op::Get("matmul"), {q, k});
    const auto weights = Call(relay::Op::Get("softmax"), {logits}, relay::SoftmaxAttrs::Create(-1));
    ExpectNear(CompileAndRunRelay(Function({q, k, v}, Call(relay::Op::Get("matmul"), {weights, v})), device,
        {qv, kv, vv}, expected.size()), expected, "attention_matmul_softmax_matmul_3x5x7x11x4", 2e-5);
}

/*! \brief 验证 Relay relu 也能经过 Compiler 与 RuntimeSession CUDA 主路径。 */
void TestCompilerRelu(const kxc::Device& device) {
    using namespace kxc;
    Var x("x", TensorType({8}, "float32"));
    Function function(
        {x}, Call(relay::Op::Get("nn_relu"), {x}, relay::ReluAttrs::Create()));
    const std::vector<float> actual = CompileAndRunRelay(
        function, device, {{-4, -1, 0, 0.5f, 2, -3, 7, -8}}, 8);
    const std::vector<float> expected{0, 0, 0, 0.5f, 2, 0, 7, 0};
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::fabs(actual[i] - expected[i]) < 1e-5f,
                "Compiler CUDA relu mismatch at " + std::to_string(i));
    }
}

#endif

}  // namespace

/*!
 * \brief 顺序执行纯源码测试和可选真实 CUDA 测试，任一失败返回非零。
 *
 * `--memcheck` 只跳过会故意触发 CUDA_ERROR_NOT_FOUND 的符号负例，避免
 * Compute Sanitizer 把预期 Driver 错误计入内存检查汇总。
 */
int main(int argc, char** argv) {
    const bool memcheck_mode =
        argc == 2 && std::string(argv[1]) == "--memcheck";
    if (argc > 2 || (argc == 2 && !memcheck_mode)) {
        std::cerr << "usage: codegen_cuda_test [--memcheck]\n";
        return 2;
    }
    const std::vector<std::pair<const char*, void (*)()>> source_tests = {
        {"source_emission", TestSourceEmission},
        {"special_value_source", TestSpecialValueSource},
        {"math_call_source", TestMathCallSource},
        {"unsupported_tir_rejected", TestUnsupportedTIRRejected},
        {"unbound_tir_rejected", TestUnboundTIRRejected},
    };
    for (const auto& test : source_tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }

#if KXC_USE_CUDA
    try {
        const kxc::Device device = kxc::Device::CUDA(0);
        const auto options = CompileOptions(device);
        TestSpecialValues<float>(device, options);
        TestSpecialValues<double>(device, options);
        std::cout << "[PASS] special_values_f32_f64\n";
        TestNVRTCLog(options);
        std::cout << "[PASS] nvrtc_log\n";
        if (!memcheck_mode) {
            TestMissingSymbol(device, options);
            std::cout << "[PASS] missing_symbol\n";
        } else {
            std::cout << "[SKIP] missing_symbol under Compute Sanitizer\n";
        }
        TestAsyncLaunchLifetime(device, options);
        std::cout << "[PASS] async_launch_lifetime\n";
        TestBatchModule(device, options);
        std::cout << "[PASS] batch_module\n";
        TestByteOffsetLaunch(device, options);
        std::cout << "[PASS] byte_offset_launch\n";
        TestCompilerAdd(device);
        std::cout << "[PASS] runtime_session_add\n";
        TestCompilerConstant(device);
        std::cout << "[PASS] runtime_session_constant\n";
        TestCompilerRelu(device);
        std::cout << "[PASS] runtime_session_relu\n";
        TestCompilerTransformerInjectiveOps1DNonEmpty(device);
        std::cout << "[PASS] runtime_session_transformer_injective_ops_1d_nonempty_local_evidence\n";
        TestCompilerMatmul<float>(device, "float32");
        TestCompilerMatmul<double>(device, "float64");
        std::cout << "[PASS] runtime_session_matmul_dense_f32_f64_local_evidence\n";
        TestCompilerBatchedMatmul(device);
        std::cout << "[PASS] runtime_session_batched_matmul_broadcast_local_evidence\n";
        TestCompilerTransformerInjectiveOpsND(device);
        std::cout << "[PASS] runtime_session_transformer_injective_ops_nd_local_evidence\n";
        TestOwnedReductionKernel<float>(device, options, false);
        TestOwnedReductionKernel<double>(device, options, false);
        TestOwnedReductionKernel<float>(device, options, true);
        TestOwnedReductionKernel<double>(device, options, true);
        std::cout << "[PASS] owned_scalar_and_2d_sum_max_f32_f64\n";
        TestCompilerSoftmax<float>(device, "float32");
        TestCompilerSoftmax<double>(device, "float64");
        std::cout << "[PASS] runtime_session_softmax_multistage_local_evidence\n";
        TestCompilerMaskedSoftmax<float>(device, "float32");
        TestCompilerMaskedSoftmax<double>(device, "float64");
        std::cout << "[PASS] runtime_session_masked_softmax_multistage_local_evidence\n";
        TestCompilerLayerNorm(device);
        std::cout << "[PASS] runtime_session_layer_norm_multistage_local_evidence\n";
        TestCompilerReduceMean<float>(device, "float32");
        TestCompilerReduceMean<double>(device, "float64");
        std::cout << "[PASS] runtime_session_reduce_mean_multistage_local_evidence\n";
        TestCompilerAttentionComposition(device);
        std::cout << "[PASS] runtime_session_attention_composition_local_evidence\n";
        TestCompilerGather<float, int32_t>(device, "float32");
        TestCompilerGather<float, int64_t>(device, "float32");
        TestCompilerGather<double, int32_t>(device, "float64");
        TestCompilerGather<double, int64_t>(device, "float64");
        TestCompilerGather<int32_t, int32_t>(device, "int32");
        TestCompilerGather<int32_t, int64_t>(device, "int32");
        TestCompilerGather<int64_t, int32_t>(device, "int64");
        TestCompilerGather<int64_t, int64_t>(device, "int64");
        TestCompilerGather<uint8_t, int32_t>(device, "bool");
        TestCompilerGather<uint8_t, int64_t>(device, "bool");
        std::cout << "[PASS] runtime_session_guarded_gather_local_evidence\n";
        TestCompilerPow<float>(device, "float32");
        TestPowFloat64Backend(device, options);
        std::cout << "[PASS] runtime_session_pow_f32_and_backend_pow_f64_local_evidence\n";
        TestCompilerRejectsUnsupportedTransformerGraphs(device);
        std::cout << "[PASS] compiler_rejects_oversized_scratch\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] cuda_runtime: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "[SKIP] CUDA runtime tests: KXC_USE_CUDA=0\n";
#endif
    return 0;
}
