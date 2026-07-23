/*! \file test/codegen_cuda_test.cpp
 * \brief 验证线程绑定 CUDA 源码、NVRTC/Driver module 和异步 NDArray 启动契约。
 */

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
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
std::vector<float> CompileAndRunRelay(
    const kxc::Function& function, const kxc::Device& device,
    const std::vector<std::vector<float>>& inputs, size_t output_elements) {
    using namespace kxc;
    runtime::RunAsyncResult run_result;
    {
        api::CompiledGraph compiled = api::Compiler::Compile(
            function, api::CompileConfig::Create(BuildTarget(device), 2));
        runtime::RuntimeSession session(compiled.module, compiled.plan);
        const Array<runtime::ValueSpec> value_specs = compiled.plan.values();
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
        for (int64_t value_id : compiled.plan.input_value_ids()) {
            const runtime::ValueSpec spec = find_value(value_id);
            Require(input_index < inputs.size(),
                    "Compiler CUDA plan has too many inputs");
            const auto& values = inputs[input_index++];
            runtime::NDArray host = runtime::NDArray::Empty(
                spec.shape(), spec->dtype, Device::CPU());
            Require(values.size() * sizeof(float) == host.NBytes(),
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
    std::vector<float> result(output_elements);
    Require(result.size() * sizeof(float) == run_result.outputs[0].NBytes(),
            "Compiler CUDA output byte count mismatch");
    run_result.outputs[0].CopyToBytes(result.data(),
                                      run_result.outputs[0].NBytes());
    return result;
}

/*! \brief Compiler 必须在 CUDA 调度阶段拒绝 reduction/nested-loop Relay 图，不能暗示 device 数值支持。 */
void TestCompilerRejectsReductionGraphs(const kxc::Device& device) {
    using namespace kxc;
    const auto expect_rejection = [&device](const Function& function,
                                            const std::string& name) {
        std::string diagnostic;
        try {
            (void)api::Compiler::Compile(
                function, api::CompileConfig::Create(BuildTarget(device), 2));
        } catch (const std::exception& error) {
            diagnostic = error.what();
        }
        Require(diagnostic.find("BindCudaThreads") != std::string::npos,
                "Compiler CUDA did not reject unsupported " + name +
                    " at the reduction scheduling gate: " + diagnostic);
    };

    Var logits("logits", TensorType({1, 2}, "float32"));
    expect_rejection(
        Function({logits}, Call(relay::Op::Get("softmax"), {logits},
                                relay::SoftmaxAttrs::Create(-1))),
        "softmax reduction");

    Var lhs("lhs", TensorType({1, 2, 2}, "float32"));
    Var rhs("rhs", TensorType({1, 2, 2}, "float32"));
    expect_rejection(Function({lhs, rhs}, Call(relay::Op::Get("matmul"), {lhs, rhs})),
                     "batched matmul reduction");
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
        TestCompilerRejectsReductionGraphs(device);
        std::cout << "[PASS] compiler_rejects_reduction_graphs\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] cuda_runtime: " << error.what() << '\n';
        return 1;
    }
#else
    std::cout << "[SKIP] CUDA runtime tests: KXC_USE_CUDA=0\n";
#endif
    return 0;
}
