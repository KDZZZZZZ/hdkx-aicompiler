/*! \file test/codegen_llvm_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "api/compiler.h"
#include "relay/relay.h"
#include "relay/transforms/lower.h"
#include "tir/transforms/pipeline.h"
#include "codegen/codegen_c.h"

#ifdef KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/compiled_kernel.h"
#include "codegen/llvm_jit.h"
#endif

// ======== 辅助函数 ========
// 使用绝对误差比较 LLVM 执行结果中的单精度数值。
bool FloatNear(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// 直接构造 TIR 加法内核，验证 LLVM 降低、加载和 NDArray 参数调用。
void TestDirect_ElemwiseAdd() {
    std::cout << "=== Test: Direct TIR ElemwiseAdd → LLVM → JIT ===" << std::endl;

#ifdef KXC_USE_LLVM
    // 手动构建一个简单的 TIR PrimFunc:
    // void main(float* a, float* b, float* c) {
    //   for (int i = 0; i < 8; i++) {
    //     c[i] = a[i] + b[i];
    //   }
    // }
    using namespace kxc;
    using namespace kxc::tir;

    tir::Var a("a", DataType::Float(32));
    tir::Var b("b", DataType::Float(32));
    tir::Var c("c", DataType::Float(32));
    tir::Var i("i", DataType::Int(32));

    // c[i] = a[i] + b[i]
    PrimExpr load_a = Load(a, PrimExpr(i));
    PrimExpr load_b = Load(b, PrimExpr(i));
    PrimExpr sum = load_a + load_b;
    Stmt store = Store(c, sum, PrimExpr(i));

    // for (int i = 0; i < 8; i++)
    Stmt loop = For(i, IntImm(0, DataType::Int(32)),
                    IntImm(8, DataType::Int(32)), ForType::Serial, store);

    // PrimFunc
    Array<tir::Var> params = {a, b, c};
    Map<tir::Var, Buffer> buffer_map;
    buffer_map.Set(a, Buffer(a, DataType::Float(32),
                             {IntImm(8, DataType::Int(64))}, {},
                             IntImm(0), "a", 0, 0));
    buffer_map.Set(b, Buffer(b, DataType::Float(32),
                             {IntImm(8, DataType::Int(64))}, {},
                             IntImm(0), "b", 0, 0));
    buffer_map.Set(c, Buffer(c, DataType::Float(32),
                             {IntImm(8, DataType::Int(64))}, {},
                             IntImm(0), "c", 0, 0));

    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String("elemwise_add"));

    PrimFunc func(params, loop, buffer_map, attrs);

    // ---- LLVM Codegen ----
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
    kxc::codegen::CodeGenLLVM codegen(*llvm_ctx);
    codegen.AddFunction(func, "elemwise_add");

    // 打印LLVM IR
    std::string ir = codegen.DumpIR();
    std::cout << "Generated LLVM IR:\n" << ir << std::endl;

    // ---- JIT Compile ----
    auto module = codegen.TakeModule();

    kxc::codegen::LLVMJITEngine jit;
    auto kernel = jit.Compile(std::move(module), std::move(llvm_ctx),
                              "elemwise_add", /*opt_level=*/2);

    std::cout << "JIT compilation successful!" << std::endl;

    // ---- Execute ----
    float data_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float data_b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float data_c[8] = {0};

    std::vector<void*> args = {data_a, data_b, data_c};
    kernel(args);

    // ---- Verify ----
    bool ok = true;
    for (int idx = 0; idx < 8; ++idx) {
        float expected = data_a[idx] + data_b[idx];
        if (!FloatNear(data_c[idx], expected)) {
            std::cerr << "FAIL: c[" << idx << "] = " << data_c[idx]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "PASS: ElemwiseAdd results correct!" << std::endl;
        std::cout << "  c = [";
        for (int idx = 0; idx < 8; ++idx) {
            std::cout << data_c[idx];
            if (idx < 7) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
#else
    std::cout << "SKIPPED: KXC_USE_LLVM not enabled" << std::endl;
#endif
}

// 从 Relay 加法函数走完整 lowering 路径，并验证生成内核数值。
void TestRelay_ElemwiseAdd() {
    std::cout << "\n=== Test: Relay ElemwiseAdd → LowerToTIR → LLVM → JIT ===" << std::endl;

#ifdef KXC_USE_LLVM
    using namespace kxc;

    // 构建Relay IR: add(x, y)
    kxc::Var x("x", TensorType({8}, "float32"));
    kxc::Var y("y", TensorType({8}, "float32"));

    Call add_call(relay::Op::Get("add"), {x, y});
    Function func({x, y}, add_call);

    // Relay → TIR
    tir::PrimFunc pf = relay::LowerToTIR(func);
    std::cout << "LowerToTIR succeeded" << std::endl;

    // TIR优化
    pf = RunTIRPassPipeline(pf, {String("optimize_default")});
    std::cout << "TIR optimization succeeded" << std::endl;

    // LLVM Codegen
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
    kxc::codegen::CodeGenLLVM codegen(*llvm_ctx);
    codegen.AddFunction(pf, "relay_add");

    std::string ir = codegen.DumpIR();
    std::cout << "Generated LLVM IR:\n" << ir << std::endl;

    // JIT
    auto module = codegen.TakeModule();
    kxc::codegen::LLVMJITEngine jit;
    auto kernel = jit.Compile(std::move(module), std::move(llvm_ctx),
                              "relay_add", 2);
    std::cout << "JIT compilation successful!" << std::endl;

    // Execute
    float data_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float data_y[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float data_out[8] = {0};

    std::vector<void*> args = {data_x, data_y, data_out};
    kernel(args);

    // Verify
    bool ok = true;
    for (int idx = 0; idx < 8; ++idx) {
        float expected = data_x[idx] + data_y[idx];
        if (!FloatNear(data_out[idx], expected)) {
            std::cerr << "FAIL: out[" << idx << "] = " << data_out[idx]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "PASS: Relay ElemwiseAdd correct!" << std::endl;
    }
#else
    std::cout << "SKIPPED: KXC_USE_LLVM not enabled" << std::endl;
#endif
}

// 验证 C 源码后端生成循环、加法和返回语句。
void TestCCodegen() {
    std::cout << "\n=== Test: C Codegen (TIR → C source) ===" << std::endl;

    using namespace kxc;
    using namespace kxc::tir;

    // Build simple TIR: c[i] = a[i] + b[i]
    tir::Var a("a", DataType::Float(32));
    tir::Var b("b", DataType::Float(32));
    tir::Var c("c", DataType::Float(32));
    tir::Var i("i", DataType::Int(32));

    PrimExpr load_a = Load(a, PrimExpr(i));
    PrimExpr load_b = Load(b, PrimExpr(i));
    PrimExpr sum = load_a + load_b;
    Stmt store = Store(c, sum, PrimExpr(i));
    Stmt loop = For(i, IntImm(0, DataType::Int(32)),
                    IntImm(4, DataType::Int(32)), ForType::Serial, store);

    Array<tir::Var> params = {a, b, c};
    Map<tir::Var, Buffer> buffer_map;
    buffer_map.Set(a, Buffer(a, DataType::Float(32),
                             {IntImm(4, DataType::Int(64))}, {},
                             IntImm(0), "a", 0, 0));
    buffer_map.Set(b, Buffer(b, DataType::Float(32),
                             {IntImm(4, DataType::Int(64))}, {},
                             IntImm(0), "b", 0, 0));
    buffer_map.Set(c, Buffer(c, DataType::Float(32),
                             {IntImm(4, DataType::Int(64))}, {},
                             IntImm(0), "c", 0, 0));

    Map<String, ObjectRef> attrs;
    attrs.Set(String("global_symbol"), String("elemwise_add_c"));
    PrimFunc func(params, loop, buffer_map, attrs);

    // Generate C code
    kxc::codegen::CodeGenC codegen_c;
    std::string c_code = codegen_c.Generate(func, "elemwise_add_c");

    std::cout << "Generated C code:\n" << c_code << std::endl;

    // Verify it contains expected patterns
    bool has_for = c_code.find("for") != std::string::npos;
    bool has_add = c_code.find("+") != std::string::npos;
    bool has_return = c_code.find("return 0") != std::string::npos;

    if (has_for && has_add && has_return) {
        std::cout << "PASS: C code generation successful!" << std::endl;
    } else {
        std::cerr << "FAIL: C code missing expected patterns" << std::endl;
    }
}

// 验证 Compiler 公共 API 可编译并执行单内核 Relay 函数。
void TestCompilerAPI() {
    std::cout << "\n=== Test: Compiler API (Relay → Compile → Run) ===" << std::endl;

#ifdef KXC_USE_LLVM
    using namespace kxc;

    // 构建模型: add(x, y)
    kxc::Var x("x", TensorType({4}, "float32"));
    kxc::Var y("y", TensorType({4}, "float32"));
    Call add_call(relay::Op::Get("add"), {x, y});
    Function func({x, y}, add_call);

    // AOT 配置显式从 cpu:0 构造 Target，避免旧的设备类型/id 拼装路径。
    auto config = api::CompileConfig::AOT(BuildTarget(Device::CPU()), 2);
    auto module = api::Compiler::Compile(func, config);

    std::cout << "Status: " << module.GetStatus() << std::endl;

    // 运行
    float data_x[4] = {100, 200, 300, 400};
    float data_y[4] = {1, 2, 3, 4};
    float data_out[4] = {0};
    std::vector<void*> args = {data_x, data_y, data_out};
    module.Run(args);

    // 验证
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        float expected = data_x[i] + data_y[i];
        if (!FloatNear(data_out[i], expected)) {
            std::cerr << "FAIL: out[" << i << "] = " << data_out[i]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }
    if (ok) {
        std::cout << "PASS: Compiler API correct! out = ["
                  << data_out[0] << ", " << data_out[1] << ", "
                  << data_out[2] << ", " << data_out[3] << "]" << std::endl;
    }

    // 导出C源码
    std::filesystem::path c_source_path =
        std::filesystem::temp_directory_path() / "kxc_elemwise_add.c";
    module.SaveCSource(c_source_path.string());
    std::cout << "C source saved to " << c_source_path.string() << std::endl;
#else
    std::cout << "SKIPPED: KXC_USE_LLVM not enabled" << std::endl;
#endif
}

// 验证 Compiler 为多阶段表达式生成并正确使用中间存储。
void TestCompilerAPIIntermediateAllocate() {
    std::cout << "\n=== Test: Compiler API intermediate Allocate ===" << std::endl;

#ifdef KXC_USE_LLVM
    using namespace kxc;

    kxc::Var x("x", TensorType({4}, "float32"));
    kxc::Var y("y", TensorType({4}, "float32"));
    Call first_add(relay::Op::Get("add"), {x, y});
    Call second_add(relay::Op::Get("add"), {first_add, y});
    Function func({x, y}, second_add);

    auto config = api::CompileConfig::AOT(BuildTarget(Device::CPU()), 2);
    auto module = api::Compiler::Compile(func, config);

    float data_x[4] = {1, 2, 3, 4};
    float data_y[4] = {10, 20, 30, 40};
    float data_out[4] = {0};
    std::vector<void*> args = {data_x, data_y, data_out};
    module.Run(args);

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        float expected = data_x[i] + data_y[i] + data_y[i];
        if (!FloatNear(data_out[i], expected)) {
            std::cerr << "FAIL: allocated out[" << i << "] = " << data_out[i]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }
    if (ok) {
        std::cout << "PASS: Compiler API intermediate Allocate correct!" << std::endl;
    }
#else
    std::cout << "SKIPPED: KXC_USE_LLVM not enabled" << std::endl;
#endif
}

// 验证自适应运行时按输入 shape 选择、缓存并执行编译内核。
void TestAdaptiveRuntime() {
    std::cout << "\n=== Test: Adaptive Runtime (auto-compile + hot-swap) ===" << std::endl;

#ifdef KXC_USE_LLVM
    using namespace kxc;

    // 构建模型: add(x, y)
    kxc::Var x("x", TensorType({8}, "float32"));
    kxc::Var y("y", TensorType({8}, "float32"));
    Call add_call(relay::Op::Get("add"), {x, y});
    Function func({x, y}, add_call);

    // 自适应编译同样使用强类型 Device 构造 Target。
    auto config = api::CompileConfig::Adaptive(BuildTarget(Device::CPU()));
    auto module = api::Compiler::Compile(func, config);

    std::cout << "Initial status: " << module.GetStatus() << std::endl;

    // 第一次运行：自动触发同步编译
    float data_x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float data_y[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float data_out[8] = {0};
    std::vector<void*> args = {data_x, data_y, data_out};

    module.Run(args, {{8}, {8}});

    // 验证第一次运行结果
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        float expected = data_x[i] + data_y[i];
        if (!FloatNear(data_out[i], expected)) {
            std::cerr << "FAIL: out[" << i << "] = " << data_out[i]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "PASS: First adaptive run correct! out = ["
                  << data_out[0] << ", " << data_out[1] << ", ..."
                  << data_out[7] << "]" << std::endl;
    }

    // 多次运行同一shape（触发shape预测和后台编译）
    for (int run = 0; run < 5; ++run) {
        std::fill(data_out, data_out + 8, 0.0f);
        module.Run(args, {{8}, {8}});
    }

    std::cout << "After 6 runs: " << module.GetStatus() << std::endl;

    // 等待所有后台编译完成
    module.WaitAll();
    std::cout << "After WaitAll: " << module.GetStatus() << std::endl;

    // 再运行一次（应该命中缓存）
    std::fill(data_out, data_out + 8, 0.0f);
    module.Run(args, {{8}, {8}});

    ok = true;
    for (int i = 0; i < 8; ++i) {
        float expected = data_x[i] + data_y[i];
        if (!FloatNear(data_out[i], expected)) {
            std::cerr << "FAIL: cached run out[" << i << "] = " << data_out[i]
                      << ", expected " << expected << std::endl;
            ok = false;
        }
    }
    if (ok) {
        std::cout << "PASS: Cached adaptive run correct!" << std::endl;
    }

    // 测试预热
    module.WarmUp({{16}, {16}});
    std::cout << "After warmup: " << module.GetStatus() << std::endl;
    std::cout << "PASS: Adaptive runtime test completed!" << std::endl;
#else
    std::cout << "SKIPPED: KXC_USE_LLVM not enabled" << std::endl;
#endif
}

// 顺序执行 LLVM/C codegen 契约测试并汇总退出状态。
int main() {
    std::cout << "==== HDKX AI Compiler - Codegen Test ====" << std::endl;
    TestDirect_ElemwiseAdd();
    TestRelay_ElemwiseAdd();
    TestCCodegen();
    TestCompilerAPI();
    TestCompilerAPIIntermediateAllocate();
    TestAdaptiveRuntime();
    std::cout << "\n==== All tests completed ====" << std::endl;
    return 0;
}
