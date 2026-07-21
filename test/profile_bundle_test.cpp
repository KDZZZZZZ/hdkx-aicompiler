/*! \file test/profile_bundle_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "base/profiling.h"
#include "api/compiler.h"
#include "relay/op.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"
#include "tir/transforms/pipeline.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    using namespace kxc;

    const fs::path bundle_dir = fs::current_path() / "profile_bundle_test_output";
    std::error_code ec;
    fs::remove_all(bundle_dir, ec);

    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = bundle_dir.string();
    options.ir_capture_mode = profiling::IRCaptureMode::kVerbose;

    auto ctx = profiling::ProfileContext::Create(options);
    const std::string run_id = ctx->NextRunId("test");
    profiling::ActivationScope activation(ctx, run_id);

    profiling::EventSpec root_spec;
    root_spec.component = "test";
    root_spec.event_type = "profile_bundle_smoke";
    profiling::ScopedSpan root_span(ctx, std::move(root_spec), run_id);

    Var x("x", TensorType({4}, "float32"));
    Var y("y", TensorType({4}, "float32"));
    Call add_call(relay::Op::Get("add"), {x, y});
    Function func({x, y}, add_call);

    Function relay_out =
        relay::RunRelayPassPipeline(func, {String("optimize_default")});
    tir::PrimFunc tir_out = relay::LowerToTIR(relay_out)->prim_func;
    tir_out = tir::RunTIRPassPipeline(tir_out, {String("optimize_default")});
    (void)tir_out;

    api::CompileConfig config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()), 1);
#if KXC_USE_LLVM
    // LLVM 构建额外经过真实 Compiler，锁定七个显式阶段都进入同一 bundle。
    api::CompiledModule module = api::Compiler::Compile(func, config);
    if (!module.IsReady()) {
        std::cerr << "Compiler profiling fixture did not produce a ready module\n";
        return 1;
    }
#endif

    // 无效 Function 必须在 validate 阶段失败，并把阶段名、状态和原因写入同一 bundle。
    bool compile_failure_observed = false;
    try {
        (void)api::Compiler::Compile(Function({}, Expr()), config);
    } catch (const std::exception& error) {
        const std::string message = error.what();
        compile_failure_observed =
            message.find("Compiler stage 'validate' failed") != std::string::npos &&
            message.find("must have a body") != std::string::npos;
    }
    if (!compile_failure_observed) {
        std::cerr << "Compiler validate failure did not preserve stage context\n";
        return 1;
    }

    ctx->Flush();

    const fs::path manifest = bundle_dir / "manifest.json";
    const fs::path events = bundle_dir / "events.jsonl";
    const fs::path trace = bundle_dir / "trace.json";
    const fs::path summary = bundle_dir / "summary.json";
    const fs::path diagnosis_json = bundle_dir / "diagnosis.json";
    const fs::path diagnosis_md = bundle_dir / "diagnosis.md";
    const fs::path artifacts = bundle_dir / "artifacts";

    const auto require_file = [](const fs::path& path) {
        if (!fs::exists(path) || !fs::is_regular_file(path)) {
            std::cerr << "Missing expected file: " << path.string() << "\n";
            return false;
        }
        return true;
    };

    if (!require_file(manifest) || !require_file(events) || !require_file(trace) ||
        !require_file(summary) || !require_file(diagnosis_json) ||
        !require_file(diagnosis_md)) {
        return 1;
    }
    if (!fs::exists(artifacts) || !fs::is_directory(artifacts)) {
        std::cerr << "Missing artifacts directory: " << artifacts.string() << "\n";
        return 1;
    }

    bool saw_relay_pass = false;
    bool saw_lower = false;
    bool saw_tir_pass = false;
    bool saw_validate_failure = false;
#if KXC_USE_LLVM
    const std::vector<std::string> compiler_stages = {
        "validate", "optimize_relay", "lower", "optimize_tir",
        "build_signature", "build_backend", "assemble"};
    size_t next_compiler_stage = 0;
    bool compiler_stage_fields_valid = true;
#endif
    std::ifstream ifs(events);
    std::string line;
    while (std::getline(ifs, line)) {
        saw_relay_pass = saw_relay_pass || line.find("\"component\":\"relay_pass\"") != std::string::npos;
        saw_lower = saw_lower || line.find("\"event_type\":\"lower_to_tir\"") != std::string::npos;
        saw_tir_pass = saw_tir_pass || line.find("\"component\":\"tir_pass\"") != std::string::npos;
        saw_validate_failure =
            saw_validate_failure ||
            (line.find("\"component\":\"compiler\"") != std::string::npos &&
             line.find("\"pass_name\":\"validate\"") != std::string::npos &&
             line.find("\"status\":\"error\"") != std::string::npos &&
             line.find("must have a body") != std::string::npos);
#if KXC_USE_LLVM
        // 事件按阶段 span 关闭顺序写入，顺序检查同时证明管线没有跳步。
        if (next_compiler_stage < compiler_stages.size() &&
            line.find("\"pass_name\":\"" +
                          compiler_stages[next_compiler_stage] + "\"") !=
                std::string::npos) {
            const std::string& stage = compiler_stages[next_compiler_stage];
            // 每个阶段都必须带共同编译身份；后续阶段再逐步补充 IR、symbol 和 backend。
            bool fields_valid =
                line.find("\"target_kind\":\"llvm\"") != std::string::npos &&
                line.find("\"device_type\":\"0\"") != std::string::npos &&
                line.find("\"device_id\":\"0\"") != std::string::npos &&
                line.find("\"opt_level\":\"1\"") != std::string::npos;
            if (stage != "validate") {
                fields_valid = fields_valid &&
                               line.find("\"ir_hash\":") != std::string::npos;
            }
            if (stage == "build_signature" || stage == "build_backend" ||
                stage == "assemble") {
                fields_valid = fields_valid &&
                               line.find("\"symbol\":") != std::string::npos;
            }
            if (stage == "build_backend" || stage == "assemble") {
                fields_valid = fields_valid &&
                               line.find("\"backend\":\"llvm\"") !=
                                   std::string::npos;
            }
            compiler_stage_fields_valid = compiler_stage_fields_valid && fields_valid;
            ++next_compiler_stage;
        }
#endif
    }

    if (!saw_relay_pass || !saw_lower || !saw_tir_pass) {
        std::cerr << "Bundle did not contain expected relay/lower/tir events\n";
        return 1;
    }
    if (!saw_validate_failure) {
        std::cerr << "Bundle did not contain the failed Compiler validate span\n";
        return 1;
    }
#if KXC_USE_LLVM
    if (next_compiler_stage != compiler_stages.size()) {
        std::cerr << "Bundle did not contain the ordered Compiler stage events\n";
        return 1;
    }
    if (!compiler_stage_fields_valid) {
        std::cerr << "Compiler stage events did not contain their required fields\n";
        return 1;
    }
#endif

    std::cout << "Bundle generated at: " << bundle_dir.string() << "\n";
    return 0;
}
