/*! \file test/profile_bundle_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "base/profiling.h"
#include "relay/op.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"
#include "tir/transforms/pipeline.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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
    tir::PrimFunc tir_out = relay::LowerToTIR(relay_out);
    tir_out = tir::RunTIRPassPipeline(tir_out, {String("optimize_default")});
    (void)tir_out;

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
    std::ifstream ifs(events);
    std::string line;
    while (std::getline(ifs, line)) {
        saw_relay_pass = saw_relay_pass || line.find("\"component\":\"relay_pass\"") != std::string::npos;
        saw_lower = saw_lower || line.find("\"event_type\":\"lower_to_tir\"") != std::string::npos;
        saw_tir_pass = saw_tir_pass || line.find("\"component\":\"tir_pass\"") != std::string::npos;
    }

    if (!saw_relay_pass || !saw_lower || !saw_tir_pass) {
        std::cerr << "Bundle did not contain expected relay/lower/tir events\n";
        return 1;
    }

    std::cout << "Bundle generated at: " << bundle_dir.string() << "\n";
    return 0;
}
