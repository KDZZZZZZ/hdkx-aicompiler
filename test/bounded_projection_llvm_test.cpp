// Immutable weights, proven broadcast, and fixed-axis RMSNorm feed ordinary
// bounded MatMul through the production compiler/module/session path.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Action>
void Rejects(Action action, const std::string& message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

NDArray Tensor(const std::vector<int64_t>& dimensions, int seed = 0) {
    size_t count = 1;
    for (int64_t extent : dimensions) count *= static_cast<size_t>(extent);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = std::sin((i + seed + 1) * 0.13) * 0.5f;
    NDArray result = NDArray::Empty(dimensions, runtime::DataTypeFromString("float32"), Device::CPU());
    if (count) result.CopyFromBytes(values.data(), count * sizeof(float));
    return result;
}

Expr Scalar(float value) {
    NDArray tensor = Tensor({});
    tensor.CopyFromBytes(&value, sizeof(value));
    return Constant(tensor);
}

std::vector<float> Read(const NDArray& array) {
    std::vector<float> result(array.NBytes() / sizeof(float));
    if (!result.empty()) array.CopyToBytes(result.data(), array.NBytes());
    return result;
}

std::vector<float> ReadFile(const std::filesystem::path& path, size_t count) {
    std::ifstream stream(path, std::ios::binary);
    Check(stream.good(), "missing fixture file: " + path.string());
    std::vector<float> values(count);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    Check(stream.gcount() == static_cast<std::streamsize>(count * sizeof(float)) &&
              stream.peek() == std::char_traits<char>::eof(), "fixture byte length mismatch: " + path.string());
    return values;
}

bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
           a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
           a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
           a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}

std::vector<restricted::InputAxisSymbol> Axes() {
    return {{0, 0, "B", 1, 3, 1}, {0, 1, "S", 1, 8, 1}};
}

struct Model {
    Function function;
    std::vector<NDArray> weights;
};

Model NormalizedProjection() {
    std::vector<NDArray> weights{Tensor({8}, 2), Tensor({6, 8}, 3),
                               Tensor({8, 4}, 4), Tensor({8, 4}, 5)};
    const Var x("activations", TensorType({1, 4, 8}, "float32"));
    const Expr cast = Call(relay::Op::Get("cast"), {x}, relay::CastAttrs::Create(0));
    const Expr square = Call(relay::Op::Get("pow"), {cast, Scalar(2.0f)});
    const Expr mean = Call(relay::Op::Get("reduce_mean"), {square},
                           relay::ReduceMeanAttrs::Create({-1}, 1));
    const Expr epsilon = Call(relay::Op::Get("add"), {mean, Scalar(1e-6f)});
    const Expr root = Call(relay::Op::Get("sqrt"), {epsilon});
    // A constant is the first logical operand but follows data in the ABI.
    const Expr reciprocal = Call(relay::Op::Get("divide"), {Scalar(1.0f), root});
    const Expr normalized = Call(relay::Op::Get("mul"), {cast, reciprocal});
    const Expr scaled = Call(relay::Op::Get("mul"), {Constant(weights[0]), normalized});
    // A wholly static weight transform and dynamic calls share one plan.
    const Expr wq = Call(relay::Op::Get("transpose"), {Constant(weights[1])},
                         relay::TransposeAttrs::Create({1, 0}));
    const Expr q = Call(relay::Op::Get("matmul"), {scaled, wq});
    const Expr k = Call(relay::Op::Get("matmul"), {scaled, Constant(weights[2])});
    const Expr v = Call(relay::Op::Get("matmul"), {scaled, Constant(weights[3])});
    return {Function({x}, Tuple({scaled, q, k, v})), std::move(weights)};
}

double CheckReference(const Array<NDArray>& outputs, const NDArray& input,
                      const std::vector<std::vector<float>>& weights) {
    Check(outputs.size() == 4, "normalized projection must return four tensors");
    const int64_t batch = input.shape()[0], sequence = input.shape()[1];
    const auto data = Read(input);
    double worst = 0;
    std::vector<double> normalized(data.size());
    for (size_t row = 0; row < data.size() / 8; ++row) {
        double square_sum = 0;
        for (size_t d = 0; d < 8; ++d) square_sum += static_cast<double>(data[row * 8 + d]) * data[row * 8 + d];
        for (size_t d = 0; d < 8; ++d) normalized[row * 8 + d] =
            data[row * 8 + d] * weights[0][d] / std::sqrt(square_sum / 8 + static_cast<double>(1e-6f));
    }
    for (size_t output = 0; output < 4; ++output) {
        const int64_t width = output == 0 ? 8 : output == 1 ? 6 : 4;
        const auto dims = outputs[output].shape();
        Check(dims.size() == 3 && dims[0] == batch && dims[1] == sequence && dims[2] == width,
              "projection output shape mismatch");
        const auto actual = Read(outputs[output]);
        for (size_t i = 0; i < actual.size(); ++i) {
            const size_t row = i / width, column = i % width;
            double expected = output == 0 ? normalized[i] : 0;
            if (output) for (size_t d = 0; d < 8; ++d) {
                const size_t weight_index = output == 1 ? column * 8 + d : d * width + column;
                expected += normalized[row * 8 + d] * weights[output][weight_index];
            }
            worst = std::max(worst, std::abs(actual[i] - expected));
            Check(std::isfinite(actual[i]) && std::abs(actual[i] - expected) < 1e-5,
                  "projection disagrees with independent double RMSNorm/MatMul");
        }
    }
    return worst;
}

void TestNormalizedProjection(const api::CompileConfig& config,
                              const std::shared_ptr<profiling::ProfileContext>& context) {
    Model source = NormalizedProjection();
    std::vector<std::vector<float>> reference_weights;
    for (const auto& weight : source.weights) reference_weights.push_back(Read(weight));
    const auto prepared = Adapter::Prepare(source.function, config, Axes());
    const auto key = prepared.graph_template().key();
    for (auto& weight : source.weights) {
        const std::vector<float> zeros(weight.NBytes() / sizeof(float), 0.0f);
        weight.CopyFromBytes(zeros.data(), weight.NBytes());
    }
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    Check(prepared.graph_template().key() == key, "caller weight mutation changed the frozen template");
    const auto compiled = api::Compiler::CompileBounded(request);
    Check(compiled.plan().input_value_ids().size() == 1 && compiled.plan().constant_value_ids().size() == 7 &&
              compiled.plan().calls().size() == 12, "weight transforms or constant roles were lost");
    auto constant_copies = compiled.module().constants();
    for (auto& item : constant_copies) {
        const std::vector<float> zeros(item.second.NBytes() / sizeof(float), 0.0f);
        item.second.CopyFromBytes(zeros.data(), item.second.NBytes());
    }
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto abi = api::BuildPlanAbiFingerprint(compiled);
    double worst = 0;
    for (const auto& dims : std::vector<std::vector<int64_t>>{{1, 1, 8}, {1, 4, 8}, {2, 3, 8}, {3, 8, 8}}) {
        const NDArray input = Tensor(dims, 11);
        const auto outputs = session.Run({input}, {{"stage", "synthetic_projection"},
            {"batch", std::to_string(dims[0])}, {"sequence", std::to_string(dims[1])}, {"plan_abi", abi.digest()}});
        worst = std::max(worst, CheckReference(outputs, input, reference_weights));
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "weighted Run performed compile/cache work");
    }
    const auto counts = [&] {
        context->Flush();
        std::ifstream stream(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
        Check(stream.good(), "projection bundle missing");
        std::pair<size_t, size_t> result{0, 0};
        std::string line;
        while (std::getline(stream, line)) {
            if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++result.first;
            if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++result.second;
        }
        return result;
    };
    Check(counts().first == 48, "four projection runs must execute 48 LLVM kernels");
    const auto reject_input = [&](Array<NDArray> inputs) {
        const auto before = counts();
        Rejects([&] { (void)session.Run(inputs); }, "invalid projection input was accepted");
        Check(counts() == before, "invalid projection input allocated or launched");
    };
    reject_input({Tensor({0, 3, 8})});
    reject_input({Tensor({4, 3, 8})});
    reject_input({Tensor({1, 0, 8})});
    reject_input({Tensor({1, 9, 8})});
    reject_input({Tensor({1, 3, 7})});
    reject_input({Tensor({3, 8})});
    reject_input({NDArray::Zeros({1, 3, 8}, runtime::DataTypeFromString("int64"), Device::CPU())});
    reject_input({Tensor({1, 3, 8}), Tensor({8, 4})});
    Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "rejected weighted Run compiled");

    const auto decision = Adapter::MintExact(prepared, Adapter::BindingsFromInputShapes(prepared, {{2, 5, 8}}));
    const auto exact = api::Compiler::Compile(Adapter::MaterializeExactFunction(prepared, decision), config);
    Adapter::VerifyCompiledExactVariant(prepared, decision, exact);
    const NDArray exact_input = Tensor({2, 5, 8}, 11);
    const runtime::RuntimeSession exact_session(exact.module(), exact.plan());
    CheckReference(exact_session.Run({exact_input}), exact_input, reference_weights);
    const auto changed = Adapter::Prepare(source.function, config, Axes());
    Check(changed.graph_template().key() != key, "changing weight payload must change graph identity");
    std::cout << "weighted projection: 4 bounded shapes, 48 launches, worst_abs=" << worst
              << "; 8 zero-launch rejections; frozen weights and exact profile verified\n";
}

void TestRealMiniMindProjection(const api::CompileConfig& config, bool heads = false) {
    const char* variable = heads ? "KXC_MINIMIND_HEADS_DIR" : "KXC_MINIMIND_PROJECTION_DIR";
    const char* directory = std::getenv(variable);
    if (!directory) {
        std::cout << "[SKIP] actual MiniMind: set " << variable << '\n';
        return;
    }
    const std::filesystem::path root(directory);
    const auto imported = heads
        ? frontend::LoadONNXShapeSource((root / "heads.json").string(), (root / "heads.params").string())
        : frontend::LoadONNXImportSpec((root / "projection.json").string(), (root / "projection.params").string());
    Check(imported.function->params.size() == 1 && imported.output_names.size() == 3,
          "actual projection must have one activation input and ordered Q/K/V outputs");
    const auto* type = imported.function->params[0]->type_annotation.As<TensorTypeNode>();
    Check(type && type->dtype == "float32" && type->shape.size() == 3 && type->shape[2] == 768,
          "actual MiniMind hidden width must be 768");
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
        Adapter::Prepare(imported.function, config, Axes(), imported.declared_output_types)));
    const size_t expected_calls = heads ? 33 : 11;
    Check(compiled.plan().calls().size() == expected_calls,
          "actual RMSNorm/QKV call count differs after shape-chain resolution");
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto abi = api::BuildPlanAbiFingerprint(compiled);
    std::ifstream receipt(root / "export_receipt.txt");
    std::string export_receipt;
    std::getline(receipt, export_receipt);
    Check(export_receipt.size() == 64 &&
              export_receipt.find_first_not_of("0123456789abcdef") == std::string::npos,
          "actual projection must identify its exported dynamic subgraph");
    std::ifstream cases(root / "cases.txt");
    Check(cases.good(), "actual projection cases missing");
    size_t case_index = 0;
    int64_t batch = 0, sequence = 0;
    double worst = 0;
    while (cases >> batch >> sequence) {
        Check(batch >= 1 && batch <= 3 && sequence >= 1 && sequence <= 8 && case_index < 4,
              "actual projection case lies outside its fixture contract");
        const auto raw = ReadFile(root / ("input_" + std::to_string(case_index) + ".bin"), batch * sequence * 768);
        NDArray input = Tensor({batch, sequence, 768});
        input.CopyFromBytes(raw.data(), input.NBytes());
        const auto outputs = session.Run({input}, {{"stage", heads ? "minimind_heads" : "minimind_projection"},
            {"export_receipt", export_receipt}, {"batch", std::to_string(batch)},
            {"sequence", std::to_string(sequence)}, {"plan_abi", abi.digest()}});
        Check(outputs.size() == 3, "actual projection result count changed");
        for (size_t output = 0; output < outputs.size(); ++output) {
            const int64_t width = output == 0 ? 768 : 384;
            const auto dims = outputs[output].shape();
            Check(dims.size() == (heads ? 4 : 3) && dims[0] == batch && dims[1] == sequence &&
                      (heads ? dims[2] == width / 96 && dims[3] == 96 : dims[2] == width),
                  "actual projection output shape/order mismatch");
            const auto reference = ReadFile(root / ("ref_" + std::to_string(case_index) + "_" + std::to_string(output) + ".bin"), batch * sequence * width);
            const auto actual = Read(outputs[output]);
            for (size_t i = 0; i < actual.size(); ++i) {
                const double difference = std::abs(static_cast<double>(actual[i]) - reference[i]);
                worst = std::max(worst, difference);
                Check(std::isfinite(actual[i]) && difference < 1e-4, "actual projection differs from ONNX ReferenceEvaluator");
            }
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "actual projection Run compiled kernels");
        ++case_index;
    }
    Check(case_index == 4 && cases.eof(), "actual projection fixture must contain four cases");
    std::cout << "actual MiniMind " << (heads ? "normalized heads" : "RMSNorm/QKV")
              << ": 4 shapes, " << 4 * expected_calls << " LLVM launches, worst_abs=" << worst << '\n';
}

void TestProofRejections(const api::CompileConfig& config) {
    const auto before = ci::GetPrimitiveCacheStats();
    const Var x("x", TensorType({1, 4, 8}, "float32"));
    const auto reject = [&](Expr body) {
        Rejects([&] { (void)Adapter::Prepare(Function({x}, body), config, Axes()); },
                "invalid weighted graph was admitted");
    };
    reject(Call(relay::Op::Get("add"), {x, Constant(Tensor({4, 8}))}));
    reject(Call(relay::Op::Get("matmul"), {x, Constant(Tensor({7, 4}))}));
    reject(Call(relay::Op::Get("reduce_mean"), {x}, relay::ReduceMeanAttrs::Create({1}, 1)));
    reject(Call(relay::Op::Get("reduce_mean"), {x}, relay::ReduceMeanAttrs::Create({2, -1}, 1)));
    reject(Call(relay::Op::Get("reduce_mean"), {x}, relay::ReduceMeanAttrs::Create({2}, 2)));
    reject(Tuple({Call(relay::Op::Get("nn_relu"), {x}), Constant(Tensor({8}))}));
    Check(SameStats(before, ci::GetPrimitiveCacheStats()), "failed constant/broadcast proofs touched cache");
}
}  // namespace

int main() {
    try {
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true;
        options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path() / "out" / "bounded_projection_profile").string();
        const auto context = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context, "projection_validation");
        const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2, options);
        TestNormalizedProjection(config, context);
        TestRealMiniMindProjection(config);
        TestRealMiniMindProjection(config, true);
        TestProofRejections(config);
        std::cout << "All bounded projection tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
