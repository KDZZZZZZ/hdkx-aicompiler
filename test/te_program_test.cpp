// P0/P1 immutable TE candidates and P2/P3 production static region execution.
#include <cmath>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "compiler/internal/execution_contract.h"
#include "compiler/internal/lowered_graph.h"
#include "compiler/internal/te_to_tir.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"
#include "kxc/te/program.h"
#include "kxc/te/topi.h"
#include "kxc/tir/printer/print_ir.h"
#if KXC_USE_LLVM
#include <llvm/Support/JSON.h>
#include <llvm/Support/Error.h>
#endif

namespace {
using namespace kxc;
using namespace api::internal;
namespace lower = relay::internal;

void Check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void Rejects(F fn, const std::string& message = "") {
    try { fn(); }
    catch (const std::exception& error) {
        Check(std::string(error.what()).find(message) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + message);
}

struct Fixture {
    Target target = BuildTarget(Device::CPU());
    std::string pipeline = std::string(ResolveCompilerExecutionContract(
        api::CompileConfig::Create(target, 0)).tir_pipeline.canonical_bytes);
    te::Tensor x, y, middle, output;
    te::Schedule schedule;
    explicit Fixture(const std::string& suffix = "", int64_t length = 10) {
        x = te::placeholder({tir::IntImm(length)}, tir::DataType::Float(32), "x" + suffix);
        y = te::placeholder({tir::IntImm(length)}, tir::DataType::Float(32), "y" + suffix);
        middle = te::topi::add(x, y, "add" + suffix);
        output = te::topi::sqrt(middle, "sqrt" + suffix);
        schedule = te::create_schedule({output->op});
    }
    te::Program Capture() const { return te::Program({x, y}, {}, {output}, schedule, target, pipeline); }
};

relay::LoweredFunction Lower(const te::Program& program, const Fixture& f) {
    return lower::LowerProgramToTIR(program, {}, f.target, f.pipeline,
        {"program_test", 0, "add+sqrt", 1, "test-semantic"});
}
std::string Text(const tir::PrimFunc& function) {
    std::ostringstream out;
    tir::printer::DumpPrimFunc(function, out);
    return out.str();
}

void SnapshotAndIdentity() {
    Fixture f, renamed("_different"), changed_length("", 11);
    const auto program = f.Capture();
    Check(program.defined() && !program.digest().empty(), "undefined Program identity");
    Check(program.canonical_bytes() == renamed.Capture().canonical_bytes(), "diagnostic names entered Program identity");
    Check(program.canonical_bytes() != changed_length.Capture().canonical_bytes(), "shape missing from Program identity");
    const auto before = Lower(program, f);
    const std::string original = Text(before->prim_func);
    Check(original == Text(Lower(f.Capture(), f)->prim_func), "repeated materialization is not deterministic");
    Check(lower::GetTEScheduleContract(before->prim_func) == program.canonical_bytes(),
          "materializer did not consume complete Program identity");
    Check(original.find("allocate") != std::string::npos, "region must expose current intermediate materialization");
    const api::UnitSemanticKey semantic("same-semantics");
    const auto key = BuildPrimitiveArtifactKey(semantic, f.target, f.pipeline,
        program.canonical_bytes().c_str(), "test-backend");
    Check(key.canonical_bytes().find("primitive-artifact-key-v4") != std::string::npos,
          "old artifact schema still used");
    auto stage = renamed.schedule[renamed.output->op];
    te::IterVar outer, inner;
    stage.split(stage->root_iter_vars[0], tir::IntImm(4), &outer, &inner);
    stage.parallel(outer);
    stage.vectorize(inner);
    const auto split = renamed.Capture();
    Check(split.canonical_bytes() != program.canonical_bytes(), "actual schedule decisions missing from identity");
    Check(BuildPrimitiveArtifactKey(semantic, f.target, f.pipeline,
        split.canonical_bytes().c_str(), "test-backend") != key, "cache aliases distinct Programs");
    Check(Text(Lower(split, renamed)->prim_func).find("if") != std::string::npos,
          "split tail predicate missing from Program lowering");
    Fixture changed_body;
    auto* compute = const_cast<te::ComputeOpNode*>(changed_body.middle->op.As<te::ComputeOpNode>());
    compute->body[0] = changed_body.x(compute->axis) - changed_body.y(compute->axis);
    Check(changed_body.Capture().canonical_bytes() != program.canonical_bytes(), "compute body missing from identity");
    Fixture changed_pipeline;
    changed_pipeline.pipeline += "different-explicit-pipeline";
    Check(changed_pipeline.Capture().canonical_bytes() != program.canonical_bytes(), "pipeline missing from identity");
    Rejects([&] { Lower(program, changed_pipeline); }, "mismatch");
    Target other = BuildTarget(Device::CPU());
    const_cast<TargetNode*>(other.operator->())->device_id = 1;
    Rejects([&] { lower::LowerProgramToTIR(program, {}, other, f.pipeline, {}); }, "mismatch");

    // Original TE handles and returned mutable TIR must not mutate the Program.
    auto original_stage = f.schedule[f.output->op];
    original_stage.parallel(original_stage->root_iter_vars[0]);
    const_cast<te::ComputeOpNode*>(f.middle->op.As<te::ComputeOpNode>())->body[0] = tir::FloatImm(123.0f);
    const_cast<te::TensorNode*>(f.x.operator->())->shape[0] = tir::IntImm(11);
    Check(Text(Lower(program, f)->prim_func) == original, "source mutation changed frozen Program");
    for (const auto& item : before->prim_func->buffer_map) const_cast<tir::BufferNode*>(item.second.operator->())->shape[0] = tir::IntImm(999);
    Check(Text(Lower(program, f)->prim_func) == original, "returned TIR leaked mutable Program state");
    std::cout << "[PASS] Program snapshot, name-independent identity, body/schedule/shape/pipeline isolation, deterministic TIR\n";
}

void InvalidCandidates() {
    Fixture f;
    Rejects([&] { te::Program({}, {}, {f.output}, f.schedule, f.target, f.pipeline); }, "undeclared");
    Rejects([&] { te::Program({f.x, f.x}, {}, {f.output}, f.schedule, f.target, f.pipeline); }, "unique");
    Rejects([&] { te::Program({f.x, f.y}, {f.x}, {f.output}, f.schedule, f.target, f.pipeline); }, "disjoint");
    Rejects([&] { te::Program({f.x, f.y}, {}, {f.output, f.output}, f.schedule, f.target, f.pipeline); }, "duplicate output");
    Rejects([&] { te::Program({f.x, f.y}, {}, {f.output}, f.schedule, Target(), f.pipeline); }, "Target");
    Rejects([&] { te::Program({f.x, f.y}, {}, {f.output}, f.schedule, f.target, ""); }, "pipeline");
    Rejects([&] { te::Program({f.x, f.y}, {}, {f.output}, f.schedule, f.target, f.pipeline, {f.x}); }, "read by");
    auto free = te::compute(f.x->shape, [&](const Array<tir::Var>& axes) {
        return f.x(axes) + tir::Var("unbound", tir::DataType::Float(32));
    });
    auto free_schedule = te::create_schedule({free->op});
    Rejects([&] { te::Program({f.x}, {}, {free}, free_schedule, f.target, f.pipeline); }, "free variable");
    Fixture invalid;
    auto stage = invalid.schedule[invalid.output->op];
    te::IterVar outer, inner;
    stage.split(stage->root_iter_vars[0], tir::IntImm(4), &outer, &inner);
    const_cast<te::IterVarNode*>(inner.operator->())->dom_extent = tir::IntImm(7);
    Rejects([&] { invalid.Capture(); }, "axis facts");
    Fixture duplicate_var;
    auto duplicate_stage = duplicate_var.schedule[duplicate_var.output->op];
    duplicate_stage.split(duplicate_stage->root_iter_vars[0], tir::IntImm(4), &outer, &inner);
    const_cast<te::IterVarNode*>(inner.operator->())->var = outer->var;
    Rejects([&] { duplicate_var.Capture(); }, "axis order");
    Fixture malformed_target;
    const_cast<TargetNode*>(malformed_target.target.operator->())->kind = "unknown";
    Rejects([&] { malformed_target.Capture(); }, "Target");
    Fixture invalid_index;
    const auto* original = invalid_index.middle->op.As<te::ComputeOpNode>();
    const_cast<te::ComputeOpNode*>(original)->body[0] = invalid_index.x(tir::FloatImm(0.0f));
    Rejects([&] { invalid_index.Capture(); }, "scalar integer");
    Fixture reordered;
    std::swap(reordered.schedule->stages[0], reordered.schedule->stages[1]);
    Rejects([&] { reordered.Capture(); }, "order");
    Fixture cyclic;
    auto* compute = const_cast<te::ComputeOpNode*>(cyclic.middle->op.As<te::ComputeOpNode>());
    compute->body[0] = cyclic.output(compute->axis);
    Rejects([&] { cyclic.Capture(); }, "topological");
    // Restore the test-created ownership cycle after rejection.
    compute->body[0] = tir::FloatImm(0.0f);
    Rejects([&] { te::Program().canonical_bytes(); }, "undefined");
    std::cout << "[PASS] malformed boundaries, dependencies, schedules and binding mismatch rejected\n";
}

Function Graph(const std::string& suffix = "", Array<int64_t> shape = {10},
               const std::string& dtype = "float32", int variant = 0) {
    Var x("x" + suffix, TensorType(shape, dtype));
    Var y("y" + suffix, TensorType(variant == 3 ? Array<int64_t>{1} : shape, dtype));
    Expr middle = Call(relay::Op::Get(variant == 4 ? "mul" : "add"), {x, variant == 5 ? Expr(x) : Expr(y)});
    Expr output = Call(relay::Op::Get("sqrt"), {middle});
    if (variant == 1) output = Tuple({middle, output});
    if (variant == 2) output = Tuple({output, Call(relay::Op::Get("nn_relu"), {middle})});
    return relay::InferTypePass(Function(variant == 5 ? Array<Var>{x} : Array<Var>{x, y}, output));
}

void PartitionProof() {
    auto plain = PartitionValueGraph(BuildValueGraph(Graph()));
    auto fused = PartitionValueGraph(BuildValueGraph(Graph()), true);
    Check(plain.units.size() == 2 && fused.units.size() == 1 && fused.units[0].producer,
          "static region did not replace two ordinary units");
    Check(fused.units[0].boundary_input_value_ids.size() == 2 && fused.units[0].output_value_ids.size() == 1,
          "region ABI includes internal producer storage");
    auto renamed = PartitionValueGraph(BuildValueGraph(Graph("_renamed")), true);
    Check(fused.units[0].semantic_key == renamed.units[0].semantic_key, "graph names entered region semantic identity");
    for (int variant : {1, 2, 3, 4}) {
        const auto graph = PartitionValueGraph(BuildValueGraph(Graph("", {10}, "float32", variant)), true);
        Check(graph.units.size() >= 2, "observable, shared, broadcast or non-add producer was fused");
        for (const auto& unit : graph.units) Check(!unit.producer, "unsupported region was admitted");
    }
    auto repeated = PartitionValueGraph(BuildValueGraph(Graph("", {10}, "float32", 5)), true);
    Check(repeated.units.size() == 1 && repeated.units[0].boundary_input_value_ids.size() == 1,
          "repeated logical arguments were not preserved through boundary deduplication");
    fused.units[0].producer->output_value_ids[0] = fused.units[0].output_value_ids[0];
    Rejects([&] { ValidatePartition(fused); });
    std::cout << "[PASS] canonical two-Call region, shared/observable/broadcast rejection and repeated arguments\n";
}

#if KXC_USE_LLVM
void CheckBundle(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream input(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(input.good(), "missing TE Program runtime bundle");
    std::map<std::string, size_t> submits, executions, runs;
    std::string line;
    while (std::getline(input, line)) {
        auto parsed = llvm::json::parse(line);
        if (!parsed) throw std::runtime_error(llvm::toString(parsed.takeError()));
        const auto* event = parsed->getAsObject();
        Check(event, "profile event must be an object");
        const auto* fields = event->getObject("fields");
        auto case_name = fields ? fields->getString("te_program_case") : std::nullopt;
        if (!case_name) continue;
        const std::string name = case_name->str();
        const auto type = event->getString("event_type");
        Check(bool(type), "profile event type missing");
        if (*type == "kernel_submit") ++submits[name];
        if (*type == "kernel_exec") {
            Check(event->getString("phase") == "complete" && event->getString("status") == "ok", "kernel completion was not successful");
            ++executions[name];
        }
        if (*type == "runtime_session_run" && (name == "reference" || name == "fused")) {
            Check(event->getString("status") == "ok", "runtime session failed");
            const auto* metrics = event->getObject("metrics");
            Check(metrics && metrics->getNumber("submit_count") == (name == "reference" ? 2 : 1), "runtime submit count mismatch");
            ++runs[name];
        }
    }
    Check(submits["reference"] == 2 && executions["reference"] == 2 && runs["reference"] == 1 &&
          submits["fused"] == 1 && executions["fused"] == 1 && runs["fused"] == 1,
          "bundle does not prove two reference calls and one completed fused call");
    Check(submits["bad_arity"] == 0 && submits["bad_shape"] == 0 &&
          executions["bad_arity"] == 0 && executions["bad_shape"] == 0, "malformed ABI launched work");
}

template<class T> runtime::NDArray Tensor(const std::vector<T>& values, const Array<int64_t>& shape) {
    auto result = runtime::NDArray::Empty(shape, runtime::DataTypeFromString(sizeof(T) == 4 ? "float32" : "float64"), Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(T));
    return result;
}
template<class T> void Numeric(const Array<int64_t>& shape) {
    size_t count = 1;
    for (const auto extent : shape) count *= extent;
    const std::string dtype = sizeof(T) == 4 ? "float32" : "float64";
    auto cpu = BuildTarget(Device::CPU());
    profiling::ProfileOptions options;
    options.enabled = true;
    options.record_pass_ir = false;
    options.bundle_dir = (std::filesystem::current_path() / "out" / "te_program" /
        (dtype + "_n" + std::to_string(count) + "_rank" + std::to_string(shape.size()))).string();
    auto context = profiling::ProfileContext::Create(options);
    profiling::ActivationScope activation(context, "te_program_compile");
    auto baseline = api::Compiler::Compile(Graph("", shape, dtype), api::CompileConfig::Create(cpu, 2));
    auto fused = api::Compiler::Compile(Graph("", shape, dtype), api::CompileConfig::Create(cpu, 3));
    Check(baseline.plan().calls().size() == 2 && fused.plan().calls().size() == 1, "production did not apply opt-level fusion policy");
    Check(baseline.plan().values().size() == 4 && fused.plan().values().size() == 3, "internal value remains in runtime storage plan");
    Check(baseline.plan().input_value_ids().size() == fused.plan().input_value_ids().size() &&
          baseline.plan().output_value_ids()[0] == fused.plan().output_value_ids()[0], "external plan boundary changed");
    std::vector<T> x(count), y(count);
    for (size_t i = 0; i < count; ++i) { x[i] = static_cast<T>(i % 13) - T(4); y[i] = T(5) + static_cast<T>(i % 3); }
    auto a = Tensor(x, shape), b = Tensor(y, shape);
    runtime::RuntimeSession reference(baseline.module(), baseline.plan()), actual(fused.module(), fused.plan());
    auto expected = reference.Run({a, b}, {{"te_program_case", "reference"}})[0];
    auto result = actual.Run({a, b}, {{"te_program_case", "fused"}})[0];
    std::vector<T> before(count), after(count);
    expected.CopyToBytes(before.data(), count * sizeof(T));
    result.CopyToBytes(after.data(), count * sizeof(T));
    Check(count == 0 || std::memcmp(before.data(), after.data(), count * sizeof(T)) == 0, "fused/unfused results differ");
    for (size_t i = 0; i < count; ++i) Check(std::abs(after[i] - std::sqrt(x[i] + y[i])) < T(1e-5), "fused result differs from scalar reference");
    Rejects([&] { actual.Run({a}, {{"te_program_case", "bad_arity"}}); });
    Rejects([&] { actual.Run({Tensor<T>({T(1), T(2)}, {2}), b}, {{"te_program_case", "bad_shape"}}); });
    CheckBundle(context);
    std::cout << "[PASS] LLVM " << dtype << " elements=" << count << " kernels=2->1 runtime_values=4->3 bitwise_equal=1\n";
}

void CacheAndLifetime() {
    ClearPrimitiveCacheForTesting();
    auto cpu = BuildTarget(Device::CPU());
    const auto before = GetPrimitiveCacheStats();
    auto first = api::Compiler::Compile(Graph(), api::CompileConfig::Create(cpu, 3));
    const auto compiled = GetPrimitiveCacheStats();
    auto second = api::Compiler::Compile(Graph("_other"), api::CompileConfig::Create(cpu, 3));
    const auto reused = GetPrimitiveCacheStats();
    Check(compiled.misses == before.misses + 1 && reused.hits == compiled.hits + 1 && reused.misses == compiled.misses,
          "equivalent production Program was not reused from cache");
    Check(first.artifact_pins()[0].record().artifact_key == second.artifact_pins()[0].record().artifact_key,
          "equivalent region artifact identity changed");
    auto different = api::Compiler::Compile(Graph("", {11}), api::CompileConfig::Create(cpu, 3));
    Check(GetPrimitiveCacheStats().misses == reused.misses + 1, "different input length reused incompatible artifact");
    runtime::RuntimeSession retained(first.module(), first.plan());
    first = {}; second = {}; different = {};
    ClearPrimitiveCacheForTesting();
    const auto cleared = GetPrimitiveCacheStats();
    std::vector<float> values(10, 2.0f), actual(10);
    const auto input = Tensor(values, {10});
    auto operation = retained.RunAsync({input, input}, DeviceStream::Default(Device::CPU()));
    operation.completion.Wait();
    operation.outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    for (float value : actual) Check(value == 2.0f, "session lost fused artifact after cache eviction");
    const auto after = GetPrimitiveCacheStats();
    Check(after.hits == cleared.hits && after.misses == cleared.misses && after.entries == cleared.entries,
          "runtime performed cache lookup or compilation");
    std::cout << "[PASS] production cache miss=1/hit=1, length isolation, asynchronous lifetime after eviction, zero runtime cache access\n";
}

void ConstantRelocation() {
    ClearPrimitiveCacheForTesting();
    const auto graph = [](float weight, bool shift) {
        const TensorType type({10}, "float32");
        Var x("data", type), unused("unused", type);
        Constant constant(Tensor(std::vector<float>(10, weight), {10}));
        Expr output = Call(relay::Op::Get("sqrt"), {Call(relay::Op::Get("add"), {x, constant})});
        return Function(shift ? Array<Var>{unused, x} : Array<Var>{x}, output);
    };
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 3);
    auto first = api::Compiler::Compile(graph(1, false), config);
    const auto before = GetPrimitiveCacheStats();
    auto second = api::Compiler::Compile(graph(5, true), config);
    Check(first.artifact_pins().size() == 1 && second.artifact_pins().size() == 1 &&
          first.artifact_pins()[0].record().artifact_key == second.artifact_pins()[0].record().artifact_key &&
          GetPrimitiveCacheStats().hits == before.hits + 1,
          "graph-local constant/value ids prevented fused artifact reuse");
    runtime::RuntimeSession a(first.module(), first.plan()), b(second.module(), second.plan());
    first = {}; second = {}; ClearPrimitiveCacheForTesting();
    const auto input = Tensor(std::vector<float>(10, 3), {10});
    std::vector<float> actual_a(10), actual_b(10);
    a.Run({input})[0].CopyToBytes(actual_a.data(), 40);
    b.Run({input, input})[0].CopyToBytes(actual_b.data(), 40);
    for (size_t i = 0; i < 10; ++i) {
        Check(actual_a[i] == 2.0f && actual_b[i] == std::sqrt(8.0f),
              "fused artifact reused stale or wrongly relocated constants");
    }
    const auto repeated = api::Compiler::Compile(Graph("", {10}, "float32", 5), config);
    runtime::RuntimeSession twice(repeated.module(), repeated.plan());
    twice.Run({input})[0].CopyToBytes(actual_a.data(), 40);
    for (float value : actual_a) Check(value == std::sqrt(6.0f), "fused repeated argument lost multiplicity");
    std::cout << "[PASS] graph-local id/constant relocation, distinct retained payloads, repeated-argument LLVM execution\n";
}
#endif
}  // namespace

int main() {
    try {
        SnapshotAndIdentity(); InvalidCandidates(); PartitionProof();
#if KXC_USE_LLVM
        Numeric<float>({10}); Numeric<float>({2, 5}); Numeric<float>({0}); Numeric<float>({});
        Numeric<double>({10}); Numeric<float>({1}); CacheAndLifetime(); ConstantRelocation();
#else
        std::cout << "[SKIP] LLVM unavailable: P2/P3 not claimed\n";
#endif
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
    return 0;
}
