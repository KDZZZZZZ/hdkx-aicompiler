// MiniMind-V with runtime image placement: one bounded prefill for N=0..3 images.
//
// The fixed joint export bakes one marker layout into each artifact. Here the
// host replaces each 64-marker run with slot indices into a capacity-shaped
// visual buffer, so a single S-bounded prefill artifact serves every image
// count and text layout. The vision chain keeps its fixed per-image contract
// and the decode keeps the session-owned capacity KV state. Nothing compiles
// after the three artifacts are built.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;
namespace fs = std::filesystem;

struct Profile final {
    int64_t vocab = 0, marker = 0, image_tokens = 0, slot_images = 0, sequence_max = 0, capacity = 0;
    float sentinel = 0;
    int64_t cases = 0, steps = 0;
    int64_t slot_rows() const { return slot_images * image_tokens; }
};

void Check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class Action> void Rejects(Action action, const std::string& diagnostic) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "wrong rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + diagnostic);
}
bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
        a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
        a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
NDArray Empty(Array<int64_t> shape, const std::string& dtype = "float32") {
    return NDArray::Empty(shape, runtime::DataTypeFromString(dtype), Device::CPU());
}
template<class T> NDArray FromValues(Array<int64_t> shape, const std::vector<T>& values, const std::string& dtype) {
    auto array = Empty(shape, dtype);
    Check(array.NBytes() == values.size() * sizeof(T), "host tensor size mismatch");
    array.CopyFromBytes(values.data(), array.NBytes());
    return array;
}
template<class T> std::vector<T> Read(const NDArray& array) {
    std::vector<T> values(array.NBytes() / sizeof(T));
    if (!values.empty()) array.CopyToBytes(values.data(), array.NBytes());
    return values;
}
template<class T> std::vector<T> ReadFile(const fs::path& file, size_t count) {
    std::ifstream in(file, std::ios::binary);
    Check(in.good(), "missing fixture: " + file.string());
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    Check(in.gcount() == static_cast<std::streamsize>(count * sizeof(T)) &&
          in.peek() == std::char_traits<char>::eof(), "fixture byte length mismatch: " + file.string());
    return values;
}
double Worst(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& what) {
    Check(actual.size() == expected.size(), what + " size mismatch");
    double worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        Check(std::isfinite(actual[i]) && std::isfinite(expected[i]), what + " is nonfinite");
        worst = std::max(worst, std::abs(double(actual[i]) - expected[i]));
    }
    return worst;
}

// Host placement contract: each image is one contiguous run of exactly
// image_tokens markers, and at most slot_images images fit the buffer. The
// compiled Gather zero-fills out-of-range indices, so this check must run
// before any launch rather than rely on the kernel.
std::vector<int64_t> SlotIds(const std::vector<int64_t>& ids, const Profile& profile, int64_t* images) {
    Check(!ids.empty() && static_cast<int64_t>(ids.size()) <= profile.sequence_max,
          "slot prefill sequence is outside the bounded profile");
    std::vector<int64_t> result(ids);
    *images = 0;
    for (size_t index = 0; index < ids.size();) {
        Check(ids[index] >= 0 && ids[index] < profile.vocab, "token id outside the text vocabulary");
        if (ids[index] != profile.marker) { ++index; continue; }
        const size_t start = index;
        while (index < ids.size() && ids[index] == profile.marker) ++index;
        Check(static_cast<int64_t>(index - start) == profile.image_tokens,
              "each image must be one contiguous run of image markers");
        Check(*images < profile.slot_images, "image count exceeds the visual slot capacity");
        for (size_t offset = 0; offset < index - start; ++offset)
            result[start + offset] = profile.vocab + *images * profile.image_tokens + static_cast<int64_t>(offset);
        ++*images;
    }
    return result;
}

std::pair<size_t, size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(fs::path(context->bundle_dir()) / "events.jsonl");
    Check(in.good(), "missing VLM bounded profile");
    std::pair<size_t, size_t> counts{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++counts.second;
    }
    return counts;
}

Function PrepareJitRelayFunction(Function function) {
    function = relay::InferTypePass(function);
    function = relay::RunRelayPassPipeline(function, {String("fold_constant"), String("simplify_expr")});
    return relay::InferTypePass(function);
}

void Run(const fs::path& root, const std::shared_ptr<profiling::ProfileContext>& context,
         const profiling::ProfileOptions& options) {
    Profile profile;
    {
        std::ifstream in(root / "profile.txt");
        Check(static_cast<bool>(in >> profile.vocab >> profile.marker >> profile.image_tokens >> profile.slot_images >>
                                profile.sequence_max >> profile.capacity >> profile.sentinel >> profile.cases >>
                                profile.steps),
              "profile.txt must hold the nine slot profile fields");
        Check(profile.vocab == 6400 && profile.marker == 12 && profile.image_tokens == 64 &&
              profile.slot_images == 3 && profile.sequence_max + profile.steps <= profile.capacity &&
              std::abs(profile.sentinel) <= 1e6f && profile.steps >= 3,
              "the fixture must keep the locked slot profile and a mask-safe sentinel");
    }
    std::string receipt;
    { std::ifstream in(root / "export_receipt.txt"); in >> receipt; }
    Check(receipt.rfind("sha256:prefill_slots.onnx:", 0) == 0, "missing slot prefill export receipt");
    const Device cpu = Device::CPU();

    // 1. Vision: fixed per-image contract, ordinary static compile.
    const auto vision_imported = frontend::LoadONNXImportSpec((root/"vision.json").string(), (root/"vision.params").string());
    Check(vision_imported.input_names.size() == 1 && vision_imported.output_names.size() == 1,
          "vision stage must map pixels to visual tokens");
    const auto vision = api::Compiler::Compile(vision_imported.function,
        api::CompileConfig::Create(BuildTarget(cpu), 2, options));
    // 2. Prefill: one bounded artifact over the text length S.
    const auto source = frontend::LoadONNXShapeSource((root/"prefill_slots.json").string(), (root/"prefill_slots.params").string());
    Check(source.input_names.size() == 2 && source.input_names[0] == "input_ids" &&
          source.input_names[1] == "visual_slots" && source.declared_output_types.size() == 17,
          "slot prefill boundary drifted");
    const std::vector<restricted::InputAxisSymbol> axes{{0, 1, "S", 1, profile.sequence_max, 1}};
    const auto config = api::CompileConfig::Create(BuildTarget(cpu), 2, options);
    const auto prepared = Adapter::Prepare(source.function, config, axes, source.declared_output_types);
    const auto prefill = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    size_t extent_calls = 0;
    for (const auto& call : prefill.plan().calls()) {
        for (const auto& arg : prefill.module().signature(call->symbol).arguments())
            if (arg->role == codegen::KernelArgRole::kRuntimeExtent) { ++extent_calls; break; }
    }
    Check(extent_calls > 0, "slot prefill silently specialized every primitive");
    // 3. Decode: capacity KV owned by the session.
    const auto decode_imported = frontend::LoadONNXImportSpec((root/"decode_capacity.json").string(),
                                                              (root/"decode_capacity.params").string());
    Check(decode_imported.input_names.size() == 19 && decode_imported.input_names[0] == "input_ids" &&
          decode_imported.input_names[1] == "position" && decode_imported.input_names[2] == "attention_mask",
          "capacity decode boundary drifted");
    const auto decode = api::Compiler::Compile(PrepareJitRelayFunction(decode_imported.function),
        api::CompileConfig::Create(BuildTarget(cpu), 0, options));
    std::vector<runtime::StateOutputBinding> bindings;
    const auto decode_inputs = decode.plan().input_value_ids();
    const auto decode_outputs = decode.plan().output_value_ids();
    for (size_t index = 0; index < 16; ++index)
        bindings.push_back({decode_inputs[index + 3], decode_outputs[index + 1], 1, profile.capacity, 1});
    const auto state_plan = decode.plan().BindStateOutputs(bindings, profile.sentinel);
    std::cout << "artifacts: vision " << vision.plan().calls().size() << " calls, bounded slot prefill "
              << prefill.plan().calls().size() << " calls (" << extent_calls << " with runtime extents), capacity decode "
              << decode.plan().calls().size() << " calls\n" << std::flush;

    runtime::RuntimeSession vision_session(vision.module(), vision.plan());
    runtime::RuntimeSession prefill_session(prefill.module(), prefill.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const int64_t hidden = 768, slot = 4 * 96, rows = profile.slot_rows();
    double vision_worst = 0, prefill_worst = 0, decode_worst = 0, state_worst = 0;
    size_t vision_runs = 0, prefill_runs = 0, decode_runs = 0, sentinel_values = 0;
    std::vector<int64_t> images_seen, sequences_seen;

    for (int64_t case_index = 0; case_index < profile.cases; ++case_index) {
        const fs::path dir = root / ("case" + std::to_string(case_index));
        int64_t images = 0, sequence = 0;
        { std::ifstream in(dir / "case.txt"); Check(static_cast<bool>(in >> images >> sequence), "invalid case.txt"); }
        const auto ids = ReadFile<int64_t>(dir/"input_ids.bin", sequence);
        int64_t scanned = 0;
        const auto slot_ids = SlotIds(ids, profile, &scanned);
        Check(scanned == images && slot_ids == ReadFile<int64_t>(dir/"slot_ids.bin", sequence),
              "host slot scan must reproduce the exported placement");
        const auto metadata = [&](const std::string& stage) {
            return runtime::ExecutionMetadata{{"model","minimind-v"},{"stage",stage},{"export_receipt",receipt},
                {"images",std::to_string(images)},{"sequence",std::to_string(sequence)},
                {"shape_case",std::to_string(case_index)}};
        };
        // Per-image vision fills the visual slots; unused rows keep a sentinel.
        const auto pixels = ReadFile<float>(dir/"pixel_values.bin", images * 3 * 256 * 256);
        const auto visual_reference = ReadFile<float>(dir/"visual_tokens.bin", images * profile.image_tokens * hidden);
        std::vector<float> slots(rows * hidden, profile.sentinel);
        for (int64_t image = 0; image < images; ++image) {
            const std::vector<float> one(pixels.begin() + image * 3 * 256 * 256, pixels.begin() + (image + 1) * 3 * 256 * 256);
            const auto out = vision_session.Run({FromValues<float>({1,3,256,256}, one, "float32")}, metadata("vision"));
            ++vision_runs;
            const auto tokens = Read<float>(out[0]);
            const std::vector<float> expected(visual_reference.begin() + image * profile.image_tokens * hidden,
                                              visual_reference.begin() + (image + 1) * profile.image_tokens * hidden);
            vision_worst = std::max(vision_worst, Worst(tokens, expected, "visual tokens"));
            std::copy(tokens.begin(), tokens.end(), slots.begin() + image * profile.image_tokens * hidden);
        }
        const auto run_prefill = [&](const std::vector<float>& buffer) {
            ++prefill_runs;
            return prefill_session.Run({FromValues<int64_t>({1,sequence}, slot_ids, "int64"),
                                        FromValues<float>({rows,hidden}, buffer, "float32")}, metadata("prefill"));
        };
        const auto outputs = run_prefill(slots);
        Check(outputs.size() == 17, "slot prefill must return logits and all 16 K/V outputs");
        for (size_t output = 0; output < outputs.size(); ++output) {
            const size_t width = output == 0 ? static_cast<size_t>(profile.vocab) : static_cast<size_t>(slot);
            const auto actual = Read<float>(outputs[output]);
            prefill_worst = std::max(prefill_worst, Worst(actual,
                ReadFile<float>(dir/("prefill_ref_"+std::to_string(output)+".bin"), sequence * width), "prefill output"));
        }
        // Unused slot rows must never be read: another fill gives identical bits.
        if (images < profile.slot_images) {
            std::vector<float> other(slots);
            std::fill(other.begin() + images * profile.image_tokens * hidden, other.end(), -1234.5f);
            const auto replay = run_prefill(other);
            for (size_t output = 0; output < outputs.size(); ++output) {
                const auto a = Read<float>(outputs[output]), b = Read<float>(replay[output]);
                Check(a == b, "unused visual slots changed the prefill outputs");
                sentinel_values += a.size();
            }
        }
        // Decode continues from the compiled prefill K/V in a session-owned cache.
        runtime::RuntimeSession session(decode.module(), state_plan);
        std::vector<const void*> addresses;
        for (size_t index = 0; index < bindings.size(); ++index) {
            session.InitializeState(bindings[index].state_value_id, outputs[index + 1], sequence);
            addresses.push_back(session.StateValue(bindings[index].state_value_id).storage().data());
        }
        const auto logits = Read<float>(outputs[0]);
        auto argmax = [&](const std::vector<float>& row, size_t offset) {
            return static_cast<int64_t>(std::max_element(row.begin() + offset, row.begin() + offset + profile.vocab) -
                                        row.begin() - offset);
        };
        int64_t token = argmax(logits, (sequence - 1) * profile.vocab), extent = sequence;
        std::ifstream step_lines(dir / "steps.txt");
        for (int64_t step = 0; step < profile.steps; ++step) {
            int64_t index = 0, expected_token = 0, expected_extent = 0;
            Check(static_cast<bool>(step_lines >> index >> expected_token >> expected_extent) && index == step &&
                  expected_token == token && expected_extent == extent,
                  "host greedy token or extent diverged from the reference chain");
            std::vector<float> mask(profile.capacity + 1, 0.0f);
            std::fill(mask.begin(), mask.begin() + extent, 1.0f);
            mask[profile.capacity] = 1.0f;
            auto step_metadata = metadata("decode");
            step_metadata["state_extent"] = std::to_string(extent);
            const auto step_out = session.Run({FromValues<int64_t>({1,1}, {token}, "int64"),
                FromValues<int64_t>({1}, {extent}, "int64"), FromValues<float>({1,profile.capacity+1}, mask, "float32")},
                step_metadata);
            ++decode_runs;
            Check(step_out.size() == 1, "stateful decode must keep present tensors private");
            const auto step_logits = Read<float>(step_out[0]);
            decode_worst = std::max(decode_worst, Worst(step_logits,
                ReadFile<float>(dir/("ref_step"+std::to_string(step)+"_logits.bin"), profile.vocab), "decode logits"));
            for (size_t state = 0; state < bindings.size(); ++state) {
                const auto id = bindings[state].state_value_id;
                Check(session.StateValue(id).storage().data() == addresses[state] && session.StateExtent(id) == extent + 1,
                      "decode must keep cache addresses and commit one slot per step");
                state_worst = std::max(state_worst, Worst(Read<float>(session.StateValue(id)),
                    ReadFile<float>(dir/("ref_step"+std::to_string(step)+"_state_"+std::to_string(state)+".bin"),
                                    profile.capacity * slot), "decode state"));
            }
            token = argmax(step_logits, 0);
            ++extent;
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "a VLM case touched the primitive cache");
        images_seen.push_back(images);
        sequences_seen.push_back(sequence);
        std::cout << "case " << case_index << ": N=" << images << " S=" << sequence << " prefill+"
                  << profile.steps << " decode steps\n" << std::flush;
    }
    Check(prefill_worst < 1e-4 && decode_worst < 1e-4 && state_worst < 1e-4 && vision_worst < 1e-4,
          "VLM outputs differ from the upstream references");
    std::sort(images_seen.begin(), images_seen.end());
    Check(images_seen == std::vector<int64_t>{0, 1, 2, 3}, "the fixture must cover N=0..3 on one artifact");

    // Host placement rejections happen before any tensor exists.
    std::vector<int64_t> short_run(70, 100);
    std::fill(short_run.begin() + 2, short_run.begin() + 65, profile.marker);
    int64_t ignored = 0;
    Rejects([&] { (void)SlotIds(short_run, profile, &ignored); }, "contiguous run");
    // Four images need 256 markers, beyond the S bound; a widened host profile
    // still reaches the independent slot-capacity check.
    std::vector<int64_t> four(4 * 65 + 1, 100);
    for (int64_t image = 0; image < 4; ++image)
        std::fill(four.begin() + 1 + image * 65, four.begin() + 1 + image * 65 + 64, profile.marker);
    Rejects([&] { (void)SlotIds(four, profile, &ignored); }, "bounded profile");
    Profile widened = profile;
    widened.sequence_max = static_cast<int64_t>(four.size());
    Rejects([&] { (void)SlotIds(four, widened, &ignored); }, "slot capacity");
    Rejects([&] { (void)SlotIds(std::vector<int64_t>(3, profile.vocab), profile, &ignored); }, "vocabulary");
    Rejects([&] { (void)SlotIds(std::vector<int64_t>(profile.sequence_max + 1, 100), profile, &ignored); }, "bounded profile");
    // Runtime rejections: outside the prepared contract, zero launches or allocations.
    const auto counts = Counts(context);
    const std::vector<float> zeros(rows * hidden, 0.0f);
    for (const auto& inputs : std::vector<Array<NDArray>>{
             {Empty({1, profile.sequence_max + 1}, "int64"), Empty({rows, hidden})},
             {Empty({1, 0}, "int64"), Empty({rows, hidden})},
             {Empty({2, 4}, "int64"), Empty({rows, hidden})},
             {Empty({1, 4}, "int32"), Empty({rows, hidden})},
             {Empty({1, 4}, "int64"), Empty({rows - 1, hidden})},
             {Empty({1, 4}, "int64"), Empty({rows, hidden}, "float64")},
             {Empty({1, 4}, "int64")}}) {
        Rejects([&] { (void)prefill_session.Run(inputs); }, "RuntimeSession");
        Check(Counts(context) == counts && SameStats(stats, ci::GetPrimitiveCacheStats()),
              "an invalid slot prefill input allocated, launched or compiled");
    }
    std::cout << "[PASS] minimind_vlm_bounded: one bounded prefill served N=0..3, S=";
    for (size_t i = 0; i < sequences_seen.size(); ++i) std::cout << (i ? "/" : "") << sequences_seen[i];
    std::cout << "; vision_runs=" << vision_runs << " prefill_runs=" << prefill_runs << " decode_runs=" << decode_runs
              << " vision_max_abs=" << vision_worst << " prefill_max_abs=" << prefill_worst
              << " decode_max_abs=" << decode_worst << " state_max_abs=" << state_worst
              << " unused_slot_values_bit_identical=" << sentinel_values
              << "; five host and seven runtime rejections, no runtime compile\n";
}
}  // namespace

int main() {
    try {
        const char* directory = std::getenv("KXC_MINIMIND_VLM_BOUNDED_DIR");
        if (!directory || !*directory) {
            std::cout << "[SKIP] set KXC_MINIMIND_VLM_BOUNDED_DIR\n";
            return 77;
        }
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true;
        options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.record_pass_ir = false;
        options.bundle_dir = (fs::current_path()/"out"/"minimind_vlm_bounded_profile").string();
        options = profiling::ApplyEnvironmentOverrides(options);
        const auto context = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(context, "minimind_vlm_bounded");
        Run(directory, context, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
