/*! \file test/kv_state_llvm_test.cpp
 * \brief M2 dynamic stateful KV contract: production-compiled append/read
 *        kernels, session-owned extents, and the prefill + multi-step decode
 *        acceptance on real CPU/LLVM modules.
 */

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/module_invocation_contract.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/kv_state_plan.h"
#include "../src/codegen/llvm/internal/llvm_jit.h"
#include "kxc/runtime/session.h"

namespace {

using namespace kxc;
using namespace kxc::api;
using namespace kxc::codegen;
namespace runtime = kxc::runtime;
using runtime::NDArray;

int g_failures = 0;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << '\n'; \
            ++g_failures;                                                     \
            return false;                                                     \
        }                                                                     \
    } while (false)

bool Throws(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    } catch (...) {
        return true;
    }
    return false;
}

constexpr int64_t kBatch = 1;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadDim = 4;
constexpr int64_t kCapacity = 8;
constexpr int64_t kMaxAppend = 3;
const double kSentinelFill = -1.5e30;

KvStatePlanDeclaration CacheDeclaration() {
    KvStatePlanDeclaration declaration;
    declaration.layout = KvStateLayout{kBatch, kHeads, kHeadDim, kCapacity};
    declaration.max_append_tokens = kMaxAppend;
    declaration.state_count = 1;
    declaration.invalid_fill = kSentinelFill;
    declaration.read_kind = KvStateReadKind::kValidRegion;
    return declaration;
}

KvStatePlanDeclaration AttentionDeclaration() {
    KvStatePlanDeclaration declaration = CacheDeclaration();
    declaration.state_count = 2;
    declaration.read_kind = KvStateReadKind::kCausalAttention;
    return declaration;
}

CompileConfig CpuConfig() {
    const char* level = std::getenv("KXC_KV_OPT");
    return CompileConfig::Create(BuildTarget(Device::CPU()),
                                level ? std::atoi(level) : 2);
}

DLDataType F32() { return runtime::DataTypeFromString("float32"); }
DLDataType U64() { return DLDataType{kDLUInt, 64, 1}; }

NDArray F32Tensor(const Array<int64_t>& shape) {
    return NDArray::Empty(shape, F32(), Device::CPU(), 8);
}

NDArray CountTensor(uint64_t count) {
    NDArray result = NDArray::Empty({1}, U64(), Device::CPU(), 8);
    result.CopyFromBytes(&count, sizeof(count));
    return result;
}

std::vector<float> ReadF32(const NDArray& value) {
    std::vector<float> result(static_cast<size_t>(value.NBytes() / sizeof(float)));
    if (!result.empty()) value.CopyToBytes(result.data(), value.NBytes());
    return result;
}

/*! \brief Tokens tensor [1, J, H, D]; only the first n slots carry declared
 *  values, the rest is finite junk the kernels must never let leak. */
NDArray TokenTensor(int64_t used_tokens, bool junk_sentinel_free = true) {
    NDArray tokens = F32Tensor({kBatch, kMaxAppend, kHeads, kHeadDim});
    std::vector<float> values;
    values.reserve(static_cast<size_t>(kMaxAppend * kHeads * kHeadDim));
    for (int64_t token = 0; token < kMaxAppend; ++token) {
        for (int64_t head = 0; head < kHeads; ++head) {
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                if (token < used_tokens) {
                    values.push_back(100.0F + static_cast<float>(token) +
                                     0.25F * static_cast<float>(head) +
                                     0.0625F * static_cast<float>(dim));
                } else {
                    values.push_back(junk_sentinel_free
                                         ? -777.0F - static_cast<float>(token)
                                         : std::numeric_limits<float>::quiet_NaN());
                }
            }
        }
    }
    tokens.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return tokens;
}

NDArray QueryTensor(int64_t used_tokens) {
    NDArray queries = F32Tensor({kBatch, kMaxAppend, kHeads, kHeadDim});
    std::vector<float> values;
    values.reserve(static_cast<size_t>(kMaxAppend * kHeads * kHeadDim));
    for (int64_t token = 0; token < kMaxAppend; ++token) {
        for (int64_t head = 0; head < kHeads; ++head) {
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                if (token < used_tokens) {
                    values.push_back(0.5F + 0.125F * static_cast<float>(token) +
                                     0.03125F * static_cast<float>(head) -
                                     0.015625F * static_cast<float>(dim));
                } else {
                    values.push_back(1e3F + static_cast<float>(token));
                }
            }
        }
    }
    queries.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return queries;
}

/*! \brief Independent host reference of the cache after appending n tokens. */
std::vector<float> ReferenceAppend(const std::vector<float>& cache,
                                   const std::vector<float>& tokens,
                                   int64_t length, int64_t count) {
    std::vector<float> result = cache;
    result.resize(static_cast<size_t>(kCapacity * kHeads * kHeadDim), 0.0F);
    const int64_t token_elements = kHeads * kHeadDim;
    for (int64_t token = 0; token < count; ++token) {
        for (int64_t element = 0; element < token_elements; ++element) {
            result[static_cast<size_t>((length + token) * token_elements +
                                       element)] =
                tokens[static_cast<size_t>(token * token_elements + element)];
        }
    }
    return result;
}

/*! \brief Independent host reference of the masked valid-region read. */
std::vector<float> ReferenceValidRead(const std::vector<float>& cache,
                                      int64_t valid_length) {
    std::vector<float> result(static_cast<size_t>(kCapacity * kHeads * kHeadDim),
                              0.0F);
    for (int64_t slot = 0; slot < valid_length; ++slot) {
        for (int64_t element = 0; element < kHeads * kHeadDim; ++element) {
            result[static_cast<size_t>(slot * kHeads * kHeadDim + element)] =
                cache[static_cast<size_t>(slot * kHeads * kHeadDim + element)];
        }
    }
    return result;
}

bool SamePrimitiveCacheStats(const internal::PrimitiveCacheStats& left,
                             const internal::PrimitiveCacheStats& right) {
    return left.hits == right.hits && left.misses == right.misses &&
           left.entries == right.entries &&
           left.accounted_bytes == right.accounted_bytes &&
           left.evictions == right.evictions &&
           left.in_flight == right.in_flight &&
           left.merged_waiters == right.merged_waiters &&
           left.failures == right.failures &&
           left.rejections == right.rejections &&
           left.active_pins == right.active_pins;
}

bool CloseEnough(float actual, float expected, float tolerance = 1e-4F) {
    return std::fabs(actual - expected) <= tolerance;
}

/*! \brief Independent host reference of softmax(scale * qK) V over the valid
 *  region; query slots >= n read as zero, exactly like the kernel. */
std::vector<float> ReferenceAttention(const std::vector<float>& cache,
                                      const std::vector<float>& queries,
                                      int64_t valid_length, int64_t used_queries) {
    const int64_t token_elements = kHeads * kHeadDim;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> result(static_cast<size_t>(kMaxAppend * kHeads * kHeadDim),
                              0.0F);
    for (int64_t query = 0; query < kMaxAppend; ++query) {
        if (query >= used_queries) continue;
        for (int64_t head = 0; head < kHeads; ++head) {
            std::vector<double> scores(static_cast<size_t>(kCapacity), -1e30);
            double max_score = -1e30;
            for (int64_t slot = 0; slot < valid_length; ++slot) {
                double dot = 0.0;
                for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                    dot += static_cast<double>(
                               queries[static_cast<size_t>(
                                   query * token_elements + head * kHeadDim + dim)]) *
                           static_cast<double>(
                               cache[static_cast<size_t>(slot * token_elements +
                                                         head * kHeadDim + dim)]);
                }
                scores[static_cast<size_t>(slot)] = scale * dot;
                max_score = std::max(max_score, scale * dot);
            }
            double denominator = 0.0;
            std::vector<double> weights(static_cast<size_t>(kCapacity), 0.0);
            for (int64_t slot = 0; slot < valid_length; ++slot) {
                weights[static_cast<size_t>(slot)] =
                    std::exp(scores[static_cast<size_t>(slot)] - max_score);
                denominator += weights[static_cast<size_t>(slot)];
            }
            for (int64_t dim = 0; dim < kHeadDim; ++dim) {
                double sum = 0.0;
                for (int64_t slot = 0; slot < valid_length; ++slot) {
                    sum += weights[static_cast<size_t>(slot)] *
                           static_cast<double>(cache[static_cast<size_t>(
                               slot * token_elements + head * kHeadDim + dim)]);
                }
                result[static_cast<size_t>(query * token_elements +
                                           head * kHeadDim + dim)] =
                    static_cast<float>(sum / denominator);
            }
        }
    }
    return result;
}

bool SameRegion(const std::vector<float>& actual,
                const std::vector<float>& expected, int64_t begin, int64_t end,
                float tolerance, const std::string& message) {
    for (int64_t index = begin; index < end; ++index) {
        if (!CloseEnough(actual[static_cast<size_t>(index)],
                         expected[static_cast<size_t>(index)], tolerance)) {
            std::cerr << "[FAIL] " << message << " at " << index << ": actual "
                      << actual[static_cast<size_t>(index)] << " expected "
                      << expected[static_cast<size_t>(index)] << '\n';
            ++g_failures;
            return false;
        }
    }
    return true;
}

/*! \brief S1/S2 shared declaration checks against the generated artifacts. */
bool CheckGeneratedPlan(const KvStatePlan& compiled, int64_t state_count,
                        int64_t call_count) {
    const runtime::ExecutablePlan& plan = compiled.plan;
    CHECK(compiled.module.IsReady(), "stateful module must be ready");
    CHECK(plan.mode() == runtime::ExecutablePlanMode::kDynamicStatefulV1,
          "compiled plan must use the dynamic stateful mode");
    CHECK(plan.state_value_ids().size() == static_cast<size_t>(state_count),
          "compiled plan must declare the requested states");
    CHECK(plan.calls().size() == static_cast<size_t>(call_count),
          "compiled plan must declare one call per generated kernel");
    CHECK(compiled.module.entry_count() == static_cast<size_t>(call_count),
          "compiled module must hold one entry per kernel");
    CHECK(compiled.artifacts.size() == static_cast<size_t>(call_count),
          "compiled plan must expose one artifact identity per call");
    CHECK(plan.state_count_input_value_id() >= 0,
          "compiled plan must declare the append-count input");
    CHECK(plan.state_extent_bindings().size() == plan.calls().size(),
          "compiled plan must bind extents for every call");
    int64_t capacity_values = 0;
    for (const runtime::ValueSpec& value : plan.values()) {
        if (value->is_state) {
            ++capacity_values;
            CHECK(value->state_capacity == kCapacity &&
                      value->state_extent_axis == 1,
                  "compiled states must declare capacity 8 on the seq axis");
            CHECK(value->state_fill == kSentinelFill,
                  "compiled states must carry the sentinel fill");
        }
    }
    CHECK(capacity_values == state_count,
          "compiled plan state count must match the declaration");
    for (const runtime::KernelCall& call : plan.calls()) {
        const KernelSignature signature = compiled.module.signature(call->symbol);
        signature.Validate();
        size_t extents = 0;
        for (const KernelArgSpec& argument : signature.arguments()) {
            if (argument->role == KernelArgRole::kRuntimeExtent) {
                ++extents;
                CHECK(argument->dtype.code == kDLUInt && argument->dtype.bits == 64 &&
                          argument.shape().size() == 1 && argument.shape()[0] == 1,
                      "state extent ABI must be a uint64[1] buffer");
            }
        }
        const ModuleInvocationContract& invocation =
            internal::BorrowCompiledModuleInvocationContract(compiled.module,
                                                             call->symbol);
        CHECK(invocation.runtime_extent_scalars().size() == extents &&
                  std::all_of(invocation.runtime_extent_scalars().begin(),
                              invocation.runtime_extent_scalars().end(),
                              [](const ModuleRuntimeExtentScalar& scalar) {
                                  return scalar.source ==
                                         ModuleRuntimeExtentScalar::Source::
                                             kStateExtent;
                              }),
              "every runtime extent scalar must be state-sourced");
    }
    return true;
}

bool TestCompileGeneratedStatefulPlan() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    std::cerr << "kv_state plan test requires LLVM and the dynamic module ABI\n";
    return true;
#else
    internal::ClearPrimitiveCacheForTesting();
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    CHECK(CheckGeneratedPlan(compiled, 1, 2),
          "single-cache compilation must produce the v1 shape");
    if (std::getenv("KXC_KV_PROBE") != nullptr) {
        const DeviceStream stream = DeviceStream::Default(Device::CPU());
        NDArray state = NDArray::Zeros({kBatch, kCapacity, kHeads, kHeadDim},
                                       F32(), Device::CPU(), 4);
        NDArray tokens = TokenTensor(3);
        NDArray count = CountTensor(3);
        std::vector<ModuleExtent> extents{0};
        Array<NDArray> outputs{state};
        internal::InvokeCompiledModuleWithOutputs(
            compiled.module, String("kv_state.append"), {state, tokens, count},
            outputs, stream, 0, &extents);
        std::vector<float> probe = ReadF32(state);
        std::cerr << "PROBE append state=[";
        for (int i = 0; i < 8; ++i) std::cerr << probe[i] << ",";
        std::cerr << "]\n";
        extents = {3};
        internal::InvokeCompiledModuleWithOutputs(
            compiled.module, String("kv_state.read_valid"), {state, count},
            outputs, stream, 0, &extents);
        NDArray read_out = F32Tensor({kBatch, kCapacity, kHeads, kHeadDim});
        Array<NDArray> read_outputs{read_out};
        internal::InvokeCompiledModuleWithOutputs(
            compiled.module, String("kv_state.read_valid"), {state, count},
            read_outputs, stream, 0, &extents);
        probe = ReadF32(read_out);
        std::cerr << "PROBE after read read_out=[";
        for (int i = 0; i < 8; ++i) std::cerr << probe[i] << ",";
        std::cerr << "]\n";
    }
    const runtime::ValueSpec state =
        [&]() {
            for (const runtime::ValueSpec& value : compiled.plan.values()) {
                if (value->is_state) return value;
            }
            return runtime::ValueSpec(ObjectRef());
        }();
    CHECK(state.defined(), "compiled plan must contain a state value");
    // The count input is a uint64[1] graph input; tokens input fits capacity.
    bool saw_count = false;
    for (const runtime::ValueSpec& value : compiled.plan.values()) {
        if (value->is_input && value->dtype.code == kDLUInt) {
            saw_count = true;
            CHECK(value.shape().size() == 1 && value.shape()[0] == 1,
                  "count input must be a uint64[1] tensor");
        }
    }
    CHECK(saw_count, "compiled plan must expose a uint64 count input");

    const PlanAbiFingerprint identity =
        BuildPlanAbiFingerprint(compiled.module, compiled.plan, compiled.artifacts);
    CHECK(identity.defined(), "stateful plan identity must be defined");
    CHECK(identity.canonical_bytes().find(
              "executable-plan-abi-v9-dynamic-stateful-v1") != std::string::npos,
          "plan identity must version the dynamic stateful contract");
    CHECK(identity.canonical_bytes().find("stateful_contract") !=
              std::string::npos &&
              identity.canonical_bytes().find("state_capacity") !=
                  std::string::npos &&
              identity.canonical_bytes().find("state_count_input_ordinal") !=
                  std::string::npos &&
              identity.canonical_bytes().find("state_extent_binding_ordinal") !=
                  std::string::npos,
          "plan identity must carry capacity, count input, and extent bindings");
    return true;
#endif
}

bool TestS1AppendReadProgression() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    const runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    CHECK(session.StateExtent(state_id) == 0,
          "initial valid length must be zero");
    const int64_t token_elements = kHeads * kHeadDim;

    std::vector<float> reference_cache(
        static_cast<size_t>(kCapacity * token_elements), 0.0F);
    int64_t length = 0;
    for (const int64_t count : {int64_t{3}, int64_t{1}, int64_t{1}}) {
        const NDArray tokens = TokenTensor(count);
        const std::vector<float> token_values = ReadF32(tokens);
        const Array<NDArray> outputs =
            session.Run({tokens, CountTensor(static_cast<uint64_t>(count))});
        length += count;
        reference_cache = ReferenceAppend(reference_cache, token_values,
                                          length - count, count);
        CHECK(outputs.size() == 1, "single-cache plan produces one output");
        const std::vector<float> read = ReadF32(outputs[0]);
        const std::vector<float> expected_read =
            ReferenceValidRead(reference_cache, length);
        CHECK(SameRegion(read, expected_read, 0, length * token_elements,
                         1e-5F,
                         "valid region must match the independent reference"),
              "valid region must match the independent reference");
        bool invalid_is_zero = true;
        for (int64_t index = length * token_elements;
             index < kCapacity * token_elements; ++index) {
            invalid_is_zero =
                invalid_is_zero && read[static_cast<size_t>(index)] == 0.0F;
        }
        CHECK(invalid_is_zero,
              "read output beyond the valid extent must be exactly zero");
        CHECK(session.StateExtent(state_id) == length,
              "committed extent must follow 3 -> 4 -> 5");
    }
    CHECK(length == 5, "progression must reach extent 5");
    return true;
#endif
}

bool TestS1ZeroLengthAppend() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    const NDArray tokens = TokenTensor(1);
    (void)session.Run({tokens, CountTensor(1)});
    CHECK(session.StateExtent(state_id) == 1, "first append must commit");
    const Storage before = session.StateValue(state_id).storage();
    const Array<NDArray> outputs = session.Run({tokens, CountTensor(0)});
    CHECK(session.StateExtent(state_id) == 1,
          "zero-length append must not move the extent");
    CHECK(outputs[0].storage().get() != before.get(),
          "read output must not alias the state storage");
    const std::vector<float> read = ReadF32(outputs[0]);
    const std::vector<float> state = ReadF32(session.StateValue(state_id));
    for (int64_t index = 0; index < kHeads * kHeadDim; ++index) {
        CHECK(CloseEnough(read[static_cast<size_t>(index)],
                          state[static_cast<size_t>(index)], 1e-6F),
              "zero-length append must keep the valid region readable");
    }
    return true;
#endif
}

bool TestS1CapacityBoundaryAndRejections() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    const NDArray tokens = TokenTensor(3);
    // Fill to exactly the capacity.
    (void)session.Run({tokens, CountTensor(3)});
    (void)session.Run({tokens, CountTensor(3)});
    (void)session.Run({tokens, CountTensor(2)});
    CHECK(session.StateExtent(state_id) == kCapacity,
          "exact-capacity append must be accepted");
    const std::vector<float> before = ReadF32(session.StateValue(state_id));

    // Over-capacity, uint64 overflow, and negative-as-huge counts must all be
    // rejected before launch, leaving length and cache unchanged.
    CHECK(Throws([&] { (void)session.Run({tokens, CountTensor(1)}); }),
          "over-capacity append must fail before launch");
    CHECK(Throws([&] {
              (void)session.Run({tokens, CountTensor(
                                            std::numeric_limits<uint64_t>::max())});
          }),
          "overflowing append count must fail before launch");
    const uint64_t negative_as_uint64 = ~static_cast<uint64_t>(0);
    CHECK(Throws([&] {
              (void)session.Run({tokens, CountTensor(negative_as_uint64)});
          }),
          "negative append count must fail before launch");
    CHECK(session.StateExtent(state_id) == kCapacity,
          "rejected runs must not change the committed extent");
    CHECK(ReadF32(session.StateValue(state_id)) == before,
          "rejected runs must not change the cache content");
    // Pre-launch rejections leave the session usable.
    const Array<NDArray> outputs = session.Run({tokens, CountTensor(0)});
    CHECK(outputs.size() == 1 &&
              session.StateExtent(state_id) == kCapacity,
          "session must stay usable after pre-launch rejections");
    // The tokens input may never describe more than its declared extent.
    CHECK(Throws([&] { (void)session.Run({TokenTensor(1), CountTensor(2)}); }),
          "append count beyond the tokens extent must fail before launch");
    return true;
#endif
}

bool TestS1InPlaceAddressAndSentinels() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    const Storage first = session.StateValue(state_id).storage();
    const void* first_address = first.data();
    {
        const std::vector<float> state = ReadF32(session.StateValue(state_id));
        bool all_sentinel = true;
        for (const float value : state) {
            all_sentinel = all_sentinel && value == static_cast<float>(kSentinelFill);
        }
        CHECK(all_sentinel, "invalid capacity region must hold the sentinel fill");
    }
    const NDArray tokens = TokenTensor(3);
    (void)session.Run({tokens, CountTensor(3)});
    const Storage second = session.StateValue(state_id).storage();
    CHECK(second.get() == first.get() && second.data() == first_address,
          "the cache buffer address must not move across runs");
    {
        const std::vector<float> state = ReadF32(session.StateValue(state_id));
        const int64_t token_elements = kHeads * kHeadDim;
        for (int64_t slot = kMaxAppend; slot < kCapacity; ++slot) {
            for (int64_t element = 0; element < token_elements; ++element) {
                CHECK(state[static_cast<size_t>(slot * token_elements + element)] ==
                          static_cast<float>(kSentinelFill),
                      "appends must never touch the invalid region sentinel");
            }
        }
    }
    // The in-place append rewrote exactly the window; the read output's
    // invalid region is zero, never the sentinel.
    const Array<NDArray> outputs = session.Run({tokens, CountTensor(0)});
    const std::vector<float> read = ReadF32(outputs[0]);
    for (int64_t index = kMaxAppend * kHeads * kHeadDim;
         index < kCapacity * kHeads * kHeadDim; ++index) {
        CHECK(read[static_cast<size_t>(index)] == 0.0F,
              "masked read must not leak the invalid-region sentinel");
    }
    return true;
#endif
}

bool TestS1IndependentSessionIsolation() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    runtime::RuntimeSession first(compiled.module, compiled.plan);
    runtime::RuntimeSession second(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    (void)first.Run({TokenTensor(1), CountTensor(1)});
    CHECK(first.StateExtent(state_id) == 1, "first session must append");
    CHECK(second.StateExtent(state_id) == 0,
          "second session must keep its own zero extent");
    CHECK(first.StateValue(state_id).storage().get() !=
              second.StateValue(state_id).storage().get(),
          "sessions must own independent cache storages");
    const std::vector<float> second_state = ReadF32(second.StateValue(state_id));
    for (const float value : second_state) {
        CHECK(value == static_cast<float>(kSentinelFill),
              "second session cache must still be untouched sentinel");
    }
    return true;
#endif
}

bool TestS1ConcurrentAppendsSerialize() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled = CompileKvStatePlan(CacheDeclaration(), CpuConfig());
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t state_id = compiled.plan.state_value_ids()[0];
    constexpr int kThreads = 4;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&, index] {
            try {
                runtime::RunAsyncResult result = session.RunAsync(
                    {TokenTensor(1), CountTensor(1)},
                    DeviceStream::Default(Device::CPU()));
                result.completion.Wait();
            } catch (const std::exception& error) {
                std::cerr << "thread " << index << ": " << error.what() << '\n';
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK(failures.load(std::memory_order_relaxed) == 0,
          "concurrent stateful submissions must all succeed serially");
    CHECK(session.StateExtent(state_id) == kThreads,
          "serialized appends must accumulate to one token per run");
    const std::vector<float> state = ReadF32(session.StateValue(state_id));
    const int64_t token_elements = kHeads * kHeadDim;
    for (int64_t slot = kThreads; slot < kCapacity; ++slot) {
        for (int64_t element = 0; element < token_elements; ++element) {
            CHECK(state[static_cast<size_t>(slot * token_elements + element)] ==
                      static_cast<float>(kSentinelFill),
                  "concurrent appends must not touch the invalid region");
        }
    }
    return true;
#endif
}

bool TestStatefulCompletionKeepsStateAlive() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    NDArray escaped_state;
    runtime::RunAsyncResult escaped;
    int64_t committed = -1;
    {
        const KvStatePlan compiled =
            CompileKvStatePlan(CacheDeclaration(), CpuConfig());
        runtime::RuntimeSession session(compiled.module, compiled.plan);
        const int64_t state_id = compiled.plan.state_value_ids()[0];
        escaped = session.RunAsync({TokenTensor(2), CountTensor(2)},
                                   DeviceStream::Default(Device::CPU()));
        escaped.completion.Wait();
        committed = session.StateExtent(state_id);
        escaped_state = session.StateValue(state_id);
    }
    CHECK(committed == 2, "commit must land before the session is destroyed");
    bool retained = false;
    for (const auto& storage : escaped.completion->retained_storage) {
        retained = retained || storage.get() == escaped_state.storage().get();
    }
    CHECK(escaped.completion.IsReady() && retained,
          "the completion handle must keep the session-owned cache alive");
    const std::vector<float> state = ReadF32(escaped_state);
    const int64_t token_elements = kHeads * kHeadDim;
    for (int64_t slot = 0; slot < 2; ++slot) {
        for (int64_t element = 0; element < token_elements; ++element) {
            CHECK(CloseEnough(
                      state[static_cast<size_t>(slot * token_elements + element)],
                      100.0F + static_cast<float>(slot) +
                          0.25F * static_cast<float>(element / kHeadDim) +
                          0.0625F * static_cast<float>(element % kHeadDim),
                      1e-5F),
                  "the retained cache must still expose the appended tokens");
        }
    }
    return true;
#endif
}

bool TestS2CausalAttentionDecodeLoop() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    const KvStatePlan compiled =
        CompileKvStatePlan(AttentionDeclaration(), CpuConfig());
    CHECK(CheckGeneratedPlan(compiled, 2, 3),
          "K/V compilation must produce append_k, append_v, and attention");
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    const int64_t k_state = compiled.plan.state_value_ids()[0];
    const int64_t v_state = compiled.plan.state_value_ids()[1];
    CHECK(session.StateExtent(k_state) == 0 && session.StateExtent(v_state) == 0,
          "both states must start at extent zero");

    const int64_t token_elements = kHeads * kHeadDim;
    std::vector<float> k_cache(static_cast<size_t>(kCapacity * token_elements),
                               0.0F);
    std::vector<float> v_cache = k_cache;
    int64_t length = 0;

    const auto step = [&](int64_t count) {
        const NDArray tokens_k = TokenTensor(count);
        const NDArray tokens_v = TokenTensor(count);
        const Array<NDArray> outputs = session.Run(
            {tokens_k, tokens_v, QueryTensor(count),
             CountTensor(static_cast<uint64_t>(count))});
        const std::vector<float> token_values = ReadF32(tokens_k);
        k_cache = ReferenceAppend(k_cache, token_values, length, count);
        v_cache = ReferenceAppend(v_cache, ReadF32(tokens_v), length, count);
        length += count;
        if (outputs.size() != 1) {
            throw std::runtime_error("decode step produces one attention result");
        }
        const std::vector<float> actual = ReadF32(outputs[0]);
        const std::vector<float> expected =
            ReferenceAttention(k_cache, ReadF32(QueryTensor(count)), length, count);
        for (int64_t index = 0; index < count * token_elements; ++index) {
            if (!CloseEnough(actual[static_cast<size_t>(index)],
                             expected[static_cast<size_t>(index)], 1e-3F)) {
                std::cerr << "[FAIL] decode step reference differs at " << index
                          << ": actual " << actual[static_cast<size_t>(index)]
                          << " expected " << expected[static_cast<size_t>(index)]
                          << '\n';
                throw std::runtime_error(
                    "decode step must match the full-recompute reference");
            }
        }
        for (int64_t index = count * token_elements;
             index < kMaxAppend * token_elements; ++index) {
            if (actual[static_cast<size_t>(index)] != 0.0F) {
                throw std::runtime_error("invalid query rows must read exactly zero");
            }
        }
        if (session.StateExtent(k_state) != length ||
            session.StateExtent(v_state) != length) {
            throw std::runtime_error("K and V extents must advance together");
        }
    };
    // Prefill 3 tokens, then at least three decode steps of one token each.
    step(3);
    step(1);
    step(1);
    step(1);
    CHECK(length == 6, "prefill plus three decodes must reach extent 6");
    // Sentinels in the invalid capacity region must not influence results.
    {
        const std::vector<float> k_state_values = ReadF32(session.StateValue(k_state));
        for (int64_t slot = length; slot < kCapacity; ++slot) {
            for (int64_t element = 0; element < token_elements; ++element) {
                CHECK(k_state_values[static_cast<size_t>(
                          slot * token_elements + element)] ==
                          static_cast<float>(kSentinelFill),
                      "invalid K region must still hold the sentinel");
            }
        }
    }
    // One more decode over a cache whose invalid region is pure sentinel.
    step(1);
    return true;
#endif
}

bool TestCursorValuesDoNotRecompile() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    internal::ClearPrimitiveCacheForTesting();
    const KvStatePlan compiled =
        CompileKvStatePlan(AttentionDeclaration(), CpuConfig());
    const PlanAbiFingerprint before =
        BuildPlanAbiFingerprint(compiled.module, compiled.plan, compiled.artifacts);
    const internal::PrimitiveCacheStats after_compile =
        internal::GetPrimitiveCacheStats();
    runtime::RuntimeSession session(compiled.module, compiled.plan);
    for (const uint64_t count : {uint64_t{3}, uint64_t{1}, uint64_t{0},
                                 uint64_t{1}}) {
        (void)session.Run({TokenTensor(static_cast<int64_t>(count)),
                           TokenTensor(static_cast<int64_t>(count)),
                           QueryTensor(static_cast<int64_t>(count)),
                           CountTensor(count)});
    }
    const PlanAbiFingerprint after =
        BuildPlanAbiFingerprint(compiled.module, compiled.plan, compiled.artifacts);
    CHECK(before.canonical_bytes() == after.canonical_bytes(),
          "cursor values must not change the plan identity");
    CHECK(SamePrimitiveCacheStats(after_compile,
                                  internal::GetPrimitiveCacheStats()),
          "running different cursor values must not compile");
    // Structural changes change identity: capacity and sentinel fill.
    KvStatePlanDeclaration wider = AttentionDeclaration();
    wider.layout.capacity = kCapacity + 4;
    const KvStatePlan wider_plan = CompileKvStatePlan(wider, CpuConfig());
    const PlanAbiFingerprint wider_identity = BuildPlanAbiFingerprint(
        wider_plan.module, wider_plan.plan, wider_plan.artifacts);
    CHECK(wider_identity.digest() != before.digest(),
          "capacity is part of the plan identity");
    CHECK(wider_identity.canonical_bytes().find("state_capacity") !=
              std::string::npos,
          "wider capacity must appear in the identity");
    KvStatePlanDeclaration other_fill = AttentionDeclaration();
    other_fill.invalid_fill = -9.0;
    const KvStatePlan other_fill_plan =
        CompileKvStatePlan(other_fill, CpuConfig());
    const PlanAbiFingerprint other_fill_identity = BuildPlanAbiFingerprint(
        other_fill_plan.module, other_fill_plan.plan,
        other_fill_plan.artifacts);
    CHECK(other_fill_identity.digest() != before.digest(),
          "the sentinel fill is part of the plan identity");
    // Same declaration compiles to the same identity.
    const KvStatePlan again = CompileKvStatePlan(AttentionDeclaration(), CpuConfig());
    CHECK(BuildPlanAbiFingerprint(again.module, again.plan, again.artifacts)
              .canonical_bytes() == before.canonical_bytes(),
          "compilation must be deterministic for the same declaration");
    return true;
#endif
}

/*! \brief Fails a launch when its injected call index is reached. */
class FailOnCallLauncher final : public codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }

    AsyncOperation Launch(const Array<NDArray>& arguments,
                          const DeviceStream& stream,
                          const ObjectRef&) const override {
        const int current = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current == fail_on_call) {
            throw std::runtime_error("kv state test injected mid-run failure");
        }
        Array<Storage> retained;
        for (const NDArray& argument : arguments) {
            retained.push_back(argument.storage());
        }
        return AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable std::atomic<int> calls{0};
    int fail_on_call{0};
};

ModuleInvocationContract StatefulContract(
    const KernelSignature& signature,
    ModuleRuntimeExtentScalar::Source source) {
    std::vector<ModuleInputContract> inputs;
    std::vector<ModuleTensorContract> outputs;
    for (const KernelArgSpec& argument : signature.arguments()) {
        if (argument->role == KernelArgRole::kInput) {
            ModuleInputContract input;
            const Array<int64_t> shape = argument.shape();
            for (size_t axis = 0; axis < shape.size(); ++axis) {
                input.axis_guards.push_back(ModuleAxisGuard{
                    axis, static_cast<ModuleExtent>(shape[axis]),
                    static_cast<ModuleExtent>(shape[axis]), 1,
                    static_cast<ModuleExtent>(shape[axis]), std::nullopt});
            }
            inputs.push_back(std::move(input));
        } else if (argument->role == KernelArgRole::kOutput) {
            ModuleTensorContract output;
            const Array<int64_t> shape = argument.shape();
            size_t elements = 1;
            for (int64_t extent : shape) {
                const ModuleShapeExpr expr =
                    ModuleShapeExpr::Const(static_cast<ModuleExtent>(extent));
                output.logical.push_back(expr);
                output.physical.push_back(expr);
                output.valid.push_back(expr);
                elements *= static_cast<size_t>(extent);
            }
            output.max_bytes =
                elements * static_cast<size_t>(argument->dtype.bits / 8);
            outputs.push_back(std::move(output));
        }
    }
    ModuleRuntimeExtentScalar scalar =
        source == ModuleRuntimeExtentScalar::Source::kStateExtent
            ? ModuleRuntimeExtentScalar::StateExtent()
            : ModuleRuntimeExtentScalar{ModuleShapeExpr::Const(1)};
    scalar.source = source;
    ModuleInvocationContract contract(std::move(inputs), std::move(outputs),
                                      {std::move(scalar)}, 0);
    contract.Validate(signature);
    return contract;
}

struct SyntheticStatefulFixture final {
    CompiledModule module;
    runtime::ExecutablePlan plan;
    std::shared_ptr<FailOnCallLauncher> launcher;
};

/*! \brief Rank-1 tokens tensor for the synthetic [4]-state fixture. */
NDArray Vec4Tensor() {
    NDArray value = NDArray::Empty({4}, F32(), Device::CPU(), 4);
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    value.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return value;
}

/*! \brief Two-call synthetic stateful graph (append + read) over a [4]
 *  state; kernels are recording stubs. */
SyntheticStatefulFixture MakeSyntheticStatefulFixture() {
    const Device cpu = Device::CPU();
    const auto append_signature = KernelSignature(
        "synthetic_append",
        {KernelArgSpec("state_in", KernelArgRole::kInput, F32(), {4}, cpu, 4),
         KernelArgSpec("tokens", KernelArgRole::kInput, F32(), {4}, cpu, 4),
         KernelArgSpec("count", KernelArgRole::kInput, U64(), {1}, cpu, 8),
         KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, U64(), {1}, cpu,
                       8),
         KernelArgSpec("state_out", KernelArgRole::kOutput, F32(), {4}, cpu, 4,
                       true)});
    const auto read_signature = KernelSignature(
        "synthetic_read",
        {KernelArgSpec("state_in", KernelArgRole::kInput, F32(), {4}, cpu, 4),
         KernelArgSpec("count", KernelArgRole::kInput, U64(), {1}, cpu, 8),
         KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, U64(), {1}, cpu,
                       8),
         KernelArgSpec("read_out", KernelArgRole::kOutput, F32(), {4}, cpu, 4,
                       true)});
    auto launcher = std::make_shared<FailOnCallLauncher>();
    const KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
    const CompiledModule module = internal::BuildCompiledModule(
        BuildTarget(cpu),
        {{append_signature, metadata,
          CompiledKernel(append_signature, metadata, launcher),
          std::make_shared<ModuleInvocationContract>(
              StatefulContract(append_signature,
                               ModuleRuntimeExtentScalar::Source::kStateExtent))},
         {read_signature, metadata,
          CompiledKernel(read_signature, metadata, launcher),
          std::make_shared<ModuleInvocationContract>(
              StatefulContract(read_signature,
                               ModuleRuntimeExtentScalar::Source::kStateExtent))}},
        {});
    const runtime::ExecutablePlan plan(
        {runtime::ValueSpec(0, 0, {4}, F32(), cpu, false, false, false, false,
                            false, true, -1,
                            runtime::ValueWriteMode::kAllocate, -1, 4, 0, 0.0),
         runtime::ValueSpec(1, 1, {4}, F32(), cpu, true),
         runtime::ValueSpec(2, 2, {1}, U64(), cpu, true),
         runtime::ValueSpec(3, 0, {4}, F32(), cpu, false, false, false, true,
                            false, false, 0,
                            runtime::ValueWriteMode::kInPlace),
         runtime::ValueSpec(4, 4, {4}, F32(), cpu, false, false, true)},
        {runtime::KernelCall("synthetic_append", {0, 1, 2}, {3}),
         runtime::KernelCall("synthetic_read", {3, 2}, {4})},
        {1, 2}, {}, {4}, {0},
        runtime::ExecutablePlanMode::kDynamicStatefulV1, {}, {{0}, {0}}, 2);
    plan.Validate();
    return SyntheticStatefulFixture{module, plan, launcher};
}

bool TestStatefulContractMismatchLauncherZero() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    internal::ClearPrimitiveCacheForTesting();
    // Well-formed fixture: construction and execution succeed.
    {
        SyntheticStatefulFixture fixture = MakeSyntheticStatefulFixture();
        runtime::RuntimeSession session(fixture.module, fixture.plan);
        (void)session.Run({Vec4Tensor(), CountTensor(1)});
        CHECK(fixture.launcher->calls.load(std::memory_order_relaxed) == 2,
              "the synthetic stateful graph must launch both kernels");
    }
    internal::ClearPrimitiveCacheForTesting();

    // (a) Input-sourced runtime extents are rejected in the stateful mode
    // with zero launches.
    {
        SyntheticStatefulFixture base = MakeSyntheticStatefulFixture();
        const Device cpu = Device::CPU();
        const auto append_signature = KernelSignature(
            "synthetic_append",
            {KernelArgSpec("state_in", KernelArgRole::kInput, F32(), {4}, cpu, 4),
             KernelArgSpec("tokens", KernelArgRole::kInput, F32(), {4}, cpu, 4),
             KernelArgSpec("count", KernelArgRole::kInput, U64(), {1}, cpu, 8),
             KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, U64(), {1},
                           cpu, 8),
             KernelArgSpec("state_out", KernelArgRole::kOutput, F32(), {4}, cpu,
                           4, true)});
        const auto read_signature = KernelSignature(
            "synthetic_read",
            {KernelArgSpec("state_in", KernelArgRole::kInput, F32(), {4}, cpu, 4),
             KernelArgSpec("count", KernelArgRole::kInput, U64(), {1}, cpu, 8),
             KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, U64(), {1},
                           cpu, 8),
             KernelArgSpec("read_out", KernelArgRole::kOutput, F32(), {4}, cpu,
                           4, true)});
        auto launcher = std::make_shared<FailOnCallLauncher>();
        const KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
        CompiledModule mismatched = internal::BuildCompiledModule(
            BuildTarget(cpu),
            {{append_signature, metadata,
              CompiledKernel(append_signature, metadata, launcher),
              std::make_shared<ModuleInvocationContract>(
                  StatefulContract(append_signature,
                                   ModuleRuntimeExtentScalar::Source::kInputAxis))},
             {read_signature, metadata,
              CompiledKernel(read_signature, metadata, launcher),
              std::make_shared<ModuleInvocationContract>(
                  StatefulContract(read_signature,
                                   ModuleRuntimeExtentScalar::Source::kInputAxis))}},
            {});
        CHECK(Throws([&] {
                  (void)runtime::RuntimeSession(mismatched, base.plan);
              }) && launcher->calls.load(std::memory_order_relaxed) == 0,
              "input-sourced extents must be rejected in stateful mode with "
              "zero launches");
    }
    internal::ClearPrimitiveCacheForTesting();

    // (b) Extent binding count must match the kernel ABI with zero launches.
    {
        SyntheticStatefulFixture base = MakeSyntheticStatefulFixture();
        const runtime::ExecutablePlan missing_bindings(
            base.plan.values(), base.plan.calls(), base.plan.input_value_ids(),
            base.plan.constant_value_ids(), base.plan.output_value_ids(),
            base.plan.state_value_ids(),
            runtime::ExecutablePlanMode::kDynamicStatefulV1, {}, {{}, {}}, 2);
        missing_bindings.Validate();
        CHECK(Throws([&] {
                  (void)runtime::RuntimeSession(base.module, missing_bindings);
              }) && base.launcher->calls.load(std::memory_order_relaxed) == 0,
              "extent binding count mismatch must be rejected before launch");
    }
    internal::ClearPrimitiveCacheForTesting();

    // (c) A missing append-count input is invalid at plan level.
    {
        CHECK(Throws([&] {
                  const SyntheticStatefulFixture base =
                      MakeSyntheticStatefulFixture();
                  const runtime::ExecutablePlan no_count(
                      base.plan.values(), base.plan.calls(),
                      base.plan.input_value_ids(),
                      base.plan.constant_value_ids(),
                      base.plan.output_value_ids(),
                      base.plan.state_value_ids(),
                      runtime::ExecutablePlanMode::kDynamicStatefulV1, {},
                      {{0}, {0}}, -1);
              }),
              "the stateful plan requires an append-count input");
    }
    internal::ClearPrimitiveCacheForTesting();

    // (d) Mid-run failure poisons the session without committing lengths.
    {
        SyntheticStatefulFixture base = MakeSyntheticStatefulFixture();
        base.launcher->fail_on_call = 2;
        runtime::RuntimeSession session(base.module, base.plan);
        CHECK(Throws([&] {
                  (void)session.Run({Vec4Tensor(), CountTensor(1)});
              }),
              "injected mid-run failure must surface");
        CHECK(base.launcher->calls.load(std::memory_order_relaxed) == 2,
              "both submits were attempted; the second launcher call injected "
              "the failure");
        CHECK(session.StateExtent(0) == 0,
              "a failed run must not commit the length");
        CHECK(Throws([&] {
                  (void)session.Run({Vec4Tensor(), CountTensor(0)});
              }),
              "the session must refuse further runs after a mid-run failure");
    }
    return true;
#endif
}

bool TestFreshOutputRejectionsRemainIntact() {
#if !KXC_USE_LLVM || !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    return true;
#else
    // The new stateful mode never relaxes the fresh-output rejection contract.
    const Device cpu = Device::CPU();
    const Array<runtime::ValueSpec> state_values(
        {runtime::ValueSpec(0, 0, {4}, F32(), cpu, false, false, false, false,
                            false, true)});
    CHECK(Throws([&] {
              (void)runtime::ExecutablePlan(
                  state_values,
                  {runtime::KernelCall("reject_state", {0}, {1})},
                  {}, {}, {1}, {},
                  runtime::ExecutablePlanMode::kDynamicFreshOutputV1);
          }),
          "fresh-output mode must still reject state");
    CHECK(Throws([&] {
              (void)runtime::ExecutablePlan(
                  {runtime::ValueSpec(0, 0, {4}, F32(), cpu, true),
                   runtime::ValueSpec(1, 0, {4}, F32(), cpu, false, false,
                                      false, true, false, false, 0,
                                      runtime::ValueWriteMode::kInPlace)},
                  {runtime::KernelCall("reject_alias", {0}, {1})},
                  {0}, {}, {1}, {},
                  runtime::ExecutablePlanMode::kDynamicFreshOutputV1);
          }),
          "fresh-output mode must still reject in-place aliasing");
    return true;
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"compile_generated_stateful_plan", TestCompileGeneratedStatefulPlan},
        {"s1_append_read_progression", TestS1AppendReadProgression},
        {"s1_zero_length_append", TestS1ZeroLengthAppend},
        {"s1_capacity_boundary_and_rejections",
         TestS1CapacityBoundaryAndRejections},
        {"s1_in_place_address_and_sentinels", TestS1InPlaceAddressAndSentinels},
        {"s1_independent_session_isolation", TestS1IndependentSessionIsolation},
        {"s1_concurrent_appends_serialize", TestS1ConcurrentAppendsSerialize},
        {"stateful_completion_keeps_state_alive",
         TestStatefulCompletionKeepsStateAlive},
        {"s2_causal_attention_decode_loop", TestS2CausalAttentionDecodeLoop},
        {"cursor_values_do_not_recompile", TestCursorValuesDoNotRecompile},
        {"stateful_contract_mismatch_launcher_zero",
         TestStatefulContractMismatchLauncherZero},
        {"fresh_output_rejections_remain_intact",
         TestFreshOutputRejectionsRemainIntact},
    };
    const char* only = std::getenv("KXC_KV_ONLY");
    for (const auto& [name, test] : tests) {
        if (only && std::string(name) != only) continue;
        g_failures = 0;
        try {
            if (!test()) {
                std::cerr << "[FAIL] " << name << '\n';
                return 1;
            }
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
        std::cout << "[PASS] " << name << '\n';
    }
    return g_failures == 0 ? 0 : 1;
}
