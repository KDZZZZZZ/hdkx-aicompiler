/*! \file src/compiler/kv_state/kv_state_plan.cc
 * \brief Production TE -> TIR -> LLVM compilation of the M2 dynamic stateful
 *        KV declaration into kernels, invocation contracts, and the plan.
 */

#include "kxc/compiler/kv_state_plan.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../internal/execution_contract.h"
#include "../internal/kernel_abi_builder.h"
#include "../internal/primitive_cache.h"
#include "../internal/te_to_tir.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/tir/printer/print_ir.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include "kxc/runtime/session.h"
#include "runtime/internal/compiled_module_node.h"
#include "runtime/internal/module_invocation_contract.h"
#include "support/hash.h"


#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif
#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/llvm/internal/codegen_llvm.h"
#include "codegen/llvm/internal/llvm_jit.h"
#endif

namespace kxc::api {
namespace {

void AppendField(std::string* out, const std::string& name,
                 const std::string& value) {
    *out += name;
    *out += '=';
    *out += value;
    *out += ';';
}

void AppendField(std::string* out, const std::string& name, int64_t value) {
    AppendField(out, name, std::to_string(value));
}

const char* BackendVersionFor(const Target& target) {
    if (target->kind == "llvm" && target->device_type == kCPU) {
        return "llvm-orc-v1";
    }
    throw std::invalid_argument(
        "Stateful KV compilation requires an LLVM/CPU target");
}

std::string DeclarationCanonical(const KvStatePlanDeclaration& declaration) {
    std::string out;
    AppendField(&out, "kind", "kv-state-plan-declaration-v1");
    AppendField(&out, "dtype_code", static_cast<int64_t>(declaration.dtype.code));
    AppendField(&out, "dtype_bits", static_cast<int64_t>(declaration.dtype.bits));
    AppendField(&out, "dtype_lanes", static_cast<int64_t>(declaration.dtype.lanes));
    AppendField(&out, "device_type",
                static_cast<int64_t>(declaration.device.device_type()));
    AppendField(&out, "device_id",
                static_cast<int64_t>(declaration.device.device_id()));
    AppendField(&out, "batch", declaration.layout.batch);
    AppendField(&out, "kv_heads", declaration.layout.kv_heads);
    AppendField(&out, "head_dim", declaration.layout.head_dim);
    AppendField(&out, "capacity", declaration.layout.capacity);
    AppendField(&out, "max_append_tokens", declaration.max_append_tokens);
    AppendField(&out, "state_count", declaration.state_count);
    uint64_t fill_bits = 0;
    static_assert(sizeof(fill_bits) == sizeof(declaration.invalid_fill),
                  "state fill must be 64-bit for identity");
    std::memcpy(&fill_bits, &declaration.invalid_fill, sizeof(fill_bits));
    AppendField(&out, "invalid_fill_bits", static_cast<int64_t>(fill_bits));
    AppendField(&out, "read_kind", static_cast<int64_t>(declaration.read_kind));
    AppendField(&out, "plan_mode",
                static_cast<int64_t>(runtime::ExecutablePlanMode::kDynamicStatefulV1));
    AppendField(&out, "module_invocation_abi_version",
                static_cast<int64_t>(ModuleInvocationContract::kAbiVersion));
    AppendField(&out, "schedule_policy",
                relay::internal::kStatefulKvTESchedulePolicy);
    return out;
}

void ValidateDeclaration(const KvStatePlanDeclaration& declaration) {
    if (!declaration.device.defined() ||
        declaration.device.device_type() != kCPU) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration requires a defined CPU device in v1");
    }
    if (declaration.dtype.code != kDLFloat || declaration.dtype.bits != 32 ||
        declaration.dtype.lanes != 1) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration v1 fixes the KV dtype to float32");
    }
    if (declaration.layout.batch < 1 || declaration.layout.kv_heads < 1 ||
        declaration.layout.head_dim < 1 || declaration.layout.capacity < 1) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration layout extents must be positive");
    }
    if (declaration.max_append_tokens < 1 ||
        declaration.max_append_tokens > declaration.layout.capacity) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration max_append_tokens must fit the capacity");
    }
    if (declaration.state_count != 1 && declaration.state_count != 2) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration state_count must be 1 or 2 in v1");
    }
    if (declaration.read_kind == KvStateReadKind::kCausalAttention &&
        declaration.state_count != 2) {
        throw std::invalid_argument(
            "causal attention read requires the K/V state pair");
    }
    if (declaration.read_kind == KvStateReadKind::kValidRegion &&
        declaration.state_count != 1) {
        throw std::invalid_argument(
            "valid-region read is defined for the single-cache declaration");
    }
    if (!std::isfinite(declaration.invalid_fill)) {
        throw std::invalid_argument(
            "KvStatePlanDeclaration invalid_fill must be finite");
    }
}

tir::PrimExpr CastI64(const tir::PrimExpr& expression) {
    if (expression.dtype() == tir::DataType::Int(64)) return expression;
    return tir::Call(tir::DataType::Int(64), "cast", {expression});
}

/*! \brief -1e30 masks invalid capacity slots; exp(-1e30 - max) underflows to
 *  the exact zero weight, so sentinel values can never reach the result. */
tir::PrimExpr MaskedScore(const tir::PrimExpr& score,
                          const tir::PrimExpr& valid) {
    return tir::Select(
        valid, score, tir::FloatImm(-1e30, tir::DataType::Float(32)));
}

te::Tensor StatePlaceholder(const std::string& name,
                            const KvStatePlanDeclaration& declaration) {
    Array<tir::PrimExpr> shape;
    shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.capacity, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.head_dim, tir::DataType::Int(64)));
    return te::placeholder(shape, tir::DataType::Float(32), name);
}

te::Tensor TokenPlaceholder(const std::string& name,
                            const KvStatePlanDeclaration& declaration) {
    Array<tir::PrimExpr> shape;
    shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.max_append_tokens, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.head_dim, tir::DataType::Int(64)));
    return te::placeholder(shape, tir::DataType::Float(32), name);
}

te::Tensor CountPlaceholder() {
    Array<tir::PrimExpr> shape;
    shape.push_back(tir::IntImm(1, tir::DataType::Int(64)));
    return te::placeholder(shape, tir::DataType::UInt(64), "kv_count");
}

Array<tir::PrimExpr> StateShapeExprs(const KvStatePlanDeclaration& declaration) {
    Array<tir::PrimExpr> shape;
    shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.capacity, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));
    shape.push_back(tir::IntImm(declaration.layout.head_dim, tir::DataType::Int(64)));
    return shape;
}

/*! \brief Append kernel: in-place identity compute over the physical
 *  capacity; only the window [length, length+n) receives token values. The
 *  token index is clamped into the declared tokens extent, so speculative
 *  loads stay in bounds and the Select discards their values. */
struct KernelSpec final {
    std::string symbol;
    std::string operator_name;
    te::Tensor output;
    Array<te::Tensor> abi_inputs;
    std::vector<relay::internal::ConstantTensor> constants;
    tir::Var state_extent{"kv_state_extent", tir::DataType::UInt(64)};
    int64_t output_elements{0};
};

KernelSpec MakeAppendKernel(const KvStatePlanDeclaration& declaration,
                            const std::string& suffix, te::Tensor state_in,
                            te::Tensor tokens, te::Tensor count) {
    const int64_t J = declaration.max_append_tokens;
    KernelSpec spec;
    spec.symbol = "kv_state.append" + suffix;
    spec.operator_name = "kv_state.append";
    const tir::PrimExpr length =
        relay::internal::LoadRuntimeExtent(spec.state_extent);
    const tir::PrimExpr count_i64 =
        CastI64(count(tir::IntImm(0, tir::DataType::Int(32))));
    spec.output = te::compute(
        StateShapeExprs(declaration),
        [&](const Array<tir::Var>& axes) {
            const tir::PrimExpr i = CastI64(axes[1]);
            const tir::PrimExpr in_window =
                tir::Not(i < length);  // length <= i
            const tir::PrimExpr below_end = i < (length + count_i64);
            const tir::PrimExpr token_index =
                tir::Min(tir::Max(i - length,
                                  tir::IntImm(0, tir::DataType::Int(64))),
                         tir::IntImm(J - 1, tir::DataType::Int(64)));
            return tir::Select(in_window && below_end,
                               tokens(axes[0], token_index, axes[2], axes[3]),
                               state_in(axes[0], axes[1], axes[2], axes[3]));
        },
        "kv_state_next");
    spec.abi_inputs = {state_in, tokens, count};
    spec.output_elements = declaration.layout.batch * declaration.layout.capacity *
                           declaration.layout.kv_heads *
                           declaration.layout.head_dim;
    return spec;
}

KernelSpec MakeValidReadKernel(const KvStatePlanDeclaration& declaration,
                               te::Tensor state_in, te::Tensor count) {
    KernelSpec spec;
    spec.symbol = "kv_state.read_valid";
    spec.operator_name = "kv_state.read_valid";
    const tir::PrimExpr length =
        relay::internal::LoadRuntimeExtent(spec.state_extent);
    const tir::PrimExpr count_i64 =
        CastI64(count(tir::IntImm(0, tir::DataType::Int(32))));
    spec.output = te::compute(
        StateShapeExprs(declaration),
        [&](const Array<tir::Var>& axes) {
            const tir::PrimExpr i = CastI64(axes[1]);
            return tir::Select(
                i < (length + count_i64),
                state_in(axes[0], axes[1], axes[2], axes[3]),
                tir::FloatImm(0.0, tir::DataType::Float(32)));
        },
        "kv_read");
    spec.abi_inputs = {state_in, count};
    spec.output_elements = declaration.layout.batch * declaration.layout.capacity *
                           declaration.layout.kv_heads *
                           declaration.layout.head_dim;
    return spec;
}

/*! \brief Tiny fixed-weight causal attention over the valid region:
 *  softmax(scale * qK) V with -1e30 masking outside [0, length+n) and
 *  zeroed output rows for invalid query slots. */
KernelSpec MakeCausalAttentionKernel(
    const KvStatePlanDeclaration& declaration, te::Tensor k_in,
    te::Tensor v_in, te::Tensor q, te::Tensor scale, te::Tensor count) {
    const int64_t C = declaration.layout.capacity;
    const int64_t D = declaration.layout.head_dim;
    KernelSpec spec;
    spec.symbol = "kv_state.causal_attention";
    spec.operator_name = "kv_state.causal_attention";
    const tir::PrimExpr length =
        relay::internal::LoadRuntimeExtent(spec.state_extent);
    const tir::PrimExpr count_i64 =
        CastI64(count(tir::IntImm(0, tir::DataType::Int(32))));
    const tir::PrimExpr valid_end = length + count_i64;
    const tir::PrimExpr scale_value =
        scale(tir::IntImm(0, tir::DataType::Int(32)));

    Array<tir::PrimExpr> score_shape;
    score_shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    score_shape.push_back(tir::IntImm(declaration.max_append_tokens, tir::DataType::Int(64)));
    score_shape.push_back(tir::IntImm(C, tir::DataType::Int(64)));
    score_shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));
    Array<tir::PrimExpr> result_shape;
    result_shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    result_shape.push_back(tir::IntImm(declaration.max_append_tokens, tir::DataType::Int(64)));
    result_shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));
    result_shape.push_back(tir::IntImm(D, tir::DataType::Int(64)));
    Array<tir::PrimExpr> head_shape;
    head_shape.push_back(tir::IntImm(declaration.layout.batch, tir::DataType::Int(64)));
    head_shape.push_back(tir::IntImm(declaration.max_append_tokens, tir::DataType::Int(64)));
    head_shape.push_back(tir::IntImm(declaration.layout.kv_heads, tir::DataType::Int(64)));

    const te::IterVar head_reduce = te::reduce_axis(
        tir::IntImm(0, tir::DataType::Int(64)),
        tir::IntImm(D, tir::DataType::Int(64)), "rhead");
    const te::IterVar slot_reduce = te::reduce_axis(
        tir::IntImm(0, tir::DataType::Int(64)),
        tir::IntImm(C, tir::DataType::Int(64)), "rslot");

    te::Tensor dot = te::compute(
        score_shape,
        [&](const Array<tir::Var>& axes) {
            return te::sum(q(axes[0], axes[1], axes[3], head_reduce) *
                               k_in(axes[0], axes[2], axes[3], head_reduce),
                           {head_reduce});
        },
        "kv_dot");
    te::Tensor scores = te::compute(
        score_shape,
        [&](const Array<tir::Var>& axes) {
            return MaskedScore(scale_value * dot(axes[0], axes[1], axes[2], axes[3]),
                               CastI64(axes[2]) < valid_end);
        },
        "kv_scores");
    te::Tensor max_score = te::compute(
        head_shape,
        [&](const Array<tir::Var>& axes) {
            return te::max(scores(axes[0], axes[1], slot_reduce, axes[2]),
                           {slot_reduce});
        },
        "kv_max_score");
    te::Tensor weights = te::compute(
        score_shape,
        [&](const Array<tir::Var>& axes) {
            const tir::PrimExpr normalized = tir::Call(
                tir::DataType::Float(32), "exp",
                {scores(axes[0], axes[1], axes[2], axes[3]) -
                 max_score(axes[0], axes[1], axes[3])});
            return tir::Select(
                (CastI64(axes[1]) < count_i64) &&
                    (CastI64(axes[2]) < valid_end),
                normalized, tir::FloatImm(0.0, tir::DataType::Float(32)));
        },
        "kv_weights");
    te::Tensor denominator = te::compute(
        head_shape,
        [&](const Array<tir::Var>& axes) {
            return te::sum(weights(axes[0], axes[1], slot_reduce, axes[2]),
                           {slot_reduce});
        },
        "kv_denominator");
    te::Tensor attention = te::compute(
        result_shape,
        [&](const Array<tir::Var>& axes) {
            return te::sum(
                weights(axes[0], axes[1], slot_reduce, axes[2]) /
                    denominator(axes[0], axes[1], axes[2]) *
                    v_in(axes[0], slot_reduce, axes[2], axes[3]),
                {slot_reduce});
        },
        "kv_attention");
    spec.output = te::compute(
        result_shape,
        [&](const Array<tir::Var>& axes) {
            return tir::Select(
                CastI64(axes[1]) < count_i64,
                attention(axes[0], axes[1], axes[2], axes[3]),
                tir::FloatImm(0.0, tir::DataType::Float(32)));
        },
        "kv_attention_result");
    spec.abi_inputs = {k_in, v_in, q, count};
    spec.output_elements = declaration.layout.batch *
                           declaration.max_append_tokens *
                           declaration.layout.kv_heads * D;
    return spec;
}



runtime::NDArray ConstantValue(float value) {
    runtime::NDArray result =
        runtime::NDArray::Empty({1}, runtime::DataTypeFromString("float32"),
                                Device::CPU(), 64);
    result.CopyFromBytes(&value, sizeof(float));
    return result;
}

ModuleInvocationContract MakeContract(const codegen::KernelSignature& signature,
                                      int64_t output_elements) {
    std::vector<ModuleInputContract> inputs;
    std::vector<ModuleTensorContract> outputs;
    for (const auto& argument : signature.arguments()) {
        if (argument->role != codegen::KernelArgRole::kInput) continue;
        ModuleInputContract input;
        const Array<int64_t> shape = argument.shape();
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            input.axis_guards.push_back(ModuleAxisGuard{
                axis, static_cast<ModuleExtent>(shape[axis]),
                static_cast<ModuleExtent>(shape[axis]), 1,
                static_cast<ModuleExtent>(shape[axis]), std::nullopt});
        }
        inputs.push_back(std::move(input));
    }
    std::vector<codegen::KernelArgSpec> output_specs;
    for (const auto& argument : signature.arguments()) {
        if (argument->role == codegen::KernelArgRole::kOutput) {
            output_specs.push_back(argument);
        }
    }
    if (output_specs.size() != 1) {
        throw std::logic_error("stateful KV kernels produce one tensor each");
    }
    ModuleTensorContract output;
    const Array<int64_t> shape = output_specs[0].shape();
    size_t elements = 1;
    for (int64_t extent : shape) {
        const ModuleShapeExpr expr =
            ModuleShapeExpr::Const(static_cast<ModuleExtent>(extent));
        output.logical.push_back(expr);
        output.physical.push_back(expr);
        output.valid.push_back(expr);
        elements *= static_cast<size_t>(extent);
    }
    const size_t item_bytes =
        static_cast<size_t>(output_specs[0]->dtype.bits / 8) *
        output_specs[0]->dtype.lanes;
    output.max_bytes = elements * item_bytes;
    if (elements != static_cast<size_t>(output_elements)) {
        throw std::logic_error(
            "stateful KV contract output element count differs from its kernel");
    }
    outputs.push_back(std::move(output));
    ModuleInvocationContract contract(std::move(inputs), std::move(outputs),
                                      {ModuleRuntimeExtentScalar::StateExtent()},
                                      0);
    contract.Validate(signature);
    return contract;
}

}  // namespace

KvStatePlan CompileKvStatePlan(const KvStatePlanDeclaration& declaration,
                               CompileConfig config) {
    ValidateDeclaration(declaration);
    if (!config.defined()) {
        throw std::invalid_argument("CompileKvStatePlan requires a CompileConfig");
    }
    config.Validate();
    const Target target = config->target;
    if (!target.defined() || target->kind != "llvm" ||
        target->device_type != kCPU) {
        throw std::invalid_argument(
            "CompileKvStatePlan requires the LLVM/CPU target in v1");
    }
#if !KXC_USE_LLVM
    (void)target;
    throw std::runtime_error(
        "CompileKvStatePlan requires a build with KXC_ENABLE_LLVM=ON");
#else
    const std::string declaration_canonical = DeclarationCanonical(declaration);
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    const int64_t B = declaration.layout.batch;
    const int64_t H = declaration.layout.kv_heads;
    const int64_t D = declaration.layout.head_dim;
    const int64_t C = declaration.layout.capacity;
    const int64_t J = declaration.max_append_tokens;

    std::vector<KernelSpec> kernels;
    std::vector<std::vector<int64_t>> extent_bindings;
    const te::Tensor count = CountPlaceholder();
    if (declaration.state_count == 1) {
        KernelSpec append = MakeAppendKernel(
            declaration, "", StatePlaceholder("kv_state_in", declaration),
            TokenPlaceholder("kv_tokens", declaration), count);
        kernels.push_back(std::move(append));
        extent_bindings.push_back({0});
        // The read consumes the append's alias output, which shares the state
        // storage; its TE input denotes the same logical buffer.
        kernels.push_back(MakeValidReadKernel(
            declaration, StatePlaceholder("kv_state_in", declaration), count));
        extent_bindings.push_back({0});
    } else {
        const float attention_scale = 1.0F / std::sqrt(static_cast<float>(D));
        const te::Tensor scale = te::placeholder(
            {tir::IntImm(1, tir::DataType::Int(64))}, tir::DataType::Float(32),
            "kv_scale");
        KernelSpec append_k = MakeAppendKernel(
            declaration, "_k", StatePlaceholder("kv_k_in", declaration),
            TokenPlaceholder("kv_tokens_k", declaration), count);
        kernels.push_back(std::move(append_k));
        extent_bindings.push_back({0});
        KernelSpec append_v = MakeAppendKernel(
            declaration, "_v", StatePlaceholder("kv_v_in", declaration),
            TokenPlaceholder("kv_tokens_v", declaration), count);
        kernels.push_back(std::move(append_v));
        extent_bindings.push_back({1});
        KernelSpec attention = MakeCausalAttentionKernel(
            declaration, StatePlaceholder("kv_k_in", declaration),
            StatePlaceholder("kv_v_in", declaration),
            TokenPlaceholder("kv_queries", declaration), scale, count);
        attention.constants.push_back(relay::internal::ConstantTensor{
            scale, String("kv.attn.scale"), ConstantValue(attention_scale)});
        kernels.push_back(std::move(attention));
        extent_bindings.push_back({0});
    }

    // Production lowering: schedule, TE->TIR, TIR pipeline, ABI, backend.
    const Device device(target->device_type, target->device_id);
    std::vector<internal::CompiledModuleEntry> entries;
    Map<String, runtime::NDArray> constants;
    std::vector<tir::PrimFunc> optimized_functions;
    optimized_functions.reserve(kernels.size());
    for (const KernelSpec& kernel : kernels) {
        const te::Schedule schedule = relay::internal::BuildStatefulKvTESchedule(
            {kernel.output}, target);
        const std::string unit_canonical =
            declaration_canonical + "kernel_role=" + kernel.operator_name + ";";
        const relay::LoweredFunction lowered =
            relay::internal::LowerTensorGraphToTIR(
                kernel.abi_inputs, kernel.constants, {kernel.output}, schedule,
                target,
                relay::internal::PrimFuncIdentity{
                    String(kernel.symbol), int64_t{-1},
                    String(kernel.operator_name), int64_t{1},
                    String(support::HashText(unit_canonical))},
                {kernel.state_extent}, {0});
        for (const auto& binding : lowered.constants()) {
            if (constants.count(binding->key) &&
                constants.at(binding->key).get() != binding->value.get()) {
                throw std::invalid_argument(
                    "stateful KV constant key resolved to different payloads");
            }
            constants.Set(binding->key, binding->value);
        }
        optimized_functions.push_back(PipelineExecutor::ExecuteTIR(
            contract.tir_pipeline, lowered->prim_func, target));
        if (std::getenv("KXC_DUMP_KV_TIR") != nullptr) {
            std::ostringstream stream;
            tir::printer::DumpPrimFunc(optimized_functions.back(), stream);
            std::cerr << "==== TIR " << kernel.symbol << " ====\n"
                      << stream.str() << "\n==== end ====";
        }
    }

    for (size_t index = 0; index < kernels.size(); ++index) {
        const KernelSpec& kernel = kernels[index];
        const tir::PrimFunc& optimized = optimized_functions[index];
        const codegen::KernelSignature signature = codegen::BuildKernelSignature(
            optimized, constants, target, String(kernel.symbol));
        signature.Validate();
        const std::string schedule_contract =
            relay::internal::GetTEScheduleContract(optimized);
        const UnitSemanticKey semantic_key(
            declaration_canonical + "kernel_role=" + kernel.operator_name + ";");
        const PrimitiveArtifactKey artifact_key =
            internal::BuildPrimitiveArtifactKey(
                semantic_key, target, contract.canonical_bytes,
                schedule_contract.c_str(), BackendVersionFor(target));
        auto module_contract = std::make_shared<ModuleInvocationContract>(
            MakeContract(signature, kernel.output_elements));
        const codegen::KernelLaunchMetadata metadata(
            device, codegen::CodeGenBackend::kLLVM);
        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM emitter(*llvm_context);
        emitter.AddFunction(optimized, String(kernel.symbol));
        std::unique_ptr<llvm::Module> llvm_module = emitter.TakeModule();
        if (std::getenv("KXC_DUMP_KV_IR") != nullptr) {
            std::string text;
            llvm::raw_string_ostream raw(text);
            llvm_module->print(raw, nullptr);
            std::cerr << "==== IR " << kernel.symbol << " ====\n" << text;
        }
        codegen::LLVMJITEngine jit;
        codegen::CompiledKernel compiled = jit.Compile(
            std::move(llvm_module), std::move(llvm_context), signature,
            metadata, config->opt_level);
        entries.push_back(internal::CompiledModuleEntry{
            signature, metadata, std::move(compiled),
            std::move(module_contract)});
    }

    // Generated dynamic stateful plan over the compiled kernels.
    struct ValueDraft {
        int64_t id;
        int64_t storage;
        Array<int64_t> shape;
        DLDataType dtype;
        bool is_input;
        bool is_constant;
        bool is_output;
        bool is_alias;
        bool is_state;
        int64_t alias_source;
    };
    std::vector<ValueDraft> drafts;
    int64_t next_id = 0;
    const auto push = [&](int64_t storage, Array<int64_t> shape, DLDataType dtype,
                          bool is_input, bool is_constant, bool is_output,
                          bool is_alias, bool is_state, int64_t alias_source) {
        const int64_t id = next_id++;
        drafts.push_back(ValueDraft{id, storage, std::move(shape), dtype,
                                    is_input, is_constant, is_output, is_alias,
                                    is_state, alias_source});
        return id;
    };
    const Array<int64_t> state_shape({B, C, H, D});
    const Array<int64_t> tokens_shape({B, J, H, D});
    const DLDataType f32 = runtime::DataTypeFromString("float32");
    const DLDataType u64 = DLDataType{kDLUInt, 64, 1};
    std::vector<int64_t> state_ids;
    for (int64_t index = 0; index < declaration.state_count; ++index) {
        const int64_t id = next_id;
        state_ids.push_back(push(id, state_shape, f32, false, false, false,
                                 false, true, -1));
    }
    const int64_t tokens_k_id =
        push(next_id, tokens_shape, f32, true, false, false, false, false, -1);
    int64_t tokens_v_id = -1;
    int64_t q_id = -1;
    if (declaration.state_count == 2) {
        tokens_v_id = push(next_id, tokens_shape, f32, true, false, false,
                           false, false, -1);
        q_id = push(next_id, tokens_shape, f32, true, false, false, false,
                    false, -1);
    }
    const int64_t count_id =
        push(next_id, {1}, u64, true, false, false, false, false, -1);
    std::vector<int64_t> alias_ids;
    for (int64_t index = 0; index < declaration.state_count; ++index) {
        alias_ids.push_back(push(state_ids[index], state_shape, f32, false,
                                 false, false, true, false, state_ids[index]));
    }
    const int64_t read_id =
        push(next_id, declaration.state_count == 1 ? state_shape : tokens_shape,
             f32, false, false, true, false, false, -1);
    int64_t scale_id = -1;
    if (declaration.state_count == 2) {
        scale_id = push(next_id, {1}, f32, false, true, false, false, false, -1);
    }

    Array<runtime::ValueSpec> values;
    for (const ValueDraft& draft : drafts) {
        values.push_back(runtime::ValueSpec(
            draft.id, draft.storage, draft.shape, draft.dtype,
            declaration.device, draft.is_input, draft.is_constant,
            draft.is_output, draft.is_alias, false, draft.is_state,
            draft.alias_source,
            draft.is_alias ? runtime::ValueWriteMode::kInPlace
                           : runtime::ValueWriteMode::kAllocate,
            -1, draft.is_state ? C : -1, draft.is_state ? 1 : -1,
            draft.is_state ? declaration.invalid_fill : 0.0));
    }

    const auto call = [](const std::string& symbol,
                         const std::vector<int64_t>& inputs,
                         const std::vector<int64_t>& outputs) {
        Array<int64_t> in;
        Array<int64_t> out;
        for (int64_t id : inputs) in.push_back(id);
        for (int64_t id : outputs) out.push_back(id);
        return runtime::KernelCall(String(symbol), in, out);
    };
    Array<runtime::KernelCall> calls;
    if (declaration.state_count == 1) {
        calls.push_back(call("kv_state.append",
                             {state_ids[0], tokens_k_id, count_id},
                             {alias_ids[0]}));
        calls.push_back(
            call("kv_state.read_valid", {alias_ids[0], count_id}, {read_id}));
    } else {
        calls.push_back(call("kv_state.append_k",
                             {state_ids[0], tokens_k_id, count_id},
                             {alias_ids[0]}));
        calls.push_back(call("kv_state.append_v",
                             {state_ids[1], tokens_v_id, count_id},
                             {alias_ids[1]}));
        calls.push_back(call("kv_state.causal_attention",
                             {alias_ids[0], alias_ids[1], q_id, count_id,
                              scale_id},
                             {read_id}));
    }
    Array<int64_t> graph_inputs;
    graph_inputs.push_back(tokens_k_id);
    if (tokens_v_id != -1) graph_inputs.push_back(tokens_v_id);
    if (q_id != -1) graph_inputs.push_back(q_id);
    graph_inputs.push_back(count_id);
    Array<int64_t> graph_constants;
    if (scale_id != -1) graph_constants.push_back(scale_id);
    Array<int64_t> graph_outputs{read_id};
    Array<int64_t> graph_states;
    for (int64_t id : state_ids) graph_states.push_back(id);

    runtime::ExecutablePlan plan(values, calls, graph_inputs, graph_constants,
                                 graph_outputs, graph_states,
                                 runtime::ExecutablePlanMode::kDynamicStatefulV1,
                                 {}, extent_bindings, count_id);
    plan.Validate();

    CompiledModule module =
        internal::BuildCompiledModule(target, std::move(entries),
                                      std::move(constants));
    // Fail fast through the runtime's own module/plan validator.
    (void)runtime::RuntimeSession(module, plan);

    KvStatePlan result{CompiledModule(ObjectRef()), runtime::ExecutablePlan(), {}};
    result.module = std::move(module);
    result.plan = std::move(plan);
    for (size_t index = 0; index < kernels.size(); ++index) {
        const std::string unit_canonical =
            declaration_canonical + "kernel_role=" +
            kernels[index].operator_name + ";";
        const PrimitiveArtifactKey key = internal::BuildPrimitiveArtifactKey(
            UnitSemanticKey(unit_canonical), target, contract.canonical_bytes,
            relay::internal::GetTEScheduleContract(optimized_functions[index])
                .c_str(),
            BackendVersionFor(target));
        result.artifacts.push_back(
            OrderedArtifactIdentity{index, kernels[index].symbol, key});
    }
    return result;
#endif
}

}  // namespace kxc::api
