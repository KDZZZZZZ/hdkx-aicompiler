/*! \file test/compiler_contract_test.cpp
 * \brief 锁定 Compiler 前端与 Relay-to-TIR lowering 的公共契约。
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/distributed/executor.h"
#include "kxc/distributed/session.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/pass/context.h"
#include "kxc/relay/visitor.h"
#include "kxc/ffi/registry.h"
#include "kxc/runtime/kernel_abi.h"
#include "../src/compiler/internal/kernel_abi_builder.h"
#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/compiler/distributed/multi_device.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

// 验证调用明确抛出标准异常，用于锁定非法编译输入的拒绝边界。
bool Throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// 验证失败消息包含稳定片段，保证缺失 backend 时不会退化为模糊异常。
bool ThrowsWithMessage(const std::function<void()>& fn, const std::string& expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

// 检查 Pass 策略是否包含指定名称，避免测试依赖容器内部表示。
bool ContainsPass(const kxc::Array<kxc::String>& passes, const char* name) {
    for (const auto& pass : passes) {
        if (std::string(pass) == name) return true;
    }
    return false;
}

// 从 PrimFunc attrs 读取 lowering 写入的整数契约，缺失或类型错误均视为测试失败。
bool ReadIntAttr(const kxc::tir::PrimFunc& function, const char* key, int64_t* value) {
    const kxc::String attr_key(key);
    if (!function.defined() || !function->attrs.count(attr_key)) {
        return false;
    }
    const auto* integer = function->attrs.at(attr_key).As<kxc::tir::IntImmNode>();
    if (!integer) {
        return false;
    }
    *value = integer->value;
    return true;
}

// 比较静态 Buffer shape，避免依赖当前临时变量名来推断参数角色。
bool BufferHasShape(const kxc::tir::Buffer& buffer,
                    const std::vector<int64_t>& expected) {
    if (!buffer.defined() || buffer->shape.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto* dimension = buffer->shape[i].As<kxc::tir::IntImmNode>();
        if (!dimension || dimension->value != expected[i]) {
            return false;
        }
    }
    return true;
}

// 返回指定参数对应的 Buffer；本辅助函数只在先验证 buffer_map 后调用。
kxc::tir::Buffer ParamBuffer(const kxc::tir::PrimFunc& function, size_t index) {
    return function->buffer_map.at(function->params[index]);
}

// 将 lowering 的有序绑定转成签名构建器需要的 key -> payload 映射。
kxc::Map<kxc::String, kxc::runtime::NDArray> ConstantMap(
    const kxc::relay::LoweredFunction& lowered) {
    kxc::Map<kxc::String, kxc::runtime::NDArray> result;
    for (const auto& binding : lowered.constants()) {
        result.Set(binding->key, binding->value);
    }
    return result;
}

// 深拷贝 attrs，确保负例修改不会通过共享 Map 别名污染基准 PrimFunc。
kxc::Map<kxc::String, kxc::ObjectRef> CloneAttrs(const kxc::tir::PrimFunc& function) {
    kxc::Map<kxc::String, kxc::ObjectRef> result;
    for (const auto& item : function->attrs) result.Set(item.first, item.second);
    return result;
}

// 深拷贝 buffer_map，并可跳过指定参数以构造缺失 Buffer 的畸形产物。
kxc::Map<kxc::tir::Var, kxc::tir::Buffer> CloneBufferMapExcept(
    const kxc::tir::PrimFunc& function, const kxc::tir::Var& skipped) {
    kxc::Map<kxc::tir::Var, kxc::tir::Buffer> result;
    for (const auto& item : function->buffer_map) {
        if (!skipped.defined() || item.first.get() != skipped.get()) {
            result.Set(item.first, item.second);
        }
    }
    return result;
}

// 未定义 Function 或未定义函数体必须在进入 TE/TIR 转换前被拒绝。
bool TestInvalidFunctionRejected() {
    TEST_CHECK(Throws([] { kxc::relay::LowerToTIR(kxc::Function()); }),
               "undefined Function should be rejected");

    kxc::Var input("input", kxc::TensorType({2}, "float32"));
    kxc::Function missing_body({input}, kxc::Expr());
    TEST_CHECK(Throws([&] { kxc::relay::LowerToTIR(missing_body); }),
               "Function with undefined body should be rejected");
    return true;
}

// 构造具有指定 kind/type/可用性的 Target，用于覆盖配置边界而不依赖本机 CUDA。
kxc::Target MakeContractTarget(const std::string& kind, kxc::DeviceTypeCode device_type,
                               int device_id, bool available) {
    auto* node = new kxc::TargetNode();
    node->kind = kind;
    node->device_type = device_type;
    node->device_id = device_id;
    node->attrs.exists = available ? 1 : 0;
    node->attrs.device_name = "contract-device";
    node->attrs.arch = device_type == kxc::kCPU ? "contract-cpu" : "sm_75";
    node->attrs.max_threads_per_block = 1;
    node->attrs.warp_size = 1;
    node->attrs.multi_processor_count = 1;
    return kxc::Target(kxc::ObjectRef(node));
}

// CompileConfig 必须只接受类型正确、优化等级合法且 Target 自洽的对象。
bool TestCompileConfigValidation() {
    using namespace kxc;

    api::CompileConfig valid = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    TEST_CHECK(valid->opt_level == 2 && valid->target->kind == "llvm",
               "valid CPU CompileConfig fields mismatch");
    valid.Validate();

    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(BuildTarget(Device::CPU()), -1);
               }),
               "negative opt_level should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(BuildTarget(Device::CPU()), 4);
               }),
               "opt_level above three should fail");
    TEST_CHECK(Throws([] { api::CompileConfig::Create(Target(), 1); }),
               "undefined target should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(Target(ObjectRef(String("not-a-target"))), 1);
               }),
               "wrong Target ObjectRef type should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig wrong(ObjectRef(String("not-a-config")));
               }),
               "wrong ObjectRef type should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(
                       MakeContractTarget("cuda", kCPU, 0, true), 1);
               }),
               "CUDA kind on CPU device should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(
                       MakeContractTarget("llvm", kCPU, 1, true), 1);
               }),
               "non-zero CPU device id should fail");
    TEST_CHECK(Throws([] {
                   api::CompileConfig::Create(
                       MakeContractTarget("cuda", kCUDA, 0, false), 1);
               }),
               "unavailable target should fail");

    valid->opt_level = 9;
    TEST_CHECK(Throws([&] { valid.Validate(); }),
               "mutated invalid opt_level should fail revalidation");
    return true;
}

// Compiler 必须只按 Target 选择 backend，并对尚不可用的执行路径给出精确诊断。
bool TestCompilerTargetDispatch() {
    using namespace kxc;

    Var lhs("lhs", TensorType({1}, "float32"));
    Var rhs("rhs", TensorType({1}, "float32"));
    Function add({lhs, rhs}, Call(relay::Op::Get("add"), {lhs, rhs}));

#if !KXC_USE_LLVM
    api::CompileConfig cpu = api::CompileConfig::Create(BuildTarget(Device::CPU()), 0);
    TEST_CHECK(ThrowsWithMessage([&] { api::Compiler::Compile(add, cpu); },
                                 "KXC_ENABLE_LLVM=ON"),
               "LLVM-disabled CPU target should report the required build feature");
#endif

#if !KXC_USE_CUDA
    api::CompileConfig cuda = api::CompileConfig::Create(
        MakeContractTarget("cuda", kCUDA, 0, true), 0);
    TEST_CHECK(ThrowsWithMessage([&] { api::Compiler::Compile(add, cuda); },
                                 "KXC_ENABLE_CUDA=ON"),
               "CUDA-disabled target should report the required build feature");
#endif
    return true;
}

// opt_level 必须映射到稳定策略，且 CUDA O3 不得运行会破坏绑定前置结构的循环 Pass。
bool TestCompilerPassPolicies() {
    using namespace kxc;
    const Array<String> relay0 = api::Compiler::RelayPassPolicy(0);
    const Array<String> relay1 = api::Compiler::RelayPassPolicy(1);
    const Array<String> relay2 = api::Compiler::RelayPassPolicy(2);
    const Array<String> relay3 = api::Compiler::RelayPassPolicy(3);
    TEST_CHECK(relay0.empty() && relay1.size() == 3 && relay2.size() == 6 &&
                   relay3.size() == 6,
               "Relay opt_level policy changed unexpectedly");
    TEST_CHECK(!ContainsPass(relay3, "eliminate_common_subexpr") &&
                   !ContainsPass(relay3, "annotate_memory_scope") &&
                   !ContainsPass(relay3, "capture_post_dfs_index_in_spans") &&
                   !ContainsPass(relay3, "infer_type"),
               "Relay O3 should exclude unsafe CSE, annotation passes and duplicate InferType");

    const Target cpu = BuildTarget(Device::CPU());
    const Target cuda = MakeContractTarget("cuda", kCUDA, 0, true);
    const Array<String> tir0 = api::Compiler::TIRPassPolicy(0, cpu);
    const Array<String> tir1 = api::Compiler::TIRPassPolicy(1, cpu);
    const Array<String> tir2 = api::Compiler::TIRPassPolicy(2, cpu);
    const Array<String> tir3 = api::Compiler::TIRPassPolicy(3, cpu);
    const Array<String> cuda3 = api::Compiler::TIRPassPolicy(3, cuda);
    TEST_CHECK(tir0.empty() && tir1.size() == 2 && tir2.size() == 4 &&
                   tir3.size() == 8,
               "TIR opt_level policy changed unexpectedly");
    TEST_CHECK(cuda3.size() == 4 && ContainsPass(cuda3, "remove_no_op") &&
                   !ContainsPass(cuda3, "convert_for_loops_serial") &&
                   !ContainsPass(cuda3, "loop_partition") &&
                   !ContainsPass(cuda3, "unroll_loop") &&
                   !ContainsPass(cuda3, "vectorize_loop"),
               "CUDA O3 should preserve the serial loop expected by BindCudaThreads");
    TEST_CHECK(Throws([] { api::Compiler::RelayPassPolicy(-1); }) &&
                   Throws([&] { api::Compiler::TIRPassPolicy(4, cpu); }),
               "pass policy should reject an invalid opt_level");
    return true;
}

// CompileConfig Target 填充缺失 placement，但不得覆盖已有的不同设备身份。
bool TestPassContextTargetMerge() {
    using namespace kxc;
    const Target cpu = BuildTarget(Device::CPU());
    const PassContext from_target = PassContext::FromTarget(cpu);
    TEST_CHECK(from_target.defined() && from_target.default_target()->kind == "llvm" &&
                   from_target.default_device() == Device::CPU(),
               "PassContext::FromTarget did not preserve CPU identity");

    Var input("input", TensorType({1}, "float32"));
    Function function({input}, input);
    const PassContext merged =
        PassContext::MergeTarget(relay::PassContextFromRelay(function), cpu);
    TEST_CHECK(merged.default_target()->kind == "llvm" &&
                   merged.default_device() == Device::CPU(),
               "target was not installed into an unplaced Relay function");

    const Target cuda = MakeContractTarget("cuda", kCUDA, 0, true);
    input.set_virtual_device(VirtualDevice(cuda));
    Function conflicting({input}, input);
    TEST_CHECK(Throws([&] {
                   PassContext::MergeTarget(relay::PassContextFromRelay(conflicting), cpu);
               }),
               "conflicting Relay placement should not be overwritten");
    return true;
}

// 锁定 lowering 的 input -> constant -> output 参数顺序及对应计数 attrs。
bool TestInputConstantOutputOrder() {
    using namespace kxc;

    Var input("input", TensorType({2, 3}, "float32"));
    runtime::NDArray constant_data = runtime::NDArray::Zeros(
        {3}, runtime::DataTypeFromString("float32"), Device::CPU());
    Constant constant(constant_data);
    Call add(relay::Op::Get("add"), {input, constant});
    relay::LoweredFunction result = relay::LowerToTIR(Function({input}, add));
    tir::PrimFunc lowered = result->prim_func;

    int64_t input_count = -1;
    int64_t constant_count = -1;
    int64_t output_count = -1;
    int64_t output_start = -1;
    TEST_CHECK(ReadIntAttr(lowered, "kxc.input_count", &input_count),
               "missing kxc.input_count");
    TEST_CHECK(ReadIntAttr(lowered, "kxc.constant_count", &constant_count),
               "missing kxc.constant_count");
    TEST_CHECK(ReadIntAttr(lowered, "kxc.output_count", &output_count),
               "missing kxc.output_count");
    TEST_CHECK(ReadIntAttr(lowered, "kxc.output_param_start", &output_start),
               "missing kxc.output_param_start");
    TEST_CHECK(input_count == 1 && constant_count == 1 && output_count == 1,
               "lowering parameter counts do not match input/constant/output roles");
    TEST_CHECK(output_start == 2 && lowered->params.size() == 3,
               "output parameter must follow input and constant parameters");
    const Array<relay::ConstantBinding> bindings = result.constants();
    TEST_CHECK(bindings.size() == 1,
               "lowering must retain one constant binding");
    TEST_CHECK(std::string(bindings[0]->key) == "relay.constant.0" &&
                   bindings[0]->param_index == 1 &&
                   bindings[0]->value.get() == constant_data.get(),
               "constant binding key, parameter slot, or payload mismatch");
    const codegen::KernelConstantKeys constant_key_list(
        lowered->attrs.at(String("kxc.constant_keys")));
    const Array<String> constant_keys = constant_key_list.keys();
    TEST_CHECK(constant_keys.size() == 1 &&
                   std::string(constant_keys[0]) == "relay.constant.0",
               "PrimFunc constant key metadata mismatch");

    for (const auto& parameter : lowered->params) {
        TEST_CHECK(lowered->buffer_map.count(parameter) == 1,
                   "every lowered parameter must have a Buffer entry");
    }
    TEST_CHECK(BufferHasShape(ParamBuffer(lowered, 0), {2, 3}),
               "input Buffer shape mismatch");
    TEST_CHECK(BufferHasShape(ParamBuffer(lowered, 1), {3}),
               "constant Buffer shape mismatch");
    TEST_CHECK(BufferHasShape(ParamBuffer(lowered, 2), {2, 3}),
               "output Buffer shape mismatch");
    TEST_CHECK(ParamBuffer(lowered, 0)->dtype == tir::DataType::Float(32) &&
                   ParamBuffer(lowered, 1)->dtype == tir::DataType::Float(32) &&
                   ParamBuffer(lowered, 2)->dtype == tir::DataType::Float(32),
               "lowered Buffer dtype mismatch");

    const codegen::KernelSignature signature = codegen::BuildKernelSignature(
        lowered, ConstantMap(result), BuildTarget(Device::CPU()), "contract_add");
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    TEST_CHECK(arguments.size() == 3 &&
                   arguments[0]->role == codegen::KernelArgRole::kInput &&
                   arguments[1]->role == codegen::KernelArgRole::kConstant &&
                   arguments[2]->role == codegen::KernelArgRole::kOutput,
               "signature roles do not match lowered parameter order");
    TEST_CHECK(std::string(arguments[1]->constant_key) == "relay.constant.0" &&
                   arguments[2]->mutable_data,
               "signature constant key or output mutability mismatch");
    return true;
}

// 同一 Constant 节点复用一次绑定；相同 payload 的不同节点仍占独立参数槽。
bool TestConstantBindingIdentity() {
    using namespace kxc;

    runtime::NDArray data = runtime::NDArray::Zeros(
        {4}, runtime::DataTypeFromString("float32"), Device::CPU());
    Constant shared(data);
    Function shared_function({}, Call(relay::Op::Get("add"), {shared, shared}));
    relay::LoweredFunction shared_lowered = relay::LowerToTIR(shared_function);
    TEST_CHECK(shared_lowered.constants().size() == 1,
               "shared Constant node should occupy one binding");

    Constant first(data);
    Constant second(data);
    Function distinct_function({}, Call(relay::Op::Get("add"), {first, second}));
    relay::LoweredFunction first_lowering = relay::LowerToTIR(distinct_function);
    relay::LoweredFunction second_lowering = relay::LowerToTIR(distinct_function);
    const Array<relay::ConstantBinding> first_bindings = first_lowering.constants();
    const Array<relay::ConstantBinding> second_bindings = second_lowering.constants();
    TEST_CHECK(first_bindings.size() == 2,
               "distinct Constant nodes should occupy separate bindings");
    TEST_CHECK(std::string(first_bindings[0]->key) == "relay.constant.0" &&
                   std::string(first_bindings[1]->key) == "relay.constant.1",
               "distinct Constant keys must follow deterministic traversal order");
    TEST_CHECK(first_bindings[0]->param_index == 0 &&
                   first_bindings[1]->param_index == 1,
               "constant parameter slots must match binding order");
    TEST_CHECK(std::string(second_bindings[0]->key) ==
                       std::string(first_bindings[0]->key) &&
                   std::string(second_bindings[1]->key) ==
                       std::string(first_bindings[1]->key),
               "repeated lowering changed stable Constant keys");
    return true;
}

// 直接构造畸形对象，验证 ObjectRef 恢复和常量 Buffer ABI 校验不能被绕过。
bool TestLoweredObjectValidation() {
    using namespace kxc;

    runtime::NDArray data = runtime::NDArray::Zeros(
        {3}, runtime::DataTypeFromString("float32"), Device::CPU());
    Var input("input", TensorType({2, 3}, "float32"));
    Constant constant(data);
    Function function({input}, Call(relay::Op::Get("add"), {input, constant}));
    relay::LoweredFunction valid = relay::LowerToTIR(function);
    const tir::PrimFunc base = valid->prim_func;
    const Array<relay::ConstantBinding> valid_bindings = valid.constants();

    Array<relay::ConstantBinding> returned = valid.constants();
    returned.push_back(valid_bindings[0]);
    TEST_CHECK(valid.constants().size() == 1,
               "mutating returned constants changed LoweredFunction");

    TEST_CHECK(Throws([] {
                   relay::ConstantBinding("", runtime::NDArray(), -1);
               }),
               "invalid ConstantBinding fields should fail");
    TEST_CHECK(Throws([] {
                   ObjectRef raw(new relay::ConstantBindingNode());
                   relay::ConstantBinding invalid(raw);
               }),
               "same-type invalid ConstantBindingNode should be revalidated");
    TEST_CHECK(Throws([] {
                   ObjectRef raw(new codegen::KernelConstantKeysNode());
                   codegen::KernelConstantKeys invalid(raw);
               }) == false,
               "empty KernelConstantKeys is valid for functions without constants");
    TEST_CHECK(Throws([] {
                   ObjectRef raw(new relay::LoweredFunctionNode());
                   relay::LoweredFunction invalid(raw);
               }),
               "same-type invalid LoweredFunctionNode should be revalidated");

    runtime::NDArray wrong_dtype = runtime::NDArray::Zeros(
        {3}, runtime::DataTypeFromString("int32"), Device::CPU());
    relay::ConstantBinding dtype_binding(valid_bindings[0]->key, wrong_dtype,
                                         valid_bindings[0]->param_index);
    TEST_CHECK(Throws([&] { relay::LoweredFunction invalid(base, {dtype_binding}); }),
               "constant dtype mismatch should fail");

    runtime::NDArray wrong_shape = runtime::NDArray::Zeros(
        {4}, runtime::DataTypeFromString("float32"), Device::CPU());
    relay::ConstantBinding shape_binding(valid_bindings[0]->key, wrong_shape,
                                         valid_bindings[0]->param_index);
    TEST_CHECK(Throws([&] { relay::LoweredFunction invalid(base, {shape_binding}); }),
               "constant shape mismatch should fail");

    relay::ConstantBinding wrong_key("relay.constant.other", data,
                                     valid_bindings[0]->param_index);
    TEST_CHECK(Throws([&] { relay::LoweredFunction invalid(base, {wrong_key}); }),
               "constant key metadata mismatch should fail");

    Map<tir::Var, tir::Buffer> missing_buffer =
        CloneBufferMapExcept(base, base->params[1]);
    tir::PrimFunc missing_buffer_func(base->params, base->body, missing_buffer, base->attrs);
    TEST_CHECK(Throws([&] {
                   relay::LoweredFunction invalid(missing_buffer_func, valid_bindings);
               }),
               "constant parameter without Buffer should fail");

    Map<String, ObjectRef> wrong_key_attr = CloneAttrs(base);
    wrong_key_attr.Set(String("kxc.constant_keys"), String("not-a-key-list"));
    tir::PrimFunc wrong_key_func(base->params, base->body, base->buffer_map, wrong_key_attr);
    TEST_CHECK(Throws([&] {
                   relay::LoweredFunction invalid(wrong_key_func, valid_bindings);
               }),
               "wrong constant key attr type should fail safely");

    Map<String, ObjectRef> wrong_output_attr = CloneAttrs(base);
    wrong_output_attr.Set(String("kxc.output_param_start"),
                          tir::IntImm(0, tir::DataType::Int(64)));
    tir::PrimFunc wrong_output_func(base->params, base->body, base->buffer_map,
                                    wrong_output_attr);
    TEST_CHECK(Throws([&] {
                   relay::LoweredFunction invalid(wrong_output_func, valid_bindings);
               }),
               "wrong output parameter range should fail");
    return true;
}

// 唯一 LowerToTIR 入口必须保活常量，同时旧转发 PackedFunc 不得继续暴露。
bool TestMultiDeviceLoweringEntry() {
    using namespace kxc;

    runtime::NDArray data = runtime::NDArray::Zeros(
        {2}, runtime::DataTypeFromString("float32"), Device::CPU());
    Var input("input", TensorType({2}, "float32"));
    Constant constant(data);
    Function function({input}, Call(relay::Op::Get("add"), {input, constant}));

    relay::LoweredFunction direct = relay::LowerToTIR(function);
    TEST_CHECK(direct.constants().size() == 1 &&
                   direct.constants()[0]->value.get() == data.get(),
               "LowerToTIR lost constant payload");
    TEST_CHECK(!Registry::Global()
                    .Get("kxc.relay.transform.lower_compute_to_tir")
                    .defined(),
               "obsolete lower_compute_to_tir forwarding entry should be removed");
    return true;
}

// ExecutionPlan 尚未绑定 CompiledModule 时必须明确失败，不能伪造零值或复制输入。
bool TestExecutionPlanKernelFailsClosed() {
    using namespace kxc;

    const PassContext pass_ctx = PassContext::FromTarget(BuildTarget(Device::CPU()));
    KernelExec kernel("add", tir::PrimFunc(), {0}, {1}, {0}, "missing_module");
    Map<int, Array<int64_t>> shapes;
    shapes.Set(0, {1});
    shapes.Set(1, {1});
    Map<int, std::string> dtypes;
    dtypes.Set(0, "float32");
    dtypes.Set(1, "float32");
    ExecutionPlan plan({ObjectRef(kernel)}, {}, {0}, {}, shapes, dtypes, 2,
                       pass_ctx, DiscoPlacement(), 1);

    disco::DiscoSession session = disco::DiscoSession::ThreadedSession(1, 1);
    disco::DRef input = session.Empty({1}, "float32", false, false);
    Map<int, disco::DRef> initial_values;
    initial_values.Set(0, input);
    disco::ExecutionPlanExecutor executor(session);
    TEST_CHECK(ThrowsWithMessage(
                   [&] { (void)executor.Execute(plan, initial_values); },
                   "CompiledModule launch is not implemented"),
               "ExecutionPlan kernel path should fail before producing an output");

    TEST_CHECK(!Registry::Global().Get("kxc.disco.execute_plan").defined() &&
                   !Registry::Global().Get("kxc.disco.execute_plan_json").defined(),
               "incomplete ExecutionPlan execution should not be exposed through FFI");
    return true;
}

// 多输出必须各占一个独立参数槽，并共享同一个稳定 output 起点。
bool TestMultiOutputMetadata() {
    using namespace kxc;

    Var lhs("lhs", TensorType({4}, "float32"));
    Var rhs("rhs", TensorType({4}, "float32"));
    Call add(relay::Op::Get("add"), {lhs, rhs});
    Call multiply(relay::Op::Get("mul"), {lhs, rhs});
    Function function({lhs, rhs}, Tuple({add, multiply}));

    relay::LoweredFunction first_result = relay::LowerToTIR(function);
    relay::LoweredFunction second_result = relay::LowerToTIR(function);
    tir::PrimFunc first = first_result->prim_func;
    tir::PrimFunc second = second_result->prim_func;
    int64_t output_count = -1;
    int64_t output_start = -1;
    TEST_CHECK(ReadIntAttr(first, "kxc.output_count", &output_count) && output_count == 2,
               "tuple lowering must expose two output parameters");
    TEST_CHECK(ReadIntAttr(first, "kxc.output_param_start", &output_start) &&
                   output_start == 2,
               "tuple outputs must start after both inputs");
    TEST_CHECK(first->params.size() == 4,
               "two inputs and two outputs should produce four parameters");
    TEST_CHECK(BufferHasShape(ParamBuffer(first, 2), {4}) &&
                   BufferHasShape(ParamBuffer(first, 3), {4}),
               "tuple output Buffer shapes mismatch");
    const codegen::KernelSignature signature = codegen::BuildKernelSignature(
        first, ConstantMap(first_result), BuildTarget(Device::CPU()), "tuple_kernel");
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    TEST_CHECK(arguments.size() == 4 &&
                   arguments[2]->role == codegen::KernelArgRole::kOutput &&
                   arguments[3]->role == codegen::KernelArgRole::kOutput,
               "multi-output signature roles mismatch");

    // 重复 lowering 必须至少保持参数数量、顺序元数据和 Buffer shape 确定。
    int64_t second_output_start = -1;
    TEST_CHECK(ReadIntAttr(second, "kxc.output_param_start", &second_output_start) &&
                   second_output_start == output_start,
               "repeated lowering changed output parameter order");
    TEST_CHECK(second->params.size() == first->params.size(),
               "repeated lowering changed parameter count");
    for (size_t i = 0; i < first->params.size(); ++i) {
        TEST_CHECK(BufferHasShape(ParamBuffer(first, i), {4}) &&
                       BufferHasShape(ParamBuffer(second, i), {4}),
                   "repeated lowering changed Buffer shape");
    }
    return true;
}

}  // namespace

// 顺序运行所有契约用例并汇总失败，确保 CI 能看到可靠的非零退出码。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"invalid_function_rejected", TestInvalidFunctionRejected},
        {"compile_config_validation", TestCompileConfigValidation},
        {"compiler_target_dispatch", TestCompilerTargetDispatch},
        {"compiler_pass_policies", TestCompilerPassPolicies},
        {"pass_context_target_merge", TestPassContextTargetMerge},
        {"input_constant_output_order", TestInputConstantOutputOrder},
        {"constant_binding_identity", TestConstantBindingIdentity},
        {"lowered_object_validation", TestLoweredObjectValidation},
        {"multi_device_lowering_entry", TestMultiDeviceLoweringEntry},
        {"execution_plan_kernel_fails_closed", TestExecutionPlanKernelFailsClosed},
        {"multi_output_metadata", TestMultiOutputMetadata},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
