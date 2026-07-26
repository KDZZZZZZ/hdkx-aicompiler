/*! \file test/pass_pipeline_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "kxc/target/target.h"
#include "kxc/pass/pass.h"
#include "kxc/relay/op.h"
#include "kxc/relay/printer/print_ir.h"
#include "kxc/relay/transforms/annotate_memory_scope.h"
#include "kxc/relay/transforms/canonicalize_cast.h"
#include "kxc/relay/transforms/capture_post_dfs_index_in_spans.h"
#include "kxc/relay/transforms/eliminate_common_subexpr.h"
#include "kxc/relay/transforms/eliminate_dead_let.h"
#include "kxc/relay/transforms/fold_constant.h"
#include "kxc/relay/transforms/fold_tuple_get_item.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/relay/transforms/remove_standalone_reshapes.h"
#include "kxc/relay/transforms/simplify_expr.h"
#include "kxc/tir/printer/print_ir.h"
#include "kxc/tir/transforms/convert_for_loops_serial.h"
#include "kxc/tir/transforms/fold_constant.h"
#include "kxc/tir/transforms/force_narrow_index_to_i32.h"
#include "kxc/tir/transforms/loop_partition.h"
#include "kxc/tir/transforms/pipeline.h"
#include "kxc/tir/transforms/remove_no_op.h"
#include "kxc/tir/transforms/simplify_expr.h"
#include "kxc/tir/transforms/unroll_loop.h"
#include "kxc/tir/transforms/vectorize_loop.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, msg)                                                     \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (msg) << "\n";     \
            return false;                                                         \
        }                                                                         \
    } while (0)

// 在显式 cpu:0 NDArray 中构造 float32 标量常量。
kxc::runtime::NDArray MakeScalarFloat(float value) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        {}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    float* data = static_cast<float*>(array->dl_tensor.data);
    *data = value;
    return array;
}

// 在显式 cpu:0 NDArray 中构造 int64 标量常量。
kxc::runtime::NDArray MakeScalarInt64(int64_t value) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        {}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
    int64_t* data = static_cast<int64_t*>(array->dl_tensor.data);
    *data = value;
    return array;
}

// 在显式 cpu:0 NDArray 中构造 int32 标量常量。
kxc::runtime::NDArray MakeScalarInt32(int32_t value) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        {}, kxc::runtime::DataTypeFromString("int32"), kxc::Device::CPU());
    int32_t* data = static_cast<int32_t*>(array->dl_tensor.data);
    *data = value;
    return array;
}

// 将 CPU 标量 NDArray 包装为 Relay float32 Constant。
kxc::Constant MakeRelayScalarFloat(float value) { return kxc::Constant(MakeScalarFloat(value)); }
// 将 CPU 标量 NDArray 包装为 Relay int64 Constant。
kxc::Constant MakeRelayScalarInt64(int64_t value) { return kxc::Constant(MakeScalarInt64(value)); }
// 将 CPU 标量 NDArray 包装为 Relay int32 Constant。
kxc::Constant MakeRelayScalarInt32(int32_t value) { return kxc::Constant(MakeScalarInt32(value)); }

kxc::Call MakeRelayBinary(const std::string& op_name, const kxc::Expr& lhs,
                          const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get(op_name), {lhs, rhs});
}

// 将 Relay 函数序列化为便于断言的稳定文本。
std::string RelayText(const kxc::Function& func) { return kxc::relay::printer::ToText(func); }

// 将 TIR PrimFunc 序列化为便于断言的稳定文本。
std::string TIRText(const kxc::tir::PrimFunc& func) {
    std::ostringstream os;
    kxc::tir::printer::DumpPrimFunc(func, os);
    return os.str();
}

std::vector<std::string> ToStdVector(const kxc::Array<kxc::String>& values) {
    std::vector<std::string> out;
    for (const auto& value : values) {
        out.push_back(static_cast<std::string>(value));
    }
    return out;
}

kxc::PassSpec MakeTestSpec(const char* name, kxc::IRDialect dialect, kxc::PassScope scope,
                           const char* phase, const char* implementation_key) {
    kxc::PassSpec spec;
    spec.name = kxc::String(name);
    spec.schema_version = 1;
    spec.dialect = dialect;
    spec.scope = scope;
    spec.phase = kxc::String(phase);
    spec.opt_level = 1;
    spec.deterministic = true;
    spec.implementation_key = kxc::String(implementation_key);
    return spec;
}

// 递归统计 Relay 表达式中的 Let 节点数量。
int CountLetNodes(const kxc::Expr& expr) {
    if (!expr.defined()) return 0;
    if (const auto* let_node = expr.As<kxc::LetNode>()) {
        return 1 + CountLetNodes(let_node->value) + CountLetNodes(let_node->body);
    }
    if (const auto* call = expr.As<kxc::CallNode>()) {
        int count = 0;
        for (const auto& arg : call->args) {
            count += CountLetNodes(arg);
        }
        return count;
    }
    if (const auto* fn = expr.As<kxc::FunctionNode>()) {
        return CountLetNodes(fn->body);
    }
    if (const auto* tuple = expr.As<kxc::TupleNode>()) {
        int count = 0;
        for (const auto& field : tuple->fields) {
            count += CountLetNodes(field);
        }
        return count;
    }
    if (const auto* tuple_get = expr.As<kxc::TupleGetItemNode>()) {
        return CountLetNodes(tuple_get->tuple);
    }
    if (const auto* if_node = expr.As<kxc::IfNode>()) {
        return CountLetNodes(if_node->cond) + CountLetNodes(if_node->true_branch) +
               CountLetNodes(if_node->false_branch);
    }
    return 0;
}

// 递归收集 TIR 循环类型，用于验证循环变换结果。
void CollectForTypes(const kxc::tir::Stmt& stmt, std::vector<kxc::tir::ForType>* out) {
    if (!stmt.defined()) return;
    if (const auto* for_node = stmt.As<kxc::tir::ForNode>()) {
        out->push_back(for_node->for_type);
        CollectForTypes(for_node->body, out);
        return;
    }
    if (const auto* let_node = stmt.As<kxc::tir::LetStmtNode>()) {
        CollectForTypes(let_node->body, out);
        return;
    }
    if (const auto* if_node = stmt.As<kxc::tir::IfThenElseNode>()) {
        CollectForTypes(if_node->then_case, out);
        CollectForTypes(if_node->else_case, out);
        return;
    }
    if (const auto* alloc_node = stmt.As<kxc::tir::AllocateNode>()) {
        CollectForTypes(alloc_node->body, out);
        return;
    }
    if (const auto* attr_node = stmt.As<kxc::tir::AttrStmtNode>()) {
        CollectForTypes(attr_node->body, out);
        return;
    }
    if (const auto* block_node = stmt.As<kxc::tir::BlockNode>()) {
        CollectForTypes(block_node->init, out);
        CollectForTypes(block_node->body, out);
        return;
    }
    if (const auto* seq_node = stmt.As<kxc::tir::SeqStmtNode>()) {
        for (const auto& child : seq_node->seq) {
            CollectForTypes(child, out);
        }
    }
}

// 验证常量索引的 TupleGetItem 被折叠为对应字段。
bool TestRelayFoldTupleGetItem() {
    kxc::Var x("x");
    kxc::Var y("y");
    kxc::Tuple tuple({x, y});
    kxc::TupleGetItem get_item(tuple, 1);
    kxc::Function func({x, y}, get_item);

    kxc::Function out = kxc::relay::FoldTupleGetItemPass(func);
    TEST_CHECK(out->body.As<kxc::VarNode>() != nullptr, "TupleGetItem should fold to Var");
    TEST_CHECK(out->body.get() == y.get(), "TupleGetItem index 1 should fold to y");
    return true;
}

// 验证 Relay 标量常量表达式在 CPU NDArray 上正确折叠。
bool TestRelayFoldConstant() {
    kxc::Expr folded_add = MakeRelayBinary("add", MakeRelayScalarInt64(2), MakeRelayScalarInt64(3));
    kxc::Expr folded_mul =
        MakeRelayBinary("mul", MakeRelayScalarInt64(3), MakeRelayScalarInt64(2));
    kxc::Expr keep_div_zero =
        MakeRelayBinary("divide", MakeRelayScalarFloat(1.0f), MakeRelayScalarFloat(0.0f));
    kxc::Tuple body({folded_add, folded_mul, keep_div_zero});
    kxc::Function func({}, body);

    kxc::Function out = kxc::relay::FoldConstantPass(func);
    const auto* tuple = out->body.As<kxc::TupleNode>();
    TEST_CHECK(tuple != nullptr, "Output must remain tuple");
    TEST_CHECK(tuple->fields[0].As<kxc::ConstantNode>() != nullptr, "add const fold should fire");
    TEST_CHECK(tuple->fields[1].As<kxc::ConstantNode>() != nullptr,
               "mul const fold should fire");
    TEST_CHECK(tuple->fields[2].As<kxc::CallNode>() != nullptr,
               "divide by zero should not fold");
    return true;
}

// 验证 Relay 代数恒等式和冗余表达式被简化。
bool TestRelaySimplifyExpr() {
    kxc::Var x("x");
    kxc::Expr add_zero = MakeRelayBinary("add", x, MakeRelayScalarFloat(0.0f));
    kxc::Expr add_one = MakeRelayBinary("add", x, MakeRelayScalarFloat(1.0f));
    kxc::Tuple body({add_zero, add_one});
    kxc::Function func({x}, body);

    kxc::Function out = kxc::relay::SimplifyExprPass(func);
    const auto* out_tuple = out->body.As<kxc::TupleNode>();
    TEST_CHECK(out_tuple != nullptr, "Output should remain tuple");
    TEST_CHECK(out_tuple->fields.size() == 2, "Tuple arity should stay 2");
    TEST_CHECK(out_tuple->fields[0].As<kxc::VarNode>() != nullptr, "x + 0 should simplify");
    TEST_CHECK(out_tuple->fields[0].get() == x.get(), "x + 0 should become x");
    TEST_CHECK(out_tuple->fields[1].As<kxc::CallNode>() != nullptr, "x + 1 should not simplify");
    return true;
}

// 验证冗余 Cast 被规范化且类型保持正确。
bool TestRelayCanonicalizeCast() {
    kxc::Var x("x");
    kxc::Expr inner = kxc::Call(kxc::relay::Op::Get("cast"), {x}, kxc::relay::CastAttrs::Create(1));
    kxc::Expr outer =
        kxc::Call(kxc::relay::Op::Get("cast"), {inner}, kxc::relay::CastAttrs::Create(1));
    kxc::Function func({x}, outer);

    kxc::Function out = kxc::relay::CanonicalizeCastPass(func);
    const auto* out_call = out->body.As<kxc::CallNode>();
    TEST_CHECK(out_call != nullptr, "Output should remain cast call");
    TEST_CHECK(out_call->args.size() == 1, "Cast should keep one arg");
    TEST_CHECK(out_call->args[0].As<kxc::VarNode>() != nullptr, "Nested cast should collapse");
    TEST_CHECK(out_call->args[0].get() == x.get(), "Collapsed cast should use original input");
    return true;
}

// 验证无语义作用的独立 Reshape 被删除。
bool TestRelayRemoveStandaloneReshapes() {
    kxc::Var x("x");
    kxc::Expr inner = kxc::Call(kxc::relay::Op::Get("reshape"), {x},
                                kxc::relay::ReshapeAttrs::Create({4}));
    kxc::Expr outer = kxc::Call(kxc::relay::Op::Get("reshape"), {inner},
                                kxc::relay::ReshapeAttrs::Create({8}));
    kxc::Function func({x}, outer);

    kxc::Function out = kxc::relay::RemoveStandaloneReshapesPass(func);
    const auto* call = out->body.As<kxc::CallNode>();
    TEST_CHECK(call != nullptr, "Output should remain reshape call");
    TEST_CHECK(call->args.size() == 1, "reshape should keep arity 1");
    TEST_CHECK(call->args[0].As<kxc::VarNode>() != nullptr, "Nested reshape should collapse");
    TEST_CHECK(call->args[0].get() == x.get(), "Collapsed reshape should use original tensor");
    return true;
}

// 验证结构相同的 Relay 子表达式被公共子表达式消除。
bool TestRelayEliminateCommonSubexpr() {
    kxc::Var x("x");
    kxc::Var a("a");
    kxc::Var b("b");
    kxc::Expr shared = MakeRelayBinary("add", x, MakeRelayScalarFloat(1.0f));
    kxc::Let inner(b, shared, MakeRelayBinary("add", a, b));
    kxc::Let outer(a, shared, inner);
    kxc::Function func({x}, outer);

    kxc::Function out = kxc::relay::EliminateCommonSubexprPass(func);
    TEST_CHECK(CountLetNodes(out->body) == 1, "Pure duplicated let should be CSE'd");
    const auto* outer_let = out->body.As<kxc::LetNode>();
    TEST_CHECK(outer_let != nullptr, "Outer let should remain");
    const auto* add_call = outer_let->body.As<kxc::CallNode>();
    TEST_CHECK(add_call != nullptr, "Inner body should become call");
    TEST_CHECK(add_call->args.size() == 2, "add should keep 2 args");
    TEST_CHECK(add_call->args[0].get() == add_call->args[1].get(),
               "CSE should rewrite b to existing a");

    return true;
}

// 验证死 Let 被删除，同时保留 VirtualDevice 放置信息。
bool TestRelayEliminateDeadLetAndVirtualDevice() {
    kxc::Var x("x");
    kxc::Var tmp("tmp");

    kxc::Let dead_let(tmp, MakeRelayScalarFloat(1.0f), x);
    kxc::VirtualDevice vd(kxc::BuildTarget(kxc::Device::CPU()), "global", 0);
    dead_let.set_virtual_device(vd);
    kxc::Function func_dead({x}, dead_let);

    kxc::Function out_dead = kxc::relay::EliminateDeadLetPass(func_dead);
    TEST_CHECK(out_dead->body.As<kxc::VarNode>() != nullptr, "Unused Let should be removed");
    TEST_CHECK(out_dead->body.get() == x.get(), "Dead Let body should become x");
    const auto* relay_node = dynamic_cast<const kxc::RelayNode*>(out_dead->body.get());
    TEST_CHECK(relay_node != nullptr, "Result should be RelayNode");
    TEST_CHECK(relay_node->virtual_device_.defined(), "virtual_device should be kept");
    TEST_CHECK(relay_node->virtual_device_.get() == vd.get(),
               "virtual_device should match original Let virtual_device");
    return true;
}

// 验证后序 DFS 编号稳定写入表达式 Span。
bool TestRelayCapturePostDfsIndexInSpans() {
    kxc::Var x("x");
    kxc::Expr body = MakeRelayBinary("add", x, MakeRelayScalarFloat(1.0f));
    kxc::Function func({x}, body);
    const auto* before_body_node = static_cast<const kxc::ExprNode*>(func->body.get());
    TEST_CHECK(!before_body_node->span.defined(), "body span should be empty before pass");

    kxc::Function out = kxc::relay::CapturePostDfsIndexInSpansPass(func);
    const auto* after_body_node = static_cast<const kxc::ExprNode*>(out->body.get());
    TEST_CHECK(after_body_node->span.defined(), "body span should be populated");
    const auto* span_node = static_cast<const kxc::SpanNode*>(after_body_node->span.get());
    TEST_CHECK(span_node != nullptr, "span node should exist");
    TEST_CHECK(span_node->source_name == "pass.capture_post_dfs", "span source tag mismatch");
    TEST_CHECK(span_node->line >= 0, "post-dfs index should be non-negative");
    TEST_CHECK(span_node->column >= 0, "dominator index should be non-negative");
    return true;
}

// 验证 Relay 值获得预期的设备内存 scope 注解。
bool TestRelayAnnotateMemoryScope() {
    kxc::VirtualDevice empty_scope_vd(kxc::BuildTarget(kxc::Device::CPU()), "", 0);
    kxc::Var x("x");
    x.set_virtual_device(empty_scope_vd);
    kxc::Constant c = MakeRelayScalarFloat(1.0f);
    c.set_virtual_device(empty_scope_vd);
    kxc::Tuple body({x, c});
    body.set_virtual_device(empty_scope_vd);
    kxc::Function func({x}, body);

    kxc::Function out = kxc::relay::AnnotateMemoryScopePass(func);
    const auto* out_tuple = out->body.As<kxc::TupleNode>();
    TEST_CHECK(out_tuple != nullptr, "Body should remain tuple");
    const auto* x_node = dynamic_cast<const kxc::RelayNode*>(out_tuple->fields[0].get());
    const auto* c_node = dynamic_cast<const kxc::RelayNode*>(out_tuple->fields[1].get());
    TEST_CHECK(x_node != nullptr && x_node->virtual_device_.defined(), "x should keep vd");
    TEST_CHECK(c_node != nullptr && c_node->virtual_device_.defined(), "const should keep vd");
    TEST_CHECK(x_node->virtual_device_->memory_scope == "global", "x scope should be global");
    TEST_CHECK(c_node->virtual_device_->memory_scope == "const", "constant scope should be const");

    kxc::VirtualDevice preset_scope_vd(kxc::BuildTarget(kxc::Device::CPU()), "shared", 0);
    kxc::Var y("y");
    y.set_virtual_device(preset_scope_vd);
    kxc::Function func2({y}, y);
    kxc::Function out2 = kxc::relay::AnnotateMemoryScopePass(func2);
    const auto* y_node = dynamic_cast<const kxc::RelayNode*>(out2->body.get());
    TEST_CHECK(y_node != nullptr && y_node->virtual_device_.defined(), "y should keep vd");
    TEST_CHECK(y_node->virtual_device_->memory_scope == "shared",
               "existing memory scope should remain unchanged");
    return true;
}

// 验证默认 Relay PassPipeline 的组合顺序和最终不变量。
bool TestRelayPipeline() {
    kxc::Var x("x", kxc::TensorType({8}, "float32"));
    kxc::Var a("a");
    kxc::Var b("b");
    kxc::Expr add_zero = MakeRelayBinary("add", x, MakeRelayScalarFloat(0.0f));
    kxc::Expr cast_chain = kxc::Call(
        kxc::relay::Op::Get("cast"),
        {kxc::Call(kxc::relay::Op::Get("cast"), {add_zero}, kxc::relay::CastAttrs::Create(1))},
        kxc::relay::CastAttrs::Create(1));
    kxc::Expr reshape_chain = kxc::Call(
        kxc::relay::Op::Get("reshape"),
        {kxc::Call(kxc::relay::Op::Get("reshape"), {cast_chain},
                   kxc::relay::ReshapeAttrs::Create({8}))},
        kxc::relay::ReshapeAttrs::Create({8}));
    kxc::Expr shared = MakeRelayBinary("add", x, MakeRelayScalarFloat(1.0f));
    kxc::Expr cse_chain = kxc::Let(a, shared, kxc::Let(b, shared, MakeRelayBinary("add", a, b)));
    kxc::Function func(
        {x}, kxc::Tuple({reshape_chain, cse_chain, MakeRelayBinary("add", MakeRelayScalarInt64(2),
                                                                    MakeRelayScalarInt64(3))}));

    kxc::Array<kxc::String> order = {
        kxc::String("fold_tuple_get_item"),        kxc::String("fold_constant"),
        kxc::String("simplify_expr"),              kxc::String("canonicalize_cast"),
        kxc::String("remove_standalone_reshapes"), kxc::String("eliminate_common_subexpr"),
        kxc::String("eliminate_dead_let"),         kxc::String("annotate_memory_scope"),
        kxc::String("capture_post_dfs_index_in_spans"), kxc::String("infer_type"),
    };
    kxc::Function by_pipeline = kxc::relay::RunRelayPassPipeline(func, order);
    kxc::Function manual = kxc::relay::InferTypePass(kxc::relay::CapturePostDfsIndexInSpansPass(
        kxc::relay::AnnotateMemoryScopePass(kxc::relay::EliminateDeadLetPass(
            kxc::relay::EliminateCommonSubexprPass(kxc::relay::RemoveStandaloneReshapesPass(
                kxc::relay::CanonicalizeCastPass(kxc::relay::SimplifyExprPass(
                    kxc::relay::FoldConstantPass(kxc::relay::FoldTupleGetItemPass(func))))))))));
    TEST_CHECK(RelayText(by_pipeline) == RelayText(manual), "Relay pipeline order mismatch");

    bool thrown = false;
    try {
        kxc::relay::RunRelayPassPipeline(func, {kxc::String("unknown_pass")});
    } catch (const std::exception&) {
        thrown = true;
    }
    TEST_CHECK(thrown, "Unknown relay pass should throw");

    kxc::Function once = kxc::relay::RunRelayPassPipeline(func, {kxc::String("optimize_default")});
    kxc::Function twice =
        kxc::relay::RunRelayPassPipeline(once, {kxc::String("optimize_default")});
    TEST_CHECK(RelayText(once) == RelayText(twice), "Relay optimize_default should be idempotent");

    for (const auto& spec : kxc::relay::RelayRegisteredPassSpecs()) {
        if (!spec.idempotent) continue;
        kxc::Function pass_once =
            kxc::relay::RunRelayPassPipeline(func, {spec.name});
        kxc::Function pass_twice =
            kxc::relay::RunRelayPassPipeline(pass_once, {spec.name});
        TEST_CHECK(RelayText(pass_once) == RelayText(pass_twice),
                   "Relay pass declares idempotence but changes on its second run: " +
                       static_cast<std::string>(spec.name));
    }

    // 默认链不能启用结构键不完整的 CSE，否则不同常量会被错误视为同一表达式。
    kxc::Var first("first");
    kxc::Var second("second");
    kxc::Expr add_one = MakeRelayBinary("add", x, MakeRelayScalarFloat(1.0f));
    kxc::Expr add_two = MakeRelayBinary("add", x, MakeRelayScalarFloat(2.0f));
    kxc::Function distinct_constants(
        {x}, kxc::Let(first, add_one,
                      kxc::Let(second, add_two, kxc::Tuple({first, second}))));
    kxc::Function safe_default = kxc::relay::RunRelayPassPipeline(
        distinct_constants, {kxc::String("optimize_default")});
    TEST_CHECK(CountLetNodes(safe_default->body) == 2,
               "Relay optimize_default must not merge expressions with different constants");
    return true;
}

bool TestPassSpecValidation() {
    kxc::PassSpec valid = MakeTestSpec("unit_test_valid", kxc::IRDialect::kRelay,
                                       kxc::PassScope::kGraph, "relay_optimize",
                                       "kxc.test.pass.valid");
    kxc::ValidatePassSpec(valid);
    kxc::ValidatePassSpecForPipeline(valid, kxc::IRDialect::kRelay, kxc::PassScope::kGraph,
                                     "relay_optimize");

    bool duplicate_thrown = false;
    try {
        kxc::ValidatePassSpecs({valid, valid});
    } catch (const std::exception&) {
        duplicate_thrown = true;
    }
    TEST_CHECK(duplicate_thrown, "duplicate pass identity should fail validation");

    kxc::PassSpec missing_key = valid;
    missing_key.name = kxc::String("unit_test_missing_key");
    missing_key.implementation_key = kxc::String("");
    bool missing_key_thrown = false;
    try {
        kxc::ValidatePassSpec(missing_key);
    } catch (const std::exception&) {
        missing_key_thrown = true;
    }
    TEST_CHECK(missing_key_thrown, "missing implementation key should fail validation");

    kxc::PassSpec bad_scope = MakeTestSpec("unit_test_bad_scope", kxc::IRDialect::kRelay,
                                          kxc::PassScope::kPrimFunc, "relay_optimize",
                                          "kxc.test.pass.bad_scope");
    bool bad_scope_thrown = false;
    try {
        kxc::ValidatePassSpec(bad_scope);
    } catch (const std::exception&) {
        bad_scope_thrown = true;
    }
    TEST_CHECK(bad_scope_thrown, "Relay prim_func scope should fail validation");

    kxc::PassSpec tir_spec = MakeTestSpec("unit_test_tir", kxc::IRDialect::kTIR,
                                         kxc::PassScope::kPrimFunc, "tir_optimize",
                                         "kxc.test.pass.tir");
    bool dialect_thrown = false;
    try {
        kxc::ValidatePassSpecForPipeline(tir_spec, kxc::IRDialect::kRelay,
                                         kxc::PassScope::kGraph, "relay_optimize");
    } catch (const std::exception&) {
        dialect_thrown = true;
    }
    TEST_CHECK(dialect_thrown, "pipeline dialect mismatch should fail validation");

    kxc::PassSpec targeted = valid;
    targeted.name = kxc::String("unit_test_targeted");
    targeted.target_requirements = {
        kxc::String("kind=llvm"), kxc::String("attr.exists>0"),
        kxc::String("attr.max_threads_per_block>0"),
        kxc::String("attr.max_shared_memory_per_block>=0")};
    kxc::ValidatePassSpec(targeted);
    TEST_CHECK(kxc::PassSpecSupportsTarget(
                   targeted, kxc::BuildTarget(kxc::Device::CPU())),
               "validated target predicates should match a CPU capability snapshot");

    kxc::PassSpec unsupported_requirement = targeted;
    unsupported_requirement.name = kxc::String("unit_test_bad_target_requirement");
    unsupported_requirement.target_requirements = {
        kxc::String("attr.unknown>0")};
    bool unsupported_requirement_thrown = false;
    try {
        kxc::ValidatePassSpec(unsupported_requirement);
    } catch (const std::exception&) {
        unsupported_requirement_thrown = true;
    }
    TEST_CHECK(unsupported_requirement_thrown,
               "unknown target predicates should fail validation");

    kxc::PassSpec contradictory_analysis = valid;
    contradictory_analysis.name = kxc::String("unit_test_analysis_conflict");
    contradictory_analysis.preserved_analyses = {kxc::String("shape")};
    contradictory_analysis.invalidated_analyses = {kxc::String("shape")};
    bool contradictory_analysis_thrown = false;
    try {
        kxc::ValidatePassSpec(contradictory_analysis);
    } catch (const std::exception&) {
        contradictory_analysis_thrown = true;
    }
    TEST_CHECK(contradictory_analysis_thrown,
               "one analysis cannot be both preserved and invalidated");
    return true;
}

bool TestPassSpecPipelineMetadata() {
    const std::vector<std::string> relay_default = ToStdVector(kxc::relay::RelayDefaultPassOrder());
    const std::vector<std::string> expected_relay = {
        "fold_tuple_get_item", "fold_constant", "simplify_expr", "canonicalize_cast",
        "remove_standalone_reshapes", "eliminate_dead_let", "annotate_memory_scope",
        "capture_post_dfs_index_in_spans", "infer_type"};
    TEST_CHECK(relay_default == expected_relay, "Relay default pass order changed");

    bool saw_cse = false;
    for (const auto& spec : kxc::relay::RelayRegisteredPassSpecs()) {
        kxc::ValidatePassSpecForPipeline(spec, kxc::IRDialect::kRelay, kxc::PassScope::kGraph,
                                         "relay_optimize");
        const std::string name = static_cast<std::string>(spec.name);
        const std::string key = static_cast<std::string>(spec.implementation_key);
        TEST_CHECK(key == "kxc.relay.transform." + name,
                   "Relay implementation key should match FFI transform name");
        if (name == "eliminate_common_subexpr") saw_cse = true;
    }
    TEST_CHECK(saw_cse, "explicit-only Relay CSE pass should still have a PassSpec");

    const std::vector<std::string> tir_default = ToStdVector(kxc::tir::TIRDefaultPassOrder());
    const std::vector<std::string> expected_tir = {
        "fold_constant", "simplify_expr", "force_narrow_index_to_i32",
        "convert_for_loops_serial", "loop_partition", "unroll_loop", "vectorize_loop",
        "remove_no_op"};
    TEST_CHECK(tir_default == expected_tir, "TIR default pass order changed");

    bool saw_bind_cuda = false;
    for (const auto& spec : kxc::tir::TIRRegisteredPassSpecs()) {
        kxc::ValidatePassSpecForPipeline(
            spec, kxc::IRDialect::kTIR, kxc::PassScope::kPrimFunc,
            static_cast<std::string>(spec.phase));
        const std::string name = static_cast<std::string>(spec.name);
        const std::string key = static_cast<std::string>(spec.implementation_key);
        TEST_CHECK(key == "kxc.tir.transform." + name,
                   "TIR implementation key should match FFI transform name");
        if (name == "bind_cuda_threads") {
            saw_bind_cuda = true;
            TEST_CHECK(!spec.target_requirements.empty() &&
                           spec.produced_invariants.size() == 1 &&
                           spec.produced_invariants[0] ==
                               kxc::String("prim_func_defined"),
                       "bind_cuda_threads should declare target policy and proof");
        }
    }
    TEST_CHECK(saw_bind_cuda, "CUDA binding pass should still have a PassSpec");
    return true;
}

// 验证 TIR 算术表达式简化。
bool TestTIRSimplifyExpr() {
    kxc::tir::Var x("x", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt body =
        kxc::tir::Evaluate(kxc::tir::Add(x, kxc::tir::IntImm(0, kxc::tir::DataType::Int(32))));
    kxc::tir::PrimFunc func({}, body);

    kxc::tir::PrimFunc out = kxc::tir::SimplifyExprPass(func);
    const auto* eval = out->body.As<kxc::tir::EvaluateNode>();
    TEST_CHECK(eval != nullptr, "Body should still be Evaluate");
    TEST_CHECK(eval->value.As<kxc::tir::VarNode>() != nullptr, "x + 0 should simplify to x");
    return true;
}

// 验证 TIR 常量表达式折叠。
bool TestTIRFoldConstant() {
    kxc::tir::Stmt body = kxc::tir::SeqStmt({
        kxc::tir::Evaluate(
            kxc::tir::Add(kxc::tir::IntImm(2, kxc::tir::DataType::Int(32)),
                          kxc::tir::IntImm(3, kxc::tir::DataType::Int(32)))),
        kxc::tir::Evaluate(
            kxc::tir::Div(kxc::tir::IntImm(1, kxc::tir::DataType::Int(32)),
                          kxc::tir::IntImm(0, kxc::tir::DataType::Int(32)))),
    });
    kxc::tir::PrimFunc func({}, body);

    kxc::tir::PrimFunc out = kxc::tir::FoldConstantPass(func);
    const auto* seq = out->body.As<kxc::tir::SeqStmtNode>();
    TEST_CHECK(seq != nullptr, "Body should remain SeqStmt");
    TEST_CHECK(seq->seq.size() == 2, "Body should keep two statements");

    const auto* eval0 = seq->seq[0].As<kxc::tir::EvaluateNode>();
    TEST_CHECK(eval0 != nullptr, "First statement should be Evaluate");
    const auto* first_value = eval0->value.As<kxc::tir::IntImmNode>();
    TEST_CHECK(first_value != nullptr && first_value->value == 5, "2 + 3 should fold to 5");

    const auto* eval1 = seq->seq[1].As<kxc::tir::EvaluateNode>();
    TEST_CHECK(eval1 != nullptr, "Second statement should be Evaluate");
    TEST_CHECK(eval1->value.As<kxc::tir::DivNode>() != nullptr,
               "Division by zero should not fold");
    return true;
}

// 验证可安全表示的索引被收窄为 int32。
bool TestTIRForceNarrowIndexToI32() {
    kxc::tir::Var i("i", kxc::tir::DataType::Int(64));
    kxc::tir::Var buf("buf", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt body = kxc::tir::For(
        i, kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)),
        kxc::tir::IntImm(4, kxc::tir::DataType::Int(64)), kxc::tir::ForType::Serial,
        kxc::tir::Store(buf, kxc::tir::IntImm(1), kxc::tir::IntImm(3, kxc::tir::DataType::Int(64))));

    kxc::Array<kxc::tir::PrimExpr> shape = {kxc::tir::IntImm(16, kxc::tir::DataType::Int(64))};
    kxc::Array<kxc::tir::PrimExpr> strides = {kxc::tir::IntImm(1, kxc::tir::DataType::Int(64))};
    kxc::tir::Buffer buffer(buf, kxc::tir::DataType::Int(32), shape, strides,
                            kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)), "buf", 0, 0);
    kxc::Map<kxc::tir::Var, kxc::tir::Buffer> buffer_map;
    buffer_map.Set(buf, buffer);
    kxc::tir::PrimFunc func({buf}, body, buffer_map);

    kxc::tir::PrimFunc out = kxc::tir::ForceNarrowIndexToI32Pass(func);
    const auto* for_node = out->body.As<kxc::tir::ForNode>();
    TEST_CHECK(for_node != nullptr, "Body should remain For");
    const auto* min_imm = for_node->min.As<kxc::tir::IntImmNode>();
    const auto* extent_imm = for_node->extent.As<kxc::tir::IntImmNode>();
    TEST_CHECK(min_imm != nullptr && min_imm->dtype.bits == 32, "For min should narrow to int32");
    TEST_CHECK(extent_imm != nullptr && extent_imm->dtype.bits == 32,
               "For extent should narrow to int32");

    const auto* store = for_node->body.As<kxc::tir::StoreNode>();
    TEST_CHECK(store != nullptr, "Loop body should remain Store");
    const auto* idx = store->index.As<kxc::tir::IntImmNode>();
    TEST_CHECK(idx != nullptr && idx->dtype.bits == 32, "Store index should narrow to int32");

    const kxc::tir::Buffer out_buf = out->buffer_map.at(buf);
    const auto* shape0 = out_buf->shape[0].As<kxc::tir::IntImmNode>();
    TEST_CHECK(shape0 != nullptr && shape0->dtype.bits == 32, "Buffer shape should narrow to int32");

    kxc::tir::PrimFunc big_func(
        {}, kxc::tir::For(i, kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)),
                          kxc::tir::IntImm((1LL << 40), kxc::tir::DataType::Int(64)),
                          kxc::tir::ForType::Serial, kxc::tir::Evaluate(kxc::tir::IntImm(1))));
    kxc::tir::PrimFunc big_out = kxc::tir::ForceNarrowIndexToI32Pass(big_func);
    const auto* big_for = big_out->body.As<kxc::tir::ForNode>();
    const auto* big_extent = big_for->extent.As<kxc::tir::IntImmNode>();
    TEST_CHECK(big_extent != nullptr && big_extent->dtype.bits == 64,
               "Out-of-range int64 extent must not narrow");
    return true;
}

// 验证循环类型可统一转换为串行循环。
bool TestTIRConvertForLoopsSerial() {
    kxc::tir::Var i("i");
    kxc::tir::Var j("j");
    kxc::tir::Stmt loop = kxc::tir::For(
        i, kxc::tir::IntImm(0), kxc::tir::IntImm(8), kxc::tir::ForType::Parallel,
        kxc::tir::For(j, kxc::tir::IntImm(0), kxc::tir::IntImm(4), kxc::tir::ForType::Vectorized,
                      kxc::tir::Evaluate(kxc::tir::IntImm(1))));
    kxc::tir::PrimFunc func({}, loop);

    kxc::tir::PrimFunc out = kxc::tir::ConvertForLoopsSerialPass(func);
    std::vector<kxc::tir::ForType> for_types;
    CollectForTypes(out->body, &for_types);
    TEST_CHECK(for_types.size() == 2, "Two loops should remain");
    TEST_CHECK(for_types[0] == kxc::tir::ForType::Serial, "Outer loop should be serial");
    TEST_CHECK(for_types[1] == kxc::tir::ForType::Serial, "Inner loop should be serial");
    return true;
}

// 验证循环分区生成正确的边界分支。
bool TestTIRLoopPartition() {
    kxc::tir::Var i("i", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt loop =
        kxc::tir::For(i, kxc::tir::IntImm(0), kxc::tir::IntImm(10), kxc::tir::ForType::Serial,
                      kxc::tir::Evaluate(i));
    kxc::tir::PrimFunc func({}, loop);

    kxc::tir::PrimFunc out = kxc::tir::LoopPartitionPass(func);
    const auto* seq = out->body.As<kxc::tir::SeqStmtNode>();
    TEST_CHECK(seq != nullptr, "Partition should produce SeqStmt");
    TEST_CHECK(seq->seq.size() == 2, "Partition should create main + tail loops");
    const auto* main_outer = seq->seq[0].As<kxc::tir::ForNode>();
    TEST_CHECK(main_outer != nullptr, "Main part should be outer for loop");
    const auto* main_extent = main_outer->extent.As<kxc::tir::IntImmNode>();
    TEST_CHECK(main_extent != nullptr && main_extent->value == 2, "Main outer extent should be 2");
    const auto* tail_loop = seq->seq[1].As<kxc::tir::ForNode>();
    TEST_CHECK(tail_loop != nullptr, "Tail part should be for loop");
    const auto* tail_extent = tail_loop->extent.As<kxc::tir::IntImmNode>();
    TEST_CHECK(tail_extent != nullptr && tail_extent->value == 2, "Tail extent should be 2");
    return true;
}

// 验证固定次数循环按策略展开。
bool TestTIRUnrollLoop() {
    kxc::tir::Var i("i", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt loop =
        kxc::tir::For(i, kxc::tir::IntImm(0), kxc::tir::IntImm(3), kxc::tir::ForType::Serial,
                      kxc::tir::Evaluate(i));
    kxc::tir::PrimFunc func({}, loop);

    kxc::tir::PrimFunc out = kxc::tir::UnrollLoopPass(func);
    const auto* seq = out->body.As<kxc::tir::SeqStmtNode>();
    TEST_CHECK(seq != nullptr, "Unrolled loop should become SeqStmt");
    TEST_CHECK(seq->seq.size() == 3, "Extent 3 should produce 3 statements");
    std::vector<kxc::tir::ForType> for_types;
    CollectForTypes(out->body, &for_types);
    TEST_CHECK(for_types.empty(), "Unrolled loop should contain no For");
    return true;
}

// 验证可向量化循环标记为向量循环。
bool TestTIRVectorizeLoop() {
    kxc::tir::Var i("i", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt vec_loop =
        kxc::tir::For(i, kxc::tir::IntImm(0), kxc::tir::IntImm(8), kxc::tir::ForType::Serial,
                      kxc::tir::Evaluate(kxc::tir::IntImm(1)));
    kxc::tir::PrimFunc vec_func({}, vec_loop);
    kxc::tir::PrimFunc vec_out = kxc::tir::VectorizeLoopPass(vec_func);
    const auto* vec_for = vec_out->body.As<kxc::tir::ForNode>();
    TEST_CHECK(vec_for != nullptr, "Body should remain for");
    TEST_CHECK(vec_for->for_type == kxc::tir::ForType::Vectorized,
               "Extent divisible by 4 should vectorize");

    kxc::tir::Stmt keep_loop =
        kxc::tir::For(i, kxc::tir::IntImm(0), kxc::tir::IntImm(6), kxc::tir::ForType::Serial,
                      kxc::tir::Evaluate(kxc::tir::IntImm(1)));
    kxc::tir::PrimFunc keep_func({}, keep_loop);
    kxc::tir::PrimFunc keep_out = kxc::tir::VectorizeLoopPass(keep_func);
    const auto* keep_for = keep_out->body.As<kxc::tir::ForNode>();
    TEST_CHECK(keep_for != nullptr, "Body should remain for");
    TEST_CHECK(keep_for->for_type == kxc::tir::ForType::Serial,
               "Extent not divisible by 4 should not vectorize");
    return true;
}

// 验证无副作用空语句从 TIR 中删除。
bool TestTIRRemoveNoOp() {
    kxc::tir::Stmt seq = kxc::tir::SeqStmt({
        kxc::tir::Evaluate(kxc::tir::IntImm(0)),
        kxc::tir::SeqStmt({
            kxc::tir::Evaluate(kxc::tir::IntImm(0)),
            kxc::tir::Evaluate(kxc::tir::IntImm(1)),
        }),
        kxc::tir::Evaluate(kxc::tir::IntImm(0)),
    });
    kxc::tir::PrimFunc func({}, seq);

    kxc::tir::PrimFunc out = kxc::tir::RemoveNoOpPass(func);
    const auto* eval = out->body.As<kxc::tir::EvaluateNode>();
    TEST_CHECK(eval != nullptr, "No-op cleanup should collapse to single Evaluate");
    const auto* int_imm = eval->value.As<kxc::tir::IntImmNode>();
    TEST_CHECK(int_imm != nullptr && int_imm->value == 1, "Remaining statement should be Evaluate(1)");
    return true;
}

// 验证默认 TIR PassPipeline 的组合顺序和最终不变量。
bool TestTIRPipeline() {
    kxc::tir::Var i("i", kxc::tir::DataType::Int(64));
    kxc::tir::Var x("x", kxc::tir::DataType::Int(32));
    kxc::tir::Stmt body = kxc::tir::For(
        i, kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)),
        kxc::tir::IntImm(10, kxc::tir::DataType::Int(64)), kxc::tir::ForType::Parallel,
        kxc::tir::SeqStmt({
            kxc::tir::Evaluate(kxc::tir::IntImm(0)),
            kxc::tir::Evaluate(kxc::tir::Add(x, kxc::tir::IntImm(0, kxc::tir::DataType::Int(32)))),
        }));
    kxc::tir::PrimFunc func({x}, body);

    kxc::Array<kxc::String> order = {
        kxc::String("fold_constant"),            kxc::String("simplify_expr"),
        kxc::String("force_narrow_index_to_i32"), kxc::String("convert_for_loops_serial"),
        kxc::String("loop_partition"),            kxc::String("unroll_loop"),
        kxc::String("vectorize_loop"),            kxc::String("remove_no_op"),
    };
    kxc::tir::PrimFunc by_pipeline = kxc::tir::RunTIRPassPipeline(func, order);
    kxc::tir::PrimFunc manual = kxc::tir::RemoveNoOpPass(kxc::tir::VectorizeLoopPass(
        kxc::tir::UnrollLoopPass(kxc::tir::LoopPartitionPass(
            kxc::tir::ConvertForLoopsSerialPass(
                kxc::tir::ForceNarrowIndexToI32Pass(
                    kxc::tir::SimplifyExprPass(kxc::tir::FoldConstantPass(func))))))));
    TEST_CHECK(TIRText(by_pipeline) == TIRText(manual), "TIR pipeline order mismatch");

    bool thrown = false;
    try {
        kxc::tir::RunTIRPassPipeline(func, {kxc::String("unknown_pass")});
    } catch (const std::exception&) {
        thrown = true;
    }
    TEST_CHECK(thrown, "Unknown tir pass should throw");

    kxc::tir::PrimFunc once =
        kxc::tir::RunTIRPassPipeline(func, {kxc::String("optimize_default")});
    kxc::tir::PrimFunc twice =
        kxc::tir::RunTIRPassPipeline(once, {kxc::String("optimize_default")});
    TEST_CHECK(TIRText(once) == TIRText(twice), "TIR optimize_default should be idempotent");
    for (const auto& spec : kxc::tir::TIRRegisteredPassSpecs()) {
        if (!spec.idempotent) continue;
        kxc::tir::PrimFunc pass_once =
            kxc::tir::RunTIRPassPipeline(func, {spec.name});
        kxc::tir::PrimFunc pass_twice =
            kxc::tir::RunTIRPassPipeline(pass_once, {spec.name});
        TEST_CHECK(TIRText(pass_once) == TIRText(pass_twice),
                   "TIR pass declares idempotence but changes on its second run: " +
                       static_cast<std::string>(spec.name));
    }
    return true;
}

}  // namespace

// 顺序运行 Relay/TIR pass 契约矩阵并汇总失败。
int main() {
    const std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"relay_fold_tuple_get_item", TestRelayFoldTupleGetItem},
        {"relay_fold_constant", TestRelayFoldConstant},
        {"relay_simplify_expr", TestRelaySimplifyExpr},
        {"relay_canonicalize_cast", TestRelayCanonicalizeCast},
        {"relay_remove_standalone_reshapes", TestRelayRemoveStandaloneReshapes},
        {"relay_eliminate_common_subexpr", TestRelayEliminateCommonSubexpr},
        {"relay_eliminate_dead_let_and_virtual_device", TestRelayEliminateDeadLetAndVirtualDevice},
        {"relay_capture_post_dfs_index_in_spans", TestRelayCapturePostDfsIndexInSpans},
        {"relay_annotate_memory_scope", TestRelayAnnotateMemoryScope},
        {"relay_pipeline", TestRelayPipeline},
        {"pass_spec_validation", TestPassSpecValidation},
        {"pass_spec_pipeline_metadata", TestPassSpecPipelineMetadata},
        {"tir_simplify_expr", TestTIRSimplifyExpr},
        {"tir_fold_constant", TestTIRFoldConstant},
        {"tir_force_narrow_index_to_i32", TestTIRForceNarrowIndexToI32},
        {"tir_convert_for_loops_serial", TestTIRConvertForLoopsSerial},
        {"tir_loop_partition", TestTIRLoopPartition},
        {"tir_unroll_loop", TestTIRUnrollLoop},
        {"tir_vectorize_loop", TestTIRVectorizeLoop},
        {"tir_remove_no_op", TestTIRRemoveNoOp},
        {"tir_pipeline", TestTIRPipeline},
    };

    for (const auto& test : tests) {
        bool ok = false;
        try {
            ok = test.second();
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << test.first << ": unexpected exception: " << e.what() << "\n";
            return 1;
        }
        if (!ok) {
            return 1;
        }
        std::cout << "[PASS] " << test.first << "\n";
    }

    std::cout << "All pass pipeline tests passed.\n";
    return 0;
}
