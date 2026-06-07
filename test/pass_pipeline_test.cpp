/*! \file test/pass_pipeline_test.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "base/target.h"
#include "relay/op.h"
#include "relay/pass/print_ir.h"
#include "relay/transforms/annotate_memory_scope.h"
#include "relay/transforms/canonicalize_cast.h"
#include "relay/transforms/capture_post_dfs_index_in_spans.h"
#include "relay/transforms/eliminate_common_subexpr.h"
#include "relay/transforms/eliminate_dead_let.h"
#include "relay/transforms/fold_constant.h"
#include "relay/transforms/fold_tuple_get_item.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/pipeline.h"
#include "relay/transforms/remove_standalone_reshapes.h"
#include "relay/transforms/simplify_expr.h"
#include "tir/pass/print_ir.h"
#include "tir/transforms/convert_for_loops_serial.h"
#include "tir/transforms/fold_constant.h"
#include "tir/transforms/force_narrow_index_to_i32.h"
#include "tir/transforms/loop_partition.h"
#include "tir/transforms/pipeline.h"
#include "tir/transforms/remove_no_op.h"
#include "tir/transforms/simplify_expr.h"
#include "tir/transforms/unroll_loop.h"
#include "tir/transforms/vectorize_loop.h"

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

kxc::runtime::NDArray MakeScalarFloat(float value) {
    kxc::runtime::NDArray array(kxc::Array<int64_t>{}, "float32");
    float* data = static_cast<float*>(array->dl_tensor.data);
    *data = value;
    return array;
}

kxc::runtime::NDArray MakeScalarInt64(int64_t value) {
    kxc::runtime::NDArray array(kxc::Array<int64_t>{}, "int64");
    int64_t* data = static_cast<int64_t*>(array->dl_tensor.data);
    *data = value;
    return array;
}

kxc::runtime::NDArray MakeScalarInt32(int32_t value) {
    kxc::runtime::NDArray array(kxc::Array<int64_t>{}, "int32");
    int32_t* data = static_cast<int32_t*>(array->dl_tensor.data);
    *data = value;
    return array;
}

kxc::Constant MakeRelayScalarFloat(float value) { return kxc::Constant(MakeScalarFloat(value)); }
kxc::Constant MakeRelayScalarInt64(int64_t value) { return kxc::Constant(MakeScalarInt64(value)); }
kxc::Constant MakeRelayScalarInt32(int32_t value) { return kxc::Constant(MakeScalarInt32(value)); }

kxc::Call MakeRelayBinary(const std::string& op_name, const kxc::Expr& lhs,
                          const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get(op_name), {lhs, rhs});
}

std::string RelayText(const kxc::Function& func) { return kxc::relay::pass::ToText(func); }

std::string TIRText(const kxc::tir::PrimFunc& func) {
    std::ostringstream os;
    kxc::tir::pass::DumpPrimFunc(func, os);
    return os.str();
}

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

bool TestRelayEliminateDeadLetAndVirtualDevice() {
    kxc::Var x("x");
    kxc::Var tmp("tmp");

    kxc::Let dead_let(tmp, MakeRelayScalarFloat(1.0f), x);
    kxc::VirtualDevice vd(kxc::BuildTarget(kxc::kCPU, 0), "global", 0);
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

bool TestRelayAnnotateMemoryScope() {
    kxc::VirtualDevice empty_scope_vd(kxc::BuildTarget(kxc::kCPU, 0), "", 0);
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

    kxc::VirtualDevice preset_scope_vd(kxc::BuildTarget(kxc::kCPU, 0), "shared", 0);
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
    return true;
}

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
    return true;
}

}  // namespace

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
