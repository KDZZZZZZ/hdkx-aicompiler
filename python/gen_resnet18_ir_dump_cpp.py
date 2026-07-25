"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import argparse
import re
from pathlib import Path

import onnx
from onnx import AttributeProto

from kxc_onnx import import_onnx


SUPPORTED_OPS = {
    "Conv",
    "Relu",
    "MaxPool",
    "Add",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
}


def sanitize(name: str) -> str:
    """将 ONNX 名称规范化为合法且稳定的 C++ 标识符。"""
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not name:
        name = "v"
    if name[0].isdigit():
        name = "v_" + name
    return name


def fmt_int_list(v):
    """将整数序列格式化为 C++ 初始化列表。"""
    return "{" + ", ".join(str(int(x)) for x in v) + "}"


def get_attr(node, name, default):
    """读取生成器支持的 ONNX 属性类型，缺失时返回默认值。"""
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == AttributeProto.INT:
            return int(a.i)
        if a.type == AttributeProto.FLOAT:
            return float(a.f)
        if a.type == AttributeProto.INTS:
            return [int(x) for x in a.ints]
    return default


def emit_cpp(model_path: Path, out_path: Path, default_batch: int | None):
    """导入 ResNet18 并生成可输出 Relay/TIR 的 C++ 诊断程序。"""
    imported = import_onnx(model_path, default_batch=default_batch)

    lines = []
    w = lines.append

    w('#include "kxc/relay/relay.h"')
    w('#include "kxc/relay/op.h"')
    w('#include "kxc/relay/transforms/infer_type.h"')
    w('#include "compiler/internal/lowered_graph.h"')
    w('#include "kxc/tir/stmt.h"')
    w('#include "kxc/tir/expr.h"')
    w("")
    w("#include <fstream>")
    w("#include <iostream>")
    w("#include <sstream>")
    w("#include <stdexcept>")
    w("#include <string>")
    w("#include <unordered_map>")
    w("#include <unordered_set>")
    w("#include <vector>")
    w("")
    w("using namespace kxc;")
    w("using namespace kxc::relay;")
    w("")
    w("namespace {")
    w("")
    w("// 生成 IR 树形文本所需的缩进。")
    w("std::string Indent(int n) { return std::string(n, ' '); }")
    w("")
    w("// 将 TIR dtype 格式化为可读名称。")
    w("std::string DTypeToString(const tir::DataType& dt) {")
    w("    if (dt.code == 2) return \"float\" + std::to_string(dt.bits);")
    w("    if (dt.code == 0) return \"int\" + std::to_string(dt.bits);")
    w("    if (dt.code == 1) return dt.bits == 1 ? \"bool\" : (\"uint\" + std::to_string(dt.bits));")
    w("    if (dt.code == 3) return \"handle\";")
    w("    return \"dtype(code=\" + std::to_string(dt.code) + \",bits=\" + std::to_string(dt.bits) + \")\";")
    w("}")
    w("")
    w("// 将 NDArray 的 DLPack dtype 格式化为可读名称。")
    w("std::string NDArrayDTypeToString(const runtime::NDArray& arr) {")
    w("    const DLDataType& dt = arr->dl_tensor.dtype;")
    w("    if (dt.code == kDLFloat) return \"float\" + std::to_string(dt.bits);")
    w("    if (dt.code == kDLInt) return \"int\" + std::to_string(dt.bits);")
    w("    if (dt.code == kDLUInt) return \"uint\" + std::to_string(dt.bits);")
    w("    if (dt.code == kDLBool) return \"bool\";")
    w("    return \"unknown\";")
    w("}")
    w("")
    w("// 递归格式化 Relay 类型及其 shape、dtype。")
    w("std::string PrintRelayType(const Type& ty) {")
    w("    if (!ty.defined()) return \"<none>\";")
    w("    if (const auto* t = ty.As<TensorTypeNode>()) {")
    w("        std::ostringstream os;")
    w("        os << \"TensorType(shape=[\";")
    w("        for (size_t i = 0; i < t->shape.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << t->shape[i];")
    w("        }")
    w("        os << \"], dtype=\" << t->dtype << \")\";")
    w("        return os.str();")
    w("    }")
    w("    return \"<Type>\";")
    w("}")
    w("")
    w("std::string PrintPrimExpr(const tir::PrimExpr& e);")
    w("")
    w("// 将 TIR 标量表达式格式化为诊断文本。")
    w("std::string PrintPrimExpr(const tir::PrimExpr& e) {")
    w("    if (!e.defined()) return \"<undef>\";")
    w("    if (const auto* n = e.As<tir::IntImmNode>()) return std::to_string(n->value);")
    w("    if (const auto* n = e.As<tir::FloatImmNode>()) {")
    w("        std::ostringstream os;")
    w("        os << n->value;")
    w("        return os.str();")
    w("    }")
    w("    if (const auto* n = e.As<tir::VarNode>()) return n->name_hint;")
    w("    if (const auto* n = e.As<tir::LoadNode>()) return n->buffer_var->name_hint + \"[\" + PrintPrimExpr(n->index) + \"]\";")
    w("    if (const auto* n = e.As<tir::AddNode>()) return \"(\" + PrintPrimExpr(n->a) + \" + \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::SubNode>()) return \"(\" + PrintPrimExpr(n->a) + \" - \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::MulNode>()) return \"(\" + PrintPrimExpr(n->a) + \" * \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::DivNode>()) return \"(\" + PrintPrimExpr(n->a) + \" / \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::ModNode>()) return \"(\" + PrintPrimExpr(n->a) + \" % \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::MinNode>()) return \"min(\" + PrintPrimExpr(n->a) + \", \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::MaxNode>()) return \"max(\" + PrintPrimExpr(n->a) + \", \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::EQNode>()) return \"(\" + PrintPrimExpr(n->a) + \" == \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::LTNode>()) return \"(\" + PrintPrimExpr(n->a) + \" < \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::AndNode>()) return \"(\" + PrintPrimExpr(n->a) + \" && \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::OrNode>()) return \"(\" + PrintPrimExpr(n->a) + \" || \" + PrintPrimExpr(n->b) + \")\";")
    w("    if (const auto* n = e.As<tir::NotNode>()) return \"(!\" + PrintPrimExpr(n->value) + \")\";")
    w("    if (const auto* n = e.As<tir::SelectNode>()) return \"select(\" + PrintPrimExpr(n->condition) + \", \" + PrintPrimExpr(n->true_value) + \", \" + PrintPrimExpr(n->false_value) + \")\";")
    w("    if (const auto* n = e.As<tir::CallNode>()) {")
    w("        std::ostringstream os;")
    w("        os << n->name << \"(\";")
    w("        for (size_t i = 0; i < n->args.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << PrintPrimExpr(n->args[i]);")
    w("        }")
    w("        os << \")\";")
    w("        return os.str();")
    w("    }")
    w("    return \"<expr>\";")
    w("}")
    w("")
    w("// 递归输出 TIR 语句树。")
    w("void DumpStmt(const tir::Stmt& s, std::ostream& os, int indent) {")
    w("    if (!s.defined()) { os << Indent(indent) << \"<empty-stmt>\\n\"; return; }")
    w("    if (const auto* n = s.As<tir::ForNode>()) {")
    w("        os << Indent(indent) << \"for (\" << n->loop_var->name_hint << \" = \" << PrintPrimExpr(n->min)")
    w("           << \"; \" << n->loop_var->name_hint << \" < (\" << PrintPrimExpr(n->min) << \" + \" << PrintPrimExpr(n->extent)")
    w("           << \"); \" << n->loop_var->name_hint << \"++) {\\n\";")
    w("        DumpStmt(n->body, os, indent + 2);")
    w("        os << Indent(indent) << \"}\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::StoreNode>()) {")
    w("        os << Indent(indent) << n->buffer_var->name_hint << \"[\" << PrintPrimExpr(n->index) << \"] = \" << PrintPrimExpr(n->value) << \";\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::AllocateNode>()) {")
    w("        os << Indent(indent) << \"allocate \" << n->buffer_var->name_hint << \" : \" << DTypeToString(n->dtype) << \" [\";")
    w("        for (size_t i = 0; i < n->extents.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << PrintPrimExpr(n->extents[i]);")
    w("        }")
    w("        os << \"] if (\" << PrintPrimExpr(n->condition) << \") {\\n\";")
    w("        DumpStmt(n->body, os, indent + 2);")
    w("        os << Indent(indent) << \"}\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::SeqStmtNode>()) {")
    w("        for (const auto& x : n->seq) DumpStmt(x, os, indent);")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::LetStmtNode>()) {")
    w("        os << Indent(indent) << \"let \" << n->var->name_hint << \" = \" << PrintPrimExpr(n->value) << \" in\\n\";")
    w("        DumpStmt(n->body, os, indent + 2);")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::IfThenElseNode>()) {")
    w("        os << Indent(indent) << \"if (\" << PrintPrimExpr(n->condition) << \") {\\n\";")
    w("        DumpStmt(n->then_case, os, indent + 2);")
    w("        if (n->else_case.defined()) {")
    w("            os << Indent(indent) << \"} else {\\n\";")
    w("            DumpStmt(n->else_case, os, indent + 2);")
    w("        }")
    w("        os << Indent(indent) << \"}\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* n = s.As<tir::EvaluateNode>()) {")
    w("        os << Indent(indent) << \"evaluate(\" << PrintPrimExpr(n->value) << \");\\n\";")
    w("        return;")
    w("    }")
    w("    os << Indent(indent) << \"<stmt>\\n\";")
    w("}")
    w("")
    w("// 输出 TIR 函数参数、buffer、attrs 和函数体。")
    w("void DumpPrimFunc(const tir::PrimFunc& f, std::ostream& os) {")
    w("    os << \"\\n================ TIR PrimFunc ================\\n\";")
    w("    os << \"params(\" << f->params.size() << \"):\\n\";")
    w("    for (size_t i = 0; i < f->params.size(); ++i) {")
    w("        const auto& p = f->params[i];")
    w("        os << \"  [\" << i << \"] \" << p->name_hint << \" : \" << DTypeToString(p->dtype) << \"\\n\";")
    w("    }")
    w("    os << \"buffer_map(\" << f->buffer_map.size() << \"):\\n\";")
    w("    for (const auto& kv : f->buffer_map) {")
    w("        const auto& v = kv.first;")
    w("        const auto& b = kv.second;")
    w("        os << \"  \" << v->name_hint << \" -> \" << b->name << \" shape=[\";")
    w("        for (size_t i = 0; i < b->shape.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << PrintPrimExpr(b->shape[i]);")
    w("        }")
    w("        os << \"] dtype=\" << DTypeToString(b->dtype) << \"\\n\";")
    w("    }")
    w("    os << \"attrs(\" << f->attrs.size() << \"):\\n\";")
    w("    for (const auto& kv : f->attrs) {")
    w("        os << \"  \" << std::string(kv.first) << \"\\n\";")
    w("    }")
    w("    os << \"body:\\n\";")
    w("    DumpStmt(f->body, os, 2);")
    w("}")
    w("")
    w("// 提取受支持算子 attrs 的关键字段用于诊断输出。")
    w("std::string DescribeCallAttrs(const ObjectRef& attrs) {")
    w("    if (!attrs.defined()) return \"<none>\";")
    w("    if (const auto* a = attrs.As<Conv2DAttrsNode>()) {")
    w("        std::ostringstream os;")
    w("        os << \"Conv2D(strides=\" << a->strides[0] << \",\" << a->strides[1]")
    w("           << \"; padding=\" << (a->padding.empty() ? 0 : a->padding[0])")
    w("           << \",\" << (a->padding.size() > 1 ? a->padding[1] : 0)")
    w("           << \"; groups=\" << a->groups << \")\";")
    w("        return os.str();")
    w("    }")
    w("    if (const auto* a = attrs.As<MaxPool2DAttrsNode>()) {")
    w("        std::ostringstream os;")
    w("        os << \"Pool2D(kernel=\" << a->pool_size[0] << \",\" << a->pool_size[1]")
    w("           << \"; strides=\" << a->strides[0] << \",\" << a->strides[1] << \")\";")
    w("        return os.str();")
    w("    }")
    w("    if (const auto* a = attrs.As<FlattenAttrsNode>()) {")
    w("        return std::string(\"Flatten(axis=\") + std::to_string(a->axis) + \")\";")
    w("    }")
    w("    if (const auto* a = attrs.As<GemmAttrsNode>()) {")
    w("        std::ostringstream os;")
    w("        os << \"Gemm(alpha=\" << a->alpha << \", beta=\" << a->beta << \", transA=\" << a->transA << \", transB=\" << a->transB << \")\";")
    w("        return os.str();")
    w("    }")
    w("    if (attrs.As<AddAttrsNode>()) return \"Add\";")
    w("    if (attrs.As<ReluAttrsNode>()) return \"Relu\";")
    w("    if (attrs.As<GlobalAvgPool2DAttrsNode>()) return \"GlobalAvgPool2D\";")
    w("    return \"<attrs>\";")
    w("}")
    w("")
    w("// 保存 Relay 图节点的稳定编号、标签和输入边。")
    w("struct RelayGraphNode {")
    w("    int id;")
    w("    const Object* obj;")
    w("    std::string kind;")
    w("    std::string label;")
    w("    std::vector<int> inputs;")
    w("};")
    w("")
    w("// 深度优先遍历 Relay 表达式并构建去重后的图节点表。")
    w("int CollectRelayGraph(const Expr& e, std::unordered_map<const Object*, int>& ids, std::vector<RelayGraphNode>& nodes) {")
    w("    if (!e.defined()) return -1;")
    w("    auto it = ids.find(e.get());")
    w("    if (it != ids.end()) return it->second;")
    w("    int id = static_cast<int>(nodes.size());")
    w("    ids[e.get()] = id;")
    w("    nodes.push_back(RelayGraphNode{id, e.get(), \"\", \"\", {}});")
    w("    std::string kind;")
    w("    std::string label;")
    w("    std::vector<int> inputs;")
    w("    if (const auto* v = e.As<VarNode>()) {")
    w("        kind = \"Var\";")
    w("        label = v->vid->name_hint + \" : \" + PrintRelayType(v->type_annotation);")
    w("    } else if (const auto* c = e.As<ConstantNode>()) {")
    w("        kind = \"Constant\";")
    w("        std::ostringstream os;")
    w("        os << \"shape=[\";")
    w("        for (size_t i = 0; i < c->data->shape.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << c->data->shape[i];")
    w("        }")
    w("        os << \"], dtype=\" << NDArrayDTypeToString(c->data);")
    w("        label = os.str();")
    w("    } else if (const auto* call = e.As<CallNode>()) {")
    w("        kind = \"Call\";")
    w("        std::string op_name = \"<op>\";")
    w("        if (const auto* opn = call->op.As<OpNode>()) op_name = opn->name;")
    w("        label = op_name + \" attrs=\" + DescribeCallAttrs(call->attrs);")
    w("        for (const auto& arg : call->args) {")
    w("            int aid = CollectRelayGraph(arg, ids, nodes);")
    w("            inputs.push_back(aid);")
    w("        }")
    w("    } else {")
    w("        kind = \"Expr\";")
    w("        label = \"<expr>\";")
    w("    }")
    w("    nodes[id].kind = std::move(kind);")
    w("    nodes[id].label = std::move(label);")
    w("    nodes[id].inputs = std::move(inputs);")
    w("    return id;")
    w("}")
    w("")
    w("// 递归输出 Relay 表达式树，并避免重复展开共享节点。")
    w("void DumpRelayExpr(const Expr& e, std::ostream& os, int indent, std::unordered_set<const Object*>& seen) {")
    w("    if (!e.defined()) { os << Indent(indent) << \"<undef-expr>\\n\"; return; }")
    w("    if (seen.count(e.get())) {")
    w("        os << Indent(indent) << \"<visited node=\" << e.get() << \">\\n\";")
    w("        return;")
    w("    }")
    w("    seen.insert(e.get());")
    w("    if (const auto* v = e.As<VarNode>()) {")
    w("        os << Indent(indent) << \"Var(\" << v->vid->name_hint << \", \" << PrintRelayType(v->type_annotation) << \")\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* c = e.As<ConstantNode>()) {")
    w("        os << Indent(indent) << \"Constant(shape=[\";")
    w("        for (size_t i = 0; i < c->data->shape.size(); ++i) {")
    w("            if (i) os << \", \";")
    w("            os << c->data->shape[i];")
    w("        }")
    w("        os << \"], dtype=\" << NDArrayDTypeToString(c->data) << \")\\n\";")
    w("        return;")
    w("    }")
    w("    if (const auto* call = e.As<CallNode>()) {")
    w("        std::string op_name = \"<op>\";")
    w("        if (const auto* opn = call->op.As<OpNode>()) op_name = opn->name;")
    w("        os << Indent(indent) << \"Call(op=\" << op_name << \", attrs=\" << DescribeCallAttrs(call->attrs)")
    w("           << \", args=\" << call->args.size() << \")\\n\";")
    w("        for (size_t i = 0; i < call->args.size(); ++i) {")
    w("            os << Indent(indent + 2) << \"arg[\" << i << \"]:\\n\";")
    w("            DumpRelayExpr(call->args[i], os, indent + 4, seen);")
    w("        }")
    w("        return;")
    w("    }")
    w("    os << Indent(indent) << \"Expr(<unknown>)\\n\";")
    w("}")
    w("")
    w("// 输出 Relay 函数签名、表达式树和图边。")
    w("void DumpRelay(const Function& f, std::ostream& os) {")
    w("    os << \"================ Relay Function ================\\n\";")
    w("    os << \"params(\" << f->params.size() << \"):\\n\";")
    w("    for (size_t i = 0; i < f->params.size(); ++i) {")
    w("        const auto& p = f->params[i];")
    w("        os << \"  [\" << i << \"] \" << p->vid->name_hint << \" : \" << PrintRelayType(p->type_annotation) << \"\\n\";")
    w("    }")
    w("")
    w("    std::unordered_map<const Object*, int> ids;")
    w("    std::vector<RelayGraphNode> nodes;")
    w("    int out_id = CollectRelayGraph(f->body, ids, nodes);")
    w("    os << \"\\nRelay Graph Nodes(\" << nodes.size() << \"), output=n\" << out_id << \":\\n\";")
    w("    for (const auto& n : nodes) {")
    w("        os << \"  n\" << n.id << \" [\" << n.kind << \"] \" << n.label << \"\\n\";")
    w("    }")
    w("    os << \"Relay Graph Edges:\\n\";")
    w("    for (const auto& n : nodes) {")
    w("        for (int in : n.inputs) {")
    w("            if (in >= 0) os << \"  n\" << in << \" -> n\" << n.id << \"\\n\";")
    w("        }")
    w("    }")
    w("")
    w("    os << \"\\nRelay Expression DAG (from body):\\n\";")
    w("    std::unordered_set<const Object*> seen;")
    w("    DumpRelayExpr(f->body, os, 2, seen);")
    w("}")
    w("")
    w("// 用显式 CPU NDArray 常量构造导入后的 ResNet18 Relay 函数。")
    w("Function BuildResNet18Function() {")
    w("    Array<Var> params;")

    # 生成的 IR dump 只需要参数元数据，使用 CPU 零张量稳定承载 shape 与 dtype。
    value_map = {}
    sym_id = 0

    for param_name in imported.param_order:
        param = imported.params[param_name]
        c_name = f"const_{sym_id}_{sanitize(param.name)}"
        nd_name = f"nd_{sym_id}"
        sym_id += 1
        w("    // IR dump 不执行权重计算，显式 CPU 零张量仅承载常量 shape/dtype。")
        w(f"    runtime::NDArray {nd_name} = runtime::NDArray::Zeros({fmt_int_list(param.shape)}, runtime::DataTypeFromString(\"{param.dtype}\"), Device::CPU());")
        w(f"    Constant {c_name}({nd_name});")
        value_map[param.name] = c_name

    w("")
    # 将非 initializer 输入保留为具有完整类型的 Relay Var 参数。
    for inp in imported.function.inputs:
        if inp.name in value_map:
            continue
        v_name = f"input_{sym_id}_{sanitize(inp.name)}"
        sym_id += 1
        w(f"    Var {v_name}(\"{inp.name}\", TensorType({fmt_int_list(inp.shape)}, \"{inp.dtype}\"));")
        w(f"    params.push_back({v_name});")
        value_map[inp.name] = v_name

    w("")
    for i, node in enumerate(imported.function.nodes):
        if len(node.outputs) != 1:
            raise RuntimeError(
                f"Node '{node.name}' has {len(node.outputs)} outputs; only single output is supported."
            )
        missing = [name for name in node.inputs if name and name not in value_map]
        if missing:
            raise RuntimeError(f"Node '{node.name}' has missing inputs in map: {missing}")

        out_var = f"node_{i}_{sanitize(node.op_name)}"
        arg_vec = f"args_{i}"
        in_vars = [value_map[name] for name in node.inputs if name]
        in_text = ", ".join(in_vars)

        w(f"    // ONNX Node {i}: {node.op_name} ({node.name})")
        w(f"    std::vector<Expr> {arg_vec} = {{{in_text}}};")

        if node.op_name == "nn_conv2d":
            attrs = node.attrs
            w(
                f"    Conv2DAttrs attrs_{i} = Conv2DAttrs::Create({fmt_int_list(attrs['strides'])}, "
                f"{fmt_int_list(attrs['pads'])}, {fmt_int_list(attrs['dilations'])}, "
                f"{int(attrs['group'])}, {int(attrs['channels'])}, "
                f"{fmt_int_list(attrs['kernel_size'])}, \"{attrs['data_layout']}\", "
                f"\"{attrs['kernel_layout']}\", \"{attrs['out_layout']}\", \"{attrs['out_dtype']}\");"
            )
            w(f"    Call {out_var}(Op::Get(\"nn_conv2d\"), {arg_vec}, attrs_{i});")
        elif node.op_name == "nn_relu":
            w(f"    Call {out_var}(Op::Get(\"nn_relu\"), {arg_vec}, ReluAttrs::Create());")
        elif node.op_name == "nn_max_pool2d":
            attrs = node.attrs
            ceil_lit = "true" if attrs["ceil_mode"] else "false"
            w(
                f"    MaxPool2DAttrs attrs_{i} = MaxPool2DAttrs::Create({fmt_int_list(attrs['strides'])}, "
                f"{fmt_int_list(attrs['pads'])}, {fmt_int_list(attrs['dilations'])}, "
                f"{fmt_int_list(attrs['pool_size'])}, \"{attrs['layout']}\", {ceil_lit});"
            )
            w(f"    Call {out_var}(Op::Get(\"nn_max_pool2d\"), {arg_vec}, attrs_{i});")
        elif node.op_name == "add":
            w(f"    Call {out_var}(Op::Get(\"add\"), {arg_vec}, AddAttrs::Create());")
        elif node.op_name == "nn_global_avg_pool2d":
            w(
                f"    Call {out_var}(Op::Get(\"nn_global_avg_pool2d\"), {arg_vec}, "
                "GlobalAvgPool2DAttrs::Create());"
            )
        elif node.op_name == "nn_flatten":
            axis = int(node.attrs["axis"])
            w(f"    Call {out_var}(Op::Get(\"nn_flatten\"), {arg_vec}, FlattenAttrs::Create({axis}));")
        elif node.op_name == "nn_gemm":
            attrs = node.attrs
            w(
                f"    Call {out_var}(Op::Get(\"nn_gemm\"), {arg_vec}, "
                f"GemmAttrs::Create({float(attrs['alpha'])}f, {float(attrs['beta'])}f, "
                f"{int(attrs['transA'])}, {int(attrs['transB'])}));"
            )
        else:
            raise RuntimeError(f"Unhandled Relay op: {node.op_name}")

        value_map[node.outputs[0]] = out_var

    if not imported.function.outputs:
        raise RuntimeError("Model has no outputs.")
    out_name = imported.function.outputs[0].name
    if out_name not in value_map:
        raise RuntimeError(f"Graph output not found in value map: {out_name}")
    w("")
    w(f"    Expr out = {value_map[out_name]};")
    w("    return Function(params, out);")
    w("}")
    w("")
    w("}  // namespace")
    w("")
    w("// 构造 ResNet18，执行类型推导与 lowering，并写出 Relay/TIR 诊断文件。")
    w("int main() {")
    w("    try {")
    w("        std::ofstream devnull(\"NUL\");")
    w("        auto* old_cout_buf = std::cout.rdbuf(devnull.rdbuf());")
    w("        Function f = BuildResNet18Function();")
    w("        std::ofstream ofs(\"resnet18_ir_dump.txt\", std::ios::out | std::ios::trunc);")
    w("        if (!ofs) throw std::runtime_error(\"Failed to open resnet18_ir_dump.txt for writing\");")
    w("")
    w("        DumpRelay(f, ofs);")
    w("")
    w("        api::internal::LoweredGraph lowered =")
    w("            api::internal::LowerGraph(relay::InferTypePass(f));")
    w("        for (const auto& primitive : lowered.primitives) {")
    w("            DumpPrimFunc(primitive.lowered->prim_func, ofs);")
    w("        }")
    w("        std::cout.rdbuf(old_cout_buf);")
    w("")
    w("        std::cout << \"\\nIR dump written to test/resnet18_ir_dump.txt\" << std::endl;")
    w("        return 0;")
    w("    } catch (const std::exception& e) {")
    w("        std::cerr << \"ERROR: \" << e.what() << std::endl;")
    w("        return 1;")
    w("    }")
    w("}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated C++ dump tool: {out_path}")


def main():
    """解析命令行参数并生成 ResNet18 IR dump 源文件。"""
    parser = argparse.ArgumentParser(
        description="Generate C++ tool that lowers ONNX ResNet18 to Relay and TIR and dumps IR."
    )
    parser.add_argument("--model", type=Path, default=Path("resnet18.onnx"))
    parser.add_argument("--out", type=Path, default=Path("test/resnet18_ir_dump.cpp"))
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Explicit positive binding for an unresolved axis-0 batch dimension.",
    )
    args = parser.parse_args()

    emit_cpp(args.model, args.out, args.batch)


if __name__ == "__main__":
    main()
