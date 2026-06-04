"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import argparse
import re
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto


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
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not name:
        name = "v"
    if name[0].isdigit():
        name = "v_" + name
    return name


def fmt_int_list(v):
    return "{" + ", ".join(str(int(x)) for x in v) + "}"


def onnx_dtype_to_kxc(dtype: int) -> str:
    if dtype == TensorProto.FLOAT:
        return "float32"
    if dtype == TensorProto.DOUBLE:
        return "float64"
    if dtype == TensorProto.INT64:
        return "int64"
    if dtype == TensorProto.INT32:
        return "int32"
    if dtype == TensorProto.INT8:
        return "int8"
    if dtype == TensorProto.UINT8:
        return "uint8"
    if dtype == TensorProto.BOOL:
        return "bool"
    # Fallback for this compiler's minimal dtype set.
    return "float32"


def get_attr(node, name, default):
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


def value_info_shape_and_dtype(value_info, default_batch):
    t = value_info.type.tensor_type
    dtype = onnx_dtype_to_kxc(t.elem_type)
    shape = []
    for i, d in enumerate(t.shape.dim):
        if d.HasField("dim_value"):
            shape.append(int(d.dim_value))
        else:
            shape.append(default_batch if i == 0 else 1)
    return shape, dtype


def emit_cpp(model_path: Path, out_path: Path, default_batch: int):
    model = onnx.load(str(model_path))
    graph = model.graph

    # Build lookup for input/output type info.
    vi_by_name = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        vi_by_name[vi.name] = vi

    lines = []
    w = lines.append

    w('#include "relay/relay.h"')
    w('#include "relay/op.h"')
    w('#include "relay/transforms/lower.h"')
    w('#include "tir/stmt.h"')
    w('#include "tir/expr.h"')
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
    w("std::string Indent(int n) { return std::string(n, ' '); }")
    w("")
    w("std::string DTypeToString(const tir::DataType& dt) {")
    w("    if (dt.code == 2) return \"float\" + std::to_string(dt.bits);")
    w("    if (dt.code == 0) return \"int\" + std::to_string(dt.bits);")
    w("    if (dt.code == 1) return dt.bits == 1 ? \"bool\" : (\"uint\" + std::to_string(dt.bits));")
    w("    if (dt.code == 3) return \"handle\";")
    w("    return \"dtype(code=\" + std::to_string(dt.code) + \",bits=\" + std::to_string(dt.bits) + \")\";")
    w("}")
    w("")
    w("std::string NDArrayDTypeToString(const runtime::NDArray& arr) {")
    w("    const DLDataType& dt = arr->dl_tensor.dtype;")
    w("    if (dt.code == kDLFloat) return \"float\" + std::to_string(dt.bits);")
    w("    if (dt.code == kDLInt) return \"int\" + std::to_string(dt.bits);")
    w("    if (dt.code == kDLUint) return dt.bits == 1 ? \"bool\" : (\"uint\" + std::to_string(dt.bits));")
    w("    return \"unknown\";")
    w("}")
    w("")
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
    w("struct RelayGraphNode {")
    w("    int id;")
    w("    const Object* obj;")
    w("    std::string kind;")
    w("    std::string label;")
    w("    std::vector<int> inputs;")
    w("};")
    w("")
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
    w("Function BuildResNet18Function() {")
    w("    Array<Var> params;")

    # Initializers as Constant.
    value_map = {}
    sym_id = 0

    for init in graph.initializer:
        c_name = f"const_{sym_id}_{sanitize(init.name)}"
        nd_name = f"nd_{sym_id}"
        sym_id += 1
        shape = [int(d) for d in init.dims]
        dtype = onnx_dtype_to_kxc(int(init.data_type))
        w(f"    runtime::NDArray {nd_name}({fmt_int_list(shape)}, \"{dtype}\");")
        w(f"    Constant {c_name}({nd_name});")
        value_map[init.name] = c_name

    w("")
    # Inputs as typed Var params.
    for inp in graph.input:
        if inp.name in value_map:
            continue
        if inp.name not in vi_by_name:
            raise RuntimeError(f"Missing value info for input: {inp.name}")
        shape, dtype = value_info_shape_and_dtype(vi_by_name[inp.name], default_batch)
        v_name = f"input_{sym_id}_{sanitize(inp.name)}"
        sym_id += 1
        w(f"    Var {v_name}(\"{inp.name}\", TensorType({fmt_int_list(shape)}, \"{dtype}\"));")
        w(f"    params.push_back({v_name});")
        value_map[inp.name] = v_name

    w("")
    for i, node in enumerate(graph.node):
        if node.op_type not in SUPPORTED_OPS:
            raise RuntimeError(
                f"Unsupported op '{node.op_type}' in resnet18.onnx for current lower path."
            )
        if len(node.output) != 1:
            raise RuntimeError(
                f"Node '{node.name or node.op_type}' has {len(node.output)} outputs; only single output is supported."
            )
        missing = [name for name in node.input if name and name not in value_map]
        if missing:
            raise RuntimeError(
                f"Node '{node.name or node.op_type}' has missing inputs in map: {missing}"
            )

        out_var = f"node_{i}_{sanitize(node.op_type)}"
        arg_vec = f"args_{i}"
        in_vars = [value_map[name] for name in node.input if name]
        in_text = ", ".join(in_vars)

        w(f"    // ONNX Node {i}: {node.op_type} ({node.name})")
        w(f"    std::vector<Expr> {arg_vec} = {{{in_text}}};")

        if node.op_type == "Conv":
            strides = get_attr(node, "strides", [1, 1])
            pads = get_attr(node, "pads", [0, 0, 0, 0])
            dilations = get_attr(node, "dilations", [1, 1])
            group = get_attr(node, "group", 1)
            kernel_shape = get_attr(node, "kernel_shape", [3, 3])
            weight_name = node.input[1]
            weight_init = next((x for x in graph.initializer if x.name == weight_name), None)
            if weight_init is None:
                raise RuntimeError(f"Conv weight initializer not found: {weight_name}")
            channels = int(weight_init.dims[0])
            w(
                f"    Conv2DAttrs attrs_{i} = Conv2DAttrs::Create({fmt_int_list(strides)}, "
                f"{fmt_int_list(pads)}, {fmt_int_list(dilations)}, {group}, {channels}, "
                f"{fmt_int_list(kernel_shape)}, \"NCHW\", \"OIHW\", \"\", \"\");"
            )
            w(f"    Call {out_var}(Op::Get(\"nn_conv2d\"), {arg_vec}, attrs_{i});")
        elif node.op_type == "Relu":
            w(f"    Call {out_var}(Op::Get(\"nn_relu\"), {arg_vec}, ReluAttrs::Create());")
        elif node.op_type == "MaxPool":
            kernel = get_attr(node, "kernel_shape", [1, 1])
            strides = get_attr(node, "strides", [1, 1])
            pads = get_attr(node, "pads", [0, 0, 0, 0])
            dilations = get_attr(node, "dilations", [1, 1])
            ceil_mode = bool(get_attr(node, "ceil_mode", 0))
            ceil_lit = "true" if ceil_mode else "false"
            w(
                f"    MaxPool2DAttrs attrs_{i} = MaxPool2DAttrs::Create({fmt_int_list(strides)}, "
                f"{fmt_int_list(pads)}, {fmt_int_list(dilations)}, {fmt_int_list(kernel)}, "
                f"\"NCHW\", {ceil_lit});"
            )
            w(f"    Call {out_var}(Op::Get(\"nn_max_pool2d\"), {arg_vec}, attrs_{i});")
        elif node.op_type == "Add":
            w(f"    Call {out_var}(Op::Get(\"add\"), {arg_vec}, AddAttrs::Create());")
        elif node.op_type == "GlobalAveragePool":
            w(
                f"    Call {out_var}(Op::Get(\"nn_global_avg_pool2d\"), {arg_vec}, "
                "GlobalAvgPool2DAttrs::Create());"
            )
        elif node.op_type == "Flatten":
            axis = get_attr(node, "axis", 1)
            w(f"    Call {out_var}(Op::Get(\"nn_flatten\"), {arg_vec}, FlattenAttrs::Create({axis}));")
        elif node.op_type == "Gemm":
            alpha = get_attr(node, "alpha", 1.0)
            beta = get_attr(node, "beta", 1.0)
            trans_a = get_attr(node, "transA", 0)
            trans_b = get_attr(node, "transB", 0)
            w(
                f"    Call {out_var}(Op::Get(\"nn_gemm\"), {arg_vec}, "
                f"GemmAttrs::Create({alpha}f, {beta}f, {trans_a}, {trans_b}));"
            )
        else:
            raise RuntimeError(f"Unhandled op: {node.op_type}")

        value_map[node.output[0]] = out_var

    if not graph.output:
        raise RuntimeError("Model has no outputs.")
    out_name = graph.output[0].name
    if out_name not in value_map:
        raise RuntimeError(f"Graph output not found in value map: {out_name}")
    w("")
    w(f"    Expr out = {value_map[out_name]};")
    w("    return Function(params, out);")
    w("}")
    w("")
    w("}  // namespace")
    w("")
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
    w("        tir::PrimFunc pf = LowerToTIR(f);")
    w("        DumpPrimFunc(pf, ofs);")
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
    parser = argparse.ArgumentParser(
        description="Generate C++ tool that lowers ONNX ResNet18 to Relay and TIR and dumps IR."
    )
    parser.add_argument("--model", type=Path, default=Path("resnet18.onnx"))
    parser.add_argument("--out", type=Path, default=Path("test/resnet18_ir_dump.cpp"))
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    emit_cpp(args.model, args.out, args.batch)


if __name__ == "__main__":
    main()
