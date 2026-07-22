/*! \file test/resnet18_ir_dump.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "kxc/relay/relay.h"
#include "kxc/relay/op.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/tir/pass/print_ir.h"
#include "kxc/tir/transforms/pipeline.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace kxc;
using namespace kxc::relay;

namespace {

// 生成 IR 树形文本所需的缩进。
std::string Indent(int n) { return std::string(n, ' '); }

// 将 NDArray 的 DLPack dtype 格式化为可读名称。
std::string NDArrayDTypeToString(const runtime::NDArray& arr) {
    const DLDataType& dt = arr->dl_tensor.dtype;
    if (dt.code == kDLFloat) return "float" + std::to_string(dt.bits);
    if (dt.code == kDLInt) return "int" + std::to_string(dt.bits);
    if (dt.code == kDLUInt) return "uint" + std::to_string(dt.bits);
    if (dt.code == kDLBool) return "bool";
    return "unknown";
}

// 递归格式化 Relay 类型及其 shape、dtype。
std::string PrintRelayType(const Type& ty) {
    if (!ty.defined()) return "<none>";
    if (const auto* t = ty.As<TensorTypeNode>()) {
        std::ostringstream os;
        os << "TensorType(shape=[";
        for (size_t i = 0; i < t->shape.size(); ++i) {
            if (i) os << ", ";
            os << t->shape[i];
        }
        os << "], dtype=" << t->dtype << ")";
        return os.str();
    }
    return "<Type>";
}

// 提取受支持算子 attrs 的关键字段用于诊断输出。
std::string DescribeCallAttrs(const ObjectRef& attrs) {
    if (!attrs.defined()) return "<none>";
    if (const auto* a = attrs.As<Conv2DAttrsNode>()) {
        std::ostringstream os;
        os << "Conv2D(strides=" << a->strides[0] << "," << a->strides[1]
           << "; padding=" << (a->padding.empty() ? 0 : a->padding[0])
           << "," << (a->padding.size() > 1 ? a->padding[1] : 0)
           << "; groups=" << a->groups << ")";
        return os.str();
    }
    if (const auto* a = attrs.As<MaxPool2DAttrsNode>()) {
        std::ostringstream os;
        os << "Pool2D(kernel=" << a->pool_size[0] << "," << a->pool_size[1]
           << "; strides=" << a->strides[0] << "," << a->strides[1] << ")";
        return os.str();
    }
    if (const auto* a = attrs.As<FlattenAttrsNode>()) {
        return std::string("Flatten(axis=") + std::to_string(a->axis) + ")";
    }
    if (const auto* a = attrs.As<GemmAttrsNode>()) {
        std::ostringstream os;
        os << "Gemm(alpha=" << a->alpha << ", beta=" << a->beta << ", transA=" << a->transA << ", transB=" << a->transB << ")";
        return os.str();
    }
    if (attrs.As<AddAttrsNode>()) return "Add";
    if (attrs.As<ReluAttrsNode>()) return "Relu";
    if (attrs.As<GlobalAvgPool2DAttrsNode>()) return "GlobalAvgPool2D";
    return "<attrs>";
}

// 保存 Relay 图节点的稳定编号、标签和输入边。
struct RelayGraphNode {
    int id;
    const Object* obj;
    std::string kind;
    std::string label;
    std::vector<int> inputs;
};

// 深度优先遍历 Relay 表达式并构建去重后的图节点表。
int CollectRelayGraph(const Expr& e, std::unordered_map<const Object*, int>& ids, std::vector<RelayGraphNode>& nodes) {
    if (!e.defined()) return -1;
    auto it = ids.find(e.get());
    if (it != ids.end()) return it->second;
    int id = static_cast<int>(nodes.size());
    ids[e.get()] = id;
    nodes.push_back(RelayGraphNode{id, e.get(), "", "", {}});
    std::string kind;
    std::string label;
    std::vector<int> inputs;
    if (const auto* v = e.As<VarNode>()) {
        kind = "Var";
        label = v->vid->name_hint + " : " + PrintRelayType(v->type_annotation);
    } else if (const auto* c = e.As<ConstantNode>()) {
        kind = "Constant";
        std::ostringstream os;
        os << "shape=[";
        for (size_t i = 0; i < c->data->shape_storage.size(); ++i) {
            if (i) os << ", ";
            os << c->data->shape_storage[i];
        }
        os << "], dtype=" << NDArrayDTypeToString(c->data);
        label = os.str();
    } else if (const auto* call = e.As<CallNode>()) {
        kind = "Call";
        std::string op_name = "<op>";
        if (const auto* opn = call->op.As<OpNode>()) op_name = opn->name;
        label = op_name + " attrs=" + DescribeCallAttrs(call->attrs);
        for (const auto& arg : call->args) {
            int aid = CollectRelayGraph(arg, ids, nodes);
            inputs.push_back(aid);
        }
    } else {
        kind = "Expr";
        label = "<expr>";
    }
    nodes[id].kind = std::move(kind);
    nodes[id].label = std::move(label);
    nodes[id].inputs = std::move(inputs);
    return id;
}

// 递归输出 Relay 表达式树，并用 seen 集合阻止共享节点重复展开。
void DumpRelayExpr(const Expr& e, std::ostream& os, int indent, std::unordered_set<const Object*>& seen) {
    if (!e.defined()) { os << Indent(indent) << "<undef-expr>\n"; return; }
    if (seen.count(e.get())) {
        os << Indent(indent) << "<visited node=" << e.get() << ">\n";
        return;
    }
    seen.insert(e.get());
    if (const auto* v = e.As<VarNode>()) {
        os << Indent(indent) << "Var(" << v->vid->name_hint << ", " << PrintRelayType(v->type_annotation) << ")\n";
        return;
    }
    if (const auto* c = e.As<ConstantNode>()) {
        os << Indent(indent) << "Constant(shape=[";
        for (size_t i = 0; i < c->data->shape_storage.size(); ++i) {
            if (i) os << ", ";
            os << c->data->shape_storage[i];
        }
        os << "], dtype=" << NDArrayDTypeToString(c->data) << ")\n";
        return;
    }
    if (const auto* call = e.As<CallNode>()) {
        std::string op_name = "<op>";
        if (const auto* opn = call->op.As<OpNode>()) op_name = opn->name;
        os << Indent(indent) << "Call(op=" << op_name << ", attrs=" << DescribeCallAttrs(call->attrs)
           << ", args=" << call->args.size() << ")\n";
        for (size_t i = 0; i < call->args.size(); ++i) {
            os << Indent(indent + 2) << "arg[" << i << "]:\n";
            DumpRelayExpr(call->args[i], os, indent + 4, seen);
        }
        return;
    }
    os << Indent(indent) << "Expr(<unknown>)\n";
}

// 输出 Relay 函数签名、表达式树和图边。
void DumpRelay(const Function& f, std::ostream& os) {
    os << "================ Relay Function ================\n";
    os << "params(" << f->params.size() << "):\n";
    for (size_t i = 0; i < f->params.size(); ++i) {
        const auto& p = f->params[i];
        os << "  [" << i << "] " << p->vid->name_hint << " : " << PrintRelayType(p->type_annotation) << "\n";
    }

    std::unordered_map<const Object*, int> ids;
    std::vector<RelayGraphNode> nodes;
    int out_id = CollectRelayGraph(f->body, ids, nodes);
    os << "\nRelay Graph Nodes(" << nodes.size() << "), output=n" << out_id << ":\n";
    for (const auto& n : nodes) {
        os << "  n" << n.id << " [" << n.kind << "] " << n.label << "\n";
    }
    os << "Relay Graph Edges:\n";
    for (const auto& n : nodes) {
        for (int in : n.inputs) {
            if (in >= 0) os << "  n" << in << " -> n" << n.id << "\n";
        }
    }

    os << "\nRelay Expression DAG (from body):\n";
    std::unordered_set<const Object*> seen;
    DumpRelayExpr(f->body, os, 2, seen);
}

// 用显式 CPU NDArray 常量构造可供 pass/lowering 检查的 ResNet18 Relay 函数。
Function BuildResNet18Function() {
    Array<Var> params;
    // IR dump 不执行权重计算，显式 cpu:0 零张量仅稳定承载常量 shape/dtype。
    const auto zeros = [](Array<int64_t> shape) {
        return runtime::NDArray::Zeros(shape, runtime::DataTypeFromString("float32"),
                                       Device::CPU());
    };
    runtime::NDArray nd_0 = zeros({1000, 512});
    Constant const_0_fc_weight(nd_0);
    runtime::NDArray nd_1 = zeros({1000});
    Constant const_1_fc_bias(nd_1);
    runtime::NDArray nd_2 = zeros({64, 3, 7, 7});
    Constant const_2_onnx__Conv_193(nd_2);
    runtime::NDArray nd_3 = zeros({64});
    Constant const_3_onnx__Conv_194(nd_3);
    runtime::NDArray nd_4 = zeros({64, 64, 3, 3});
    Constant const_4_onnx__Conv_196(nd_4);
    runtime::NDArray nd_5 = zeros({64});
    Constant const_5_onnx__Conv_197(nd_5);
    runtime::NDArray nd_6 = zeros({64, 64, 3, 3});
    Constant const_6_onnx__Conv_199(nd_6);
    runtime::NDArray nd_7 = zeros({64});
    Constant const_7_onnx__Conv_200(nd_7);
    runtime::NDArray nd_8 = zeros({64, 64, 3, 3});
    Constant const_8_onnx__Conv_202(nd_8);
    runtime::NDArray nd_9 = zeros({64});
    Constant const_9_onnx__Conv_203(nd_9);
    runtime::NDArray nd_10 = zeros({64, 64, 3, 3});
    Constant const_10_onnx__Conv_205(nd_10);
    runtime::NDArray nd_11 = zeros({64});
    Constant const_11_onnx__Conv_206(nd_11);
    runtime::NDArray nd_12 = zeros({128, 64, 3, 3});
    Constant const_12_onnx__Conv_208(nd_12);
    runtime::NDArray nd_13 = zeros({128});
    Constant const_13_onnx__Conv_209(nd_13);
    runtime::NDArray nd_14 = zeros({128, 128, 3, 3});
    Constant const_14_onnx__Conv_211(nd_14);
    runtime::NDArray nd_15 = zeros({128});
    Constant const_15_onnx__Conv_212(nd_15);
    runtime::NDArray nd_16 = zeros({128, 64, 1, 1});
    Constant const_16_onnx__Conv_214(nd_16);
    runtime::NDArray nd_17 = zeros({128});
    Constant const_17_onnx__Conv_215(nd_17);
    runtime::NDArray nd_18 = zeros({128, 128, 3, 3});
    Constant const_18_onnx__Conv_217(nd_18);
    runtime::NDArray nd_19 = zeros({128});
    Constant const_19_onnx__Conv_218(nd_19);
    runtime::NDArray nd_20 = zeros({128, 128, 3, 3});
    Constant const_20_onnx__Conv_220(nd_20);
    runtime::NDArray nd_21 = zeros({128});
    Constant const_21_onnx__Conv_221(nd_21);
    runtime::NDArray nd_22 = zeros({256, 128, 3, 3});
    Constant const_22_onnx__Conv_223(nd_22);
    runtime::NDArray nd_23 = zeros({256});
    Constant const_23_onnx__Conv_224(nd_23);
    runtime::NDArray nd_24 = zeros({256, 256, 3, 3});
    Constant const_24_onnx__Conv_226(nd_24);
    runtime::NDArray nd_25 = zeros({256});
    Constant const_25_onnx__Conv_227(nd_25);
    runtime::NDArray nd_26 = zeros({256, 128, 1, 1});
    Constant const_26_onnx__Conv_229(nd_26);
    runtime::NDArray nd_27 = zeros({256});
    Constant const_27_onnx__Conv_230(nd_27);
    runtime::NDArray nd_28 = zeros({256, 256, 3, 3});
    Constant const_28_onnx__Conv_232(nd_28);
    runtime::NDArray nd_29 = zeros({256});
    Constant const_29_onnx__Conv_233(nd_29);
    runtime::NDArray nd_30 = zeros({256, 256, 3, 3});
    Constant const_30_onnx__Conv_235(nd_30);
    runtime::NDArray nd_31 = zeros({256});
    Constant const_31_onnx__Conv_236(nd_31);
    runtime::NDArray nd_32 = zeros({512, 256, 3, 3});
    Constant const_32_onnx__Conv_238(nd_32);
    runtime::NDArray nd_33 = zeros({512});
    Constant const_33_onnx__Conv_239(nd_33);
    runtime::NDArray nd_34 = zeros({512, 512, 3, 3});
    Constant const_34_onnx__Conv_241(nd_34);
    runtime::NDArray nd_35 = zeros({512});
    Constant const_35_onnx__Conv_242(nd_35);
    runtime::NDArray nd_36 = zeros({512, 256, 1, 1});
    Constant const_36_onnx__Conv_244(nd_36);
    runtime::NDArray nd_37 = zeros({512});
    Constant const_37_onnx__Conv_245(nd_37);
    runtime::NDArray nd_38 = zeros({512, 512, 3, 3});
    Constant const_38_onnx__Conv_247(nd_38);
    runtime::NDArray nd_39 = zeros({512});
    Constant const_39_onnx__Conv_248(nd_39);
    runtime::NDArray nd_40 = zeros({512, 512, 3, 3});
    Constant const_40_onnx__Conv_250(nd_40);
    runtime::NDArray nd_41 = zeros({512});
    Constant const_41_onnx__Conv_251(nd_41);

    Var input_42_input("input", TensorType({1, 3, 224, 224}, "float32"));
    params.push_back(input_42_input);

    // ONNX Node 0: Conv (/conv1/Conv)
    std::vector<Expr> args_0 = {input_42_input, const_2_onnx__Conv_193, const_3_onnx__Conv_194};
    Conv2DAttrs attrs_0 = Conv2DAttrs::Create({2, 2}, {3, 3, 3, 3}, {1, 1}, 1, 64, {7, 7}, "NCHW", "OIHW", "", "");
    Call node_0_Conv(Op::Get("nn_conv2d"), args_0, attrs_0);
    // ONNX Node 1: Relu (/relu/Relu)
    std::vector<Expr> args_1 = {node_0_Conv};
    Call node_1_Relu(Op::Get("nn_relu"), args_1, ReluAttrs::Create());
    // ONNX Node 2: MaxPool (/maxpool/MaxPool)
    std::vector<Expr> args_2 = {node_1_Relu};
    MaxPool2DAttrs attrs_2 = MaxPool2DAttrs::Create({2, 2}, {1, 1, 1, 1}, {1, 1}, {3, 3}, "NCHW", false);
    Call node_2_MaxPool(Op::Get("nn_max_pool2d"), args_2, attrs_2);
    // ONNX Node 3: Conv (/layer1/layer1.0/conv1/Conv)
    std::vector<Expr> args_3 = {node_2_MaxPool, const_4_onnx__Conv_196, const_5_onnx__Conv_197};
    Conv2DAttrs attrs_3 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 64, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_3_Conv(Op::Get("nn_conv2d"), args_3, attrs_3);
    // ONNX Node 4: Relu (/layer1/layer1.0/relu/Relu)
    std::vector<Expr> args_4 = {node_3_Conv};
    Call node_4_Relu(Op::Get("nn_relu"), args_4, ReluAttrs::Create());
    // ONNX Node 5: Conv (/layer1/layer1.0/conv2/Conv)
    std::vector<Expr> args_5 = {node_4_Relu, const_6_onnx__Conv_199, const_7_onnx__Conv_200};
    Conv2DAttrs attrs_5 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 64, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_5_Conv(Op::Get("nn_conv2d"), args_5, attrs_5);
    // ONNX Node 6: Add (/layer1/layer1.0/Add)
    std::vector<Expr> args_6 = {node_5_Conv, node_2_MaxPool};
    Call node_6_Add(Op::Get("add"), args_6, AddAttrs::Create());
    // ONNX Node 7: Relu (/layer1/layer1.0/relu_1/Relu)
    std::vector<Expr> args_7 = {node_6_Add};
    Call node_7_Relu(Op::Get("nn_relu"), args_7, ReluAttrs::Create());
    // ONNX Node 8: Conv (/layer1/layer1.1/conv1/Conv)
    std::vector<Expr> args_8 = {node_7_Relu, const_8_onnx__Conv_202, const_9_onnx__Conv_203};
    Conv2DAttrs attrs_8 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 64, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_8_Conv(Op::Get("nn_conv2d"), args_8, attrs_8);
    // ONNX Node 9: Relu (/layer1/layer1.1/relu/Relu)
    std::vector<Expr> args_9 = {node_8_Conv};
    Call node_9_Relu(Op::Get("nn_relu"), args_9, ReluAttrs::Create());
    // ONNX Node 10: Conv (/layer1/layer1.1/conv2/Conv)
    std::vector<Expr> args_10 = {node_9_Relu, const_10_onnx__Conv_205, const_11_onnx__Conv_206};
    Conv2DAttrs attrs_10 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 64, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_10_Conv(Op::Get("nn_conv2d"), args_10, attrs_10);
    // ONNX Node 11: Add (/layer1/layer1.1/Add)
    std::vector<Expr> args_11 = {node_10_Conv, node_7_Relu};
    Call node_11_Add(Op::Get("add"), args_11, AddAttrs::Create());
    // ONNX Node 12: Relu (/layer1/layer1.1/relu_1/Relu)
    std::vector<Expr> args_12 = {node_11_Add};
    Call node_12_Relu(Op::Get("nn_relu"), args_12, ReluAttrs::Create());
    // ONNX Node 13: Conv (/layer2/layer2.0/conv1/Conv)
    std::vector<Expr> args_13 = {node_12_Relu, const_12_onnx__Conv_208, const_13_onnx__Conv_209};
    Conv2DAttrs attrs_13 = Conv2DAttrs::Create({2, 2}, {1, 1, 1, 1}, {1, 1}, 1, 128, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_13_Conv(Op::Get("nn_conv2d"), args_13, attrs_13);
    // ONNX Node 14: Relu (/layer2/layer2.0/relu/Relu)
    std::vector<Expr> args_14 = {node_13_Conv};
    Call node_14_Relu(Op::Get("nn_relu"), args_14, ReluAttrs::Create());
    // ONNX Node 15: Conv (/layer2/layer2.0/conv2/Conv)
    std::vector<Expr> args_15 = {node_14_Relu, const_14_onnx__Conv_211, const_15_onnx__Conv_212};
    Conv2DAttrs attrs_15 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 128, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_15_Conv(Op::Get("nn_conv2d"), args_15, attrs_15);
    // ONNX Node 16: Conv (/layer2/layer2.0/downsample/downsample.0/Conv)
    std::vector<Expr> args_16 = {node_12_Relu, const_16_onnx__Conv_214, const_17_onnx__Conv_215};
    Conv2DAttrs attrs_16 = Conv2DAttrs::Create({2, 2}, {0, 0, 0, 0}, {1, 1}, 1, 128, {1, 1}, "NCHW", "OIHW", "", "");
    Call node_16_Conv(Op::Get("nn_conv2d"), args_16, attrs_16);
    // ONNX Node 17: Add (/layer2/layer2.0/Add)
    std::vector<Expr> args_17 = {node_15_Conv, node_16_Conv};
    Call node_17_Add(Op::Get("add"), args_17, AddAttrs::Create());
    // ONNX Node 18: Relu (/layer2/layer2.0/relu_1/Relu)
    std::vector<Expr> args_18 = {node_17_Add};
    Call node_18_Relu(Op::Get("nn_relu"), args_18, ReluAttrs::Create());
    // ONNX Node 19: Conv (/layer2/layer2.1/conv1/Conv)
    std::vector<Expr> args_19 = {node_18_Relu, const_18_onnx__Conv_217, const_19_onnx__Conv_218};
    Conv2DAttrs attrs_19 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 128, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_19_Conv(Op::Get("nn_conv2d"), args_19, attrs_19);
    // ONNX Node 20: Relu (/layer2/layer2.1/relu/Relu)
    std::vector<Expr> args_20 = {node_19_Conv};
    Call node_20_Relu(Op::Get("nn_relu"), args_20, ReluAttrs::Create());
    // ONNX Node 21: Conv (/layer2/layer2.1/conv2/Conv)
    std::vector<Expr> args_21 = {node_20_Relu, const_20_onnx__Conv_220, const_21_onnx__Conv_221};
    Conv2DAttrs attrs_21 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 128, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_21_Conv(Op::Get("nn_conv2d"), args_21, attrs_21);
    // ONNX Node 22: Add (/layer2/layer2.1/Add)
    std::vector<Expr> args_22 = {node_21_Conv, node_18_Relu};
    Call node_22_Add(Op::Get("add"), args_22, AddAttrs::Create());
    // ONNX Node 23: Relu (/layer2/layer2.1/relu_1/Relu)
    std::vector<Expr> args_23 = {node_22_Add};
    Call node_23_Relu(Op::Get("nn_relu"), args_23, ReluAttrs::Create());
    // ONNX Node 24: Conv (/layer3/layer3.0/conv1/Conv)
    std::vector<Expr> args_24 = {node_23_Relu, const_22_onnx__Conv_223, const_23_onnx__Conv_224};
    Conv2DAttrs attrs_24 = Conv2DAttrs::Create({2, 2}, {1, 1, 1, 1}, {1, 1}, 1, 256, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_24_Conv(Op::Get("nn_conv2d"), args_24, attrs_24);
    // ONNX Node 25: Relu (/layer3/layer3.0/relu/Relu)
    std::vector<Expr> args_25 = {node_24_Conv};
    Call node_25_Relu(Op::Get("nn_relu"), args_25, ReluAttrs::Create());
    // ONNX Node 26: Conv (/layer3/layer3.0/conv2/Conv)
    std::vector<Expr> args_26 = {node_25_Relu, const_24_onnx__Conv_226, const_25_onnx__Conv_227};
    Conv2DAttrs attrs_26 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 256, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_26_Conv(Op::Get("nn_conv2d"), args_26, attrs_26);
    // ONNX Node 27: Conv (/layer3/layer3.0/downsample/downsample.0/Conv)
    std::vector<Expr> args_27 = {node_23_Relu, const_26_onnx__Conv_229, const_27_onnx__Conv_230};
    Conv2DAttrs attrs_27 = Conv2DAttrs::Create({2, 2}, {0, 0, 0, 0}, {1, 1}, 1, 256, {1, 1}, "NCHW", "OIHW", "", "");
    Call node_27_Conv(Op::Get("nn_conv2d"), args_27, attrs_27);
    // ONNX Node 28: Add (/layer3/layer3.0/Add)
    std::vector<Expr> args_28 = {node_26_Conv, node_27_Conv};
    Call node_28_Add(Op::Get("add"), args_28, AddAttrs::Create());
    // ONNX Node 29: Relu (/layer3/layer3.0/relu_1/Relu)
    std::vector<Expr> args_29 = {node_28_Add};
    Call node_29_Relu(Op::Get("nn_relu"), args_29, ReluAttrs::Create());
    // ONNX Node 30: Conv (/layer3/layer3.1/conv1/Conv)
    std::vector<Expr> args_30 = {node_29_Relu, const_28_onnx__Conv_232, const_29_onnx__Conv_233};
    Conv2DAttrs attrs_30 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 256, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_30_Conv(Op::Get("nn_conv2d"), args_30, attrs_30);
    // ONNX Node 31: Relu (/layer3/layer3.1/relu/Relu)
    std::vector<Expr> args_31 = {node_30_Conv};
    Call node_31_Relu(Op::Get("nn_relu"), args_31, ReluAttrs::Create());
    // ONNX Node 32: Conv (/layer3/layer3.1/conv2/Conv)
    std::vector<Expr> args_32 = {node_31_Relu, const_30_onnx__Conv_235, const_31_onnx__Conv_236};
    Conv2DAttrs attrs_32 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 256, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_32_Conv(Op::Get("nn_conv2d"), args_32, attrs_32);
    // ONNX Node 33: Add (/layer3/layer3.1/Add)
    std::vector<Expr> args_33 = {node_32_Conv, node_29_Relu};
    Call node_33_Add(Op::Get("add"), args_33, AddAttrs::Create());
    // ONNX Node 34: Relu (/layer3/layer3.1/relu_1/Relu)
    std::vector<Expr> args_34 = {node_33_Add};
    Call node_34_Relu(Op::Get("nn_relu"), args_34, ReluAttrs::Create());
    // ONNX Node 35: Conv (/layer4/layer4.0/conv1/Conv)
    std::vector<Expr> args_35 = {node_34_Relu, const_32_onnx__Conv_238, const_33_onnx__Conv_239};
    Conv2DAttrs attrs_35 = Conv2DAttrs::Create({2, 2}, {1, 1, 1, 1}, {1, 1}, 1, 512, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_35_Conv(Op::Get("nn_conv2d"), args_35, attrs_35);
    // ONNX Node 36: Relu (/layer4/layer4.0/relu/Relu)
    std::vector<Expr> args_36 = {node_35_Conv};
    Call node_36_Relu(Op::Get("nn_relu"), args_36, ReluAttrs::Create());
    // ONNX Node 37: Conv (/layer4/layer4.0/conv2/Conv)
    std::vector<Expr> args_37 = {node_36_Relu, const_34_onnx__Conv_241, const_35_onnx__Conv_242};
    Conv2DAttrs attrs_37 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 512, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_37_Conv(Op::Get("nn_conv2d"), args_37, attrs_37);
    // ONNX Node 38: Conv (/layer4/layer4.0/downsample/downsample.0/Conv)
    std::vector<Expr> args_38 = {node_34_Relu, const_36_onnx__Conv_244, const_37_onnx__Conv_245};
    Conv2DAttrs attrs_38 = Conv2DAttrs::Create({2, 2}, {0, 0, 0, 0}, {1, 1}, 1, 512, {1, 1}, "NCHW", "OIHW", "", "");
    Call node_38_Conv(Op::Get("nn_conv2d"), args_38, attrs_38);
    // ONNX Node 39: Add (/layer4/layer4.0/Add)
    std::vector<Expr> args_39 = {node_37_Conv, node_38_Conv};
    Call node_39_Add(Op::Get("add"), args_39, AddAttrs::Create());
    // ONNX Node 40: Relu (/layer4/layer4.0/relu_1/Relu)
    std::vector<Expr> args_40 = {node_39_Add};
    Call node_40_Relu(Op::Get("nn_relu"), args_40, ReluAttrs::Create());
    // ONNX Node 41: Conv (/layer4/layer4.1/conv1/Conv)
    std::vector<Expr> args_41 = {node_40_Relu, const_38_onnx__Conv_247, const_39_onnx__Conv_248};
    Conv2DAttrs attrs_41 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 512, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_41_Conv(Op::Get("nn_conv2d"), args_41, attrs_41);
    // ONNX Node 42: Relu (/layer4/layer4.1/relu/Relu)
    std::vector<Expr> args_42 = {node_41_Conv};
    Call node_42_Relu(Op::Get("nn_relu"), args_42, ReluAttrs::Create());
    // ONNX Node 43: Conv (/layer4/layer4.1/conv2/Conv)
    std::vector<Expr> args_43 = {node_42_Relu, const_40_onnx__Conv_250, const_41_onnx__Conv_251};
    Conv2DAttrs attrs_43 = Conv2DAttrs::Create({1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 512, {3, 3}, "NCHW", "OIHW", "", "");
    Call node_43_Conv(Op::Get("nn_conv2d"), args_43, attrs_43);
    // ONNX Node 44: Add (/layer4/layer4.1/Add)
    std::vector<Expr> args_44 = {node_43_Conv, node_40_Relu};
    Call node_44_Add(Op::Get("add"), args_44, AddAttrs::Create());
    // ONNX Node 45: Relu (/layer4/layer4.1/relu_1/Relu)
    std::vector<Expr> args_45 = {node_44_Add};
    Call node_45_Relu(Op::Get("nn_relu"), args_45, ReluAttrs::Create());
    // ONNX Node 46: GlobalAveragePool (/avgpool/GlobalAveragePool)
    std::vector<Expr> args_46 = {node_45_Relu};
    Call node_46_GlobalAveragePool(Op::Get("nn_global_avg_pool2d"), args_46, GlobalAvgPool2DAttrs::Create());
    // ONNX Node 47: Flatten (/Flatten)
    std::vector<Expr> args_47 = {node_46_GlobalAveragePool};
    Call node_47_Flatten(Op::Get("nn_flatten"), args_47, FlattenAttrs::Create(1));
    // ONNX Node 48: Gemm (/fc/Gemm)
    std::vector<Expr> args_48 = {node_47_Flatten, const_0_fc_weight, const_1_fc_bias};
    Call node_48_Gemm(Op::Get("nn_gemm"), args_48, GemmAttrs::Create(1.0f, 1.0f, 0, 1));

    Expr out = node_48_Gemm;
    return Function(params, out);
}

}  // namespace

// 构造 ResNet18，执行类型推导与 lowering，并写出 Relay/TIR 诊断文件。
int main() {
    try {
        std::ofstream devnull("NUL");
        auto* old_cout_buf = std::cout.rdbuf(devnull.rdbuf());
        Function f = BuildResNet18Function();
        std::ofstream ofs("resnet18_ir_dump.txt", std::ios::out | std::ios::trunc);
        if (!ofs) throw std::runtime_error("Failed to open resnet18_ir_dump.txt for writing");

        DumpRelay(f, ofs);

        tir::PrimFunc pf = LowerToTIR(f)->prim_func;
        pf = tir::RunTIRPassPipeline(
            pf, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
        tir::pass::DumpPrimFunc(pf, ofs);
        std::cout.rdbuf(old_cout_buf);

        std::cout << "\nIR dump written to test/resnet18_ir_dump.txt" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
