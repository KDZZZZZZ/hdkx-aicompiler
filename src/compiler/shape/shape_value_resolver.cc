/*! \file src/compiler/shape/shape_value_resolver.cc
 * \brief M3 受限形状值解析：链式证明、折叠与目标表达式投影。
 */

#include "shape_value_resolver.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "kxc/relay/relay.h"
#include "kxc/runtime/ndarray.h"

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {
namespace shape_resolution {

namespace {

using DimExpr = kxc::shape::experimental::v1::DimExpr;
using BindingSet = kxc::shape::experimental::v1::BindingSet;

// 受限形状表达式的最大元素数（声明子集；Select 链与编码按此封顶）。
constexpr size_t kMaxShapeExprElements = 16;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("RestrictedShapeValueResolver: " + message);
}

enum class ValueKind { kData, kShapeValue };

struct NodeInfo final {
    ValueKind kind{ValueKind::kData};
    std::vector<DimExpr> dims;      // kData：符号维表达式
    std::vector<DimExpr> elements;  // kShapeValue：元素表达式（相对根源）
    const Object* source{nullptr};  // kShapeValue：根数据节点（原树）
};

// 读取 rank-1 int32/int64 常量张量负载（Gather 索引）。
std::vector<int64_t> ReadConstantVector(const kxc::ConstantNode* constant,
                                        const std::string& context) {
    if (!constant || !constant->data.defined()) {
        Reject(context + " requires a defined constant tensor payload");
    }
    const auto& shape = constant->data->shape_storage;
    if (shape.size() != 1) {
        Reject(context + " constant must be rank 1");
    }
    const DLDataType dtype = constant->data->dl_tensor.dtype;
    if (dtype.lanes != 1 ||
        !(dtype.code == kDLInt && (dtype.bits == 32 || dtype.bits == 64))) {
        Reject(context + " constant dtype must be int32 or int64");
    }
    std::vector<int64_t> values;
    if (shape[0] > 0) {
        values.resize(static_cast<size_t>(shape[0]));
        constant->data.CopyToBytes(values.data(),
                                   values.size() * sizeof(int64_t));
    }
    return values;
}

bool SameDims(const std::vector<DimExpr>& left, const std::vector<DimExpr>& right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (!(left[i] == right[i])) return false;
    }
    return true;
}

DimExpr ProductOf(const std::vector<DimExpr>& terms) {
    return DimExpr::Mul(terms);
}

// 把元素表达式编码为 (kinds/values/axes)：符号 → (1,0,axis)，常量 → (0,v,0)。
// reference_dims 是编码基准（shape_expr 源维 / reshape / expand 数据维）。
EncodedExpr EncodeElements(const std::vector<DimExpr>& elements,
                           const std::vector<DimExpr>& reference_dims,
                           const std::string& context) {
    EncodedExpr encoded;
    for (const DimExpr& element : elements) {
        if (element.kind() == DimExpr::Kind::kConst) {
            const int64_t value = element.Evaluate(BindingSet());
            if (value < 0) {
                Reject(context + " element must be >= 0; dynamic -1 inference is "
                                 "outside the restricted bounded subset");
            }
            encoded.kinds.push_back(kExprKindConst);
            encoded.values.push_back(value);
            encoded.axes.push_back(0);
            continue;
        }
        if (element.kind() != DimExpr::Kind::kSymbol) {
            Reject(context + " element expressions must be Const or direct Symbol");
        }
        int matched_axis = -1;
        for (size_t axis = 0; axis < reference_dims.size(); ++axis) {
            if (element == reference_dims[axis]) {
                if (matched_axis >= 0) {
                    Reject(context +
                           " a symbolic element matches multiple reference axes");
                }
                matched_axis = static_cast<int>(axis);
            }
        }
        if (matched_axis < 0) {
            Reject(context +
                   " a symbolic element is not rooted at the reference tensor");
        }
        encoded.kinds.push_back(kExprKindInputAxis);
        encoded.values.push_back(0);
        encoded.axes.push_back(matched_axis);
    }
    return encoded;
}

int NormalizeAxis(const std::string& context, int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        Reject(context + " axis out of range");
    }
    return static_cast<int>(axis);
}

class Resolver final {
public:
    Resolver(const Function& snapshot,
             const std::vector<InputAxisSymbol>& symbols)
        : snapshot_(snapshot) {
        if (!snapshot.defined() || snapshot_->params.empty() ||
            !snapshot_->body.defined()) {
            Reject("the representative snapshot must have params and a body");
        }
        for (size_t index = 0; index < snapshot_->params.size(); ++index) {
            param_index_.emplace(snapshot_->params[index].get(), index);
        }
        for (const InputAxisSymbol& symbol : symbols) {
            axis_symbols_.emplace(
                std::make_pair(symbol.parameter_index, symbol.axis),
                symbol.symbol);
        }
    }

    Resolution Run() {
        const NodeInfo body = Resolve(snapshot_->body);
        (void)body;
        Resolution result;
        result.rewritten =
            Function(snapshot_->params, rewritten_.at(snapshot_->body.get()));
        WalkRewritten(result.rewritten->body, &result);
        return result;
    }

private:
    const Function& snapshot_;
    std::map<const Object*, size_t> param_index_;
    std::map<std::pair<size_t, size_t>, std::string> axis_symbols_;
    std::map<const Object*, NodeInfo> info_;
    std::map<const Object*, NodeInfo> rewritten_info_;
    std::map<const Object*, Expr> rewritten_;
    std::map<const Object*, EncodedExpr> value_expr_overrides_;

    std::vector<DimExpr> ParameterDims(const kxc::VarNode* parameter) const {
        const auto* type = parameter->type_annotation.As<kxc::TensorTypeNode>();
        if (!type) Reject("every parameter must carry a TensorType annotation");
        const size_t index = param_index_.at(parameter);
        std::vector<DimExpr> dims;
        dims.reserve(type->shape.size());
        for (size_t axis = 0; axis < type->shape.size(); ++axis) {
            const auto found = axis_symbols_.find(std::make_pair(index, axis));
            if (found != axis_symbols_.end()) {
                dims.push_back(DimExpr::Symbol(found->second));
            } else {
                dims.push_back(DimExpr::Const(type->shape[axis]));
            }
        }
        return dims;
    }

    static std::string OpName(const kxc::CallNode* call) {
        const auto* op = call->op.As<kxc::relay::OpNode>();
        if (!op) Reject("only calls to registered Relay operators are supported");
        return op->name;
    }

    static size_t ExpectedArity(const std::string& name) {
        if (name == "add" || name == "mul" || name == "gather" ||
            name == "concatenate" || name == "reshape_dynamic" || name == "expand_dynamic") {
            return 2;
        }
        return 1;
    }

    static void RequireData(const NodeInfo& info, const std::string& context) {
        if (info.kind != ValueKind::kData) {
            Reject(context + " must be a data tensor, not a shape value");
        }
    }

    static void RequireShapeValue(const NodeInfo& info, const std::string& context) {
        if (info.kind != ValueKind::kShapeValue) {
            Reject(context +
                   " must be a restricted shape value (shape_of / folded chain)");
        }
    }

    // 属性策略：attr-free 算子不得携带属性；受控算子（reshape_dynamic/
    // expand）必须无调用方属性——目标表达式由本解析器唯一写入。
    static void RequireNoCallerAttrs(const kxc::CallNode* call,
                                     const std::string& name) {
        if (call->attrs.defined()) {
            Reject(name + " must arrive without caller attrs; the restricted "
                           "preparation is the only attrs authority");
        }
    }

    NodeInfo Resolve(const Expr& expr) {
        if (!expr.defined()) Reject("expression is undefined");
        const Object* node = expr.get();
        const auto memo = info_.find(node);
        if (memo != info_.end()) return memo->second;

        if (const auto* var = expr.As<kxc::VarNode>()) {
            if (param_index_.find(node) == param_index_.end()) {
                Reject("graph references a non-parameter variable");
            }
            NodeInfo info;
            info.kind = ValueKind::kData;
            info.dims = ParameterDims(var);
            info_[node] = info;
            rewritten_[node] = expr;
            rewritten_info_[node] = info;
            return info;
        }
        if (expr.As<kxc::ConstantNode>()) {
            Reject("bounded restricted graphs cannot contain constant tensors; "
                   "gather indices are consumed by the chain fold");
        }

        const auto* call = expr.As<kxc::CallNode>();
        if (!call) {
            Reject("only a tree of restricted calls is supported");
        }
        const std::string name = OpName(call);
        if (call->args.size() != ExpectedArity(name)) {
            Reject("operator " + name + " arity mismatch");
        }

        NodeInfo info;
        if (name == "add" || name == "mul") {
            const NodeInfo& lhs = Resolve(call->args[0]);
            const NodeInfo& rhs = Resolve(call->args[1]);
            RequireData(lhs, name + " lhs");
            RequireData(rhs, name + " rhs");
            if (!SameDims(lhs.dims, rhs.dims)) {
                Reject("restricted " + name + " requires exact equal shapes");
            }
            info.kind = ValueKind::kData;
            info.dims = lhs.dims;
            rewritten_[node] =
                RebuildCall(name, call, kxc::relay::Attrs(),
                            {rewritten_.at(call->args[0].get()),
                             rewritten_.at(call->args[1].get())});
        } else if (name == "relu" || name == "nn_relu" || name == "sqrt") {
            const NodeInfo& input = Resolve(call->args[0]);
            RequireData(input, name + " input");
            info.kind = ValueKind::kData;
            info.dims = input.dims;
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::Attrs(),
            {rewritten_.at(call->args[0].get())});
        } else if (name == "shape_of") {
            const NodeInfo& source = Resolve(call->args[0]);
            RequireData(source, "shape_of source");
            if (source.dims.empty()) {
                Reject("shape_of requires a rank >= 1 source");
            }
            info.kind = ValueKind::kShapeValue;
            info.elements = source.dims;
            info.source = call->args[0].get();
            rewritten_[node] = RebuildCall(
                name, call, kxc::relay::Attrs(),
            {rewritten_.at(call->args[0].get())});
        } else if (name == "gather") {
            info = ResolveGather(call, node);
        } else if (name == "concatenate") {
            info = ResolveConcatenate(call, node);
        } else if (name == "reshape_dynamic") {
            info = ResolveReshapeDynamic(call, node);
        } else if (name == "expand_dynamic") {
            info = ResolveExpand(call, node);
        } else if (name == "squeeze") {
            info = ResolveSqueeze(call, node);
        } else if (name == "unsqueeze") {
            info = ResolveUnsqueeze(call, node);
        } else {
            Reject("unsupported Relay operation in the restricted shape-value "
                   "walk: " + name);
        }

        info_[node] = info;
        rewritten_info_[rewritten_[node].get()] = info;
        return info;
    }

    Expr RebuildCall(const std::string& name, const kxc::CallNode* call,
                     kxc::relay::Attrs attrs, kxc::Array<kxc::Expr> args) {
        if (!attrs.defined()) {
            RequireNoCallerAttrs(call, name);
            return kxc::Call(kxc::relay::Op::Get(name), std::move(args));
        }
        return kxc::Call(kxc::relay::Op::Get(name), std::move(args),
                         std::move(attrs));
    }

    // 链折叠：以根源为唯一输入生成 shape_expr，记录 value 表达式覆盖。
    NodeInfo FoldShapeExpr(const kxc::CallNode* call, const Object* node,
                           const Object* source,
                           std::vector<DimExpr> elements) {
        const NodeInfo& source_info = info_.at(source);
        if (elements.size() > kMaxShapeExprElements) {
            Reject("folded shape value exceeds the declared element subset");
        }
        const EncodedExpr encoded = EncodeElements(
            elements, source_info.dims, "shape_expr fold");
        Expr folded = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {rewritten_.at(source)},
            kxc::relay::ShapeExprAttrs::Create(encoded.kinds, encoded.values,
                                               encoded.axes));
        rewritten_[node] = folded;
        value_expr_overrides_[folded.get()] = encoded;
        NodeInfo info;
        info.kind = ValueKind::kShapeValue;
        info.elements = std::move(elements);
        info.source = source;
        return info;
    }

    // Gather：仅形状值数据 + 常量索引 + 轴 0；越界拒绝；折叠为 shape_expr。
    NodeInfo ResolveGather(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        // 形状值恒为 rank-1 int64 向量，其长度即元素数；这里只需确认来源
        // 是形状值，长度由下面的索引边界检查裁决。
        RequireShapeValue(data, "gather data");
        const auto* gather_attrs = call->attrs.As<kxc::relay::GatherAttrsNode>();
        if (!gather_attrs) {
            Reject("gather requires GatherAttrs");
        }
        if (NormalizeAxis("gather", gather_attrs->axis, 1) != 0) {
            Reject("bounded gather is restricted to axis 0");
        }
        const std::vector<int64_t> raw =
            ReadConstantVector(call->args[1].As<kxc::ConstantNode>(),
                               "gather indices");
        const int64_t length = static_cast<int64_t>(data.elements.size());
        std::vector<DimExpr> elements;
        elements.reserve(raw.size());
        for (int64_t index : raw) {
            if (index < 0) index += length;
            if (index < 0 || index >= length) {
                Reject("gather index is out of bounds for the shape value");
            }
            elements.push_back(data.elements[static_cast<size_t>(index)]);
        }
        return FoldShapeExpr(call, node, data.source, std::move(elements));
    }

    NodeInfo ResolveConcatenate(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& lhs = Resolve(call->args[0]);
        const NodeInfo& rhs = Resolve(call->args[1]);
        // 两侧都是 rank-1 形状值；长度可任意，拼接结果长度是二者之和，
        // 上限由 FoldShapeExpr 的元素子集检查裁决。
        RequireShapeValue(lhs, "concatenate lhs");
        RequireShapeValue(rhs, "concatenate rhs");
        const auto* concatenate_attrs =
            call->attrs.As<kxc::relay::ConcatenateAttrsNode>();
        if (!concatenate_attrs) {
            Reject("concatenate requires ConcatenateAttrs");
        }
        if (NormalizeAxis("concatenate", concatenate_attrs->axis, 1) != 0) {
            Reject("bounded concatenate is restricted to axis 0");
        }
        if (lhs.source != rhs.source) {
            Reject("shape chains must stay rooted at a single source tensor");
        }
        std::vector<DimExpr> elements = lhs.elements;
        elements.insert(elements.end(), rhs.elements.begin(), rhs.elements.end());
        return FoldShapeExpr(call, node, lhs.source, std::move(elements));
    }

    // 控制目标投影：opset-17 0-copy 解析；-1 与符号歧义拒绝；元素总数以
    // DimExpr canonical 乘积等值证明。
    NodeInfo ResolveReshapeDynamic(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        const NodeInfo& control = Resolve(call->args[1]);
        RequireData(data, "reshape_dynamic data");
        RequireShapeValue(control, "reshape_dynamic control");
        RequireNoCallerAttrs(call, "reshape_dynamic");
        std::vector<DimExpr> target;
        target.reserve(control.elements.size());
        for (size_t i = 0; i < control.elements.size(); ++i) {
            const DimExpr& element = control.elements[i];
            if (element.kind() == DimExpr::Kind::kConst) {
                const int64_t value = element.Evaluate(BindingSet());
                if (value == 0) {
                    // opset 17（allowzero=0）：0 → 复制数据同位置维。
                    if (i >= data.dims.size()) {
                        Reject("reshape 0-dim copy index exceeds the data rank");
                    }
                    target.push_back(data.dims[i]);
                    continue;
                }
                if (value < 0) {
                    Reject("reshape -1 inference is outside the restricted "
                           "bounded subset");
                }
                target.push_back(element);
                continue;
            }
            int matched = -1;
            for (size_t axis = 0; axis < data.dims.size(); ++axis) {
                if (element == data.dims[axis]) {
                    if (matched >= 0) {
                        Reject("reshape target symbol matches multiple data axes");
                    }
                    matched = static_cast<int>(axis);
                }
            }
            if (matched < 0) {
                Reject("reshape control must stay rooted at the reshape data");
            }
            target.push_back(element);
        }
        if (target.empty()) {
            Reject("reshape_dynamic requires a rank >= 1 target");
        }
        if (!(ProductOf(data.dims) == ProductOf(target))) {
            Reject("reshape element count cannot be proven equal");
        }
        const EncodedExpr encoded =
            EncodeElements(target, data.dims, "reshape_dynamic target");
        rewritten_[node] = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"),
            {rewritten_.at(call->args[0].get()),
             rewritten_.at(call->args[1].get())},
            kxc::relay::ReshapeDynamicAttrs::Create(encoded.kinds, encoded.values,
                                                    encoded.axes));
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(target);
        return info;
    }

    NodeInfo ResolveExpand(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& data = Resolve(call->args[0]);
        const NodeInfo& control = Resolve(call->args[1]);
        RequireData(data, "expand data");
        RequireShapeValue(control, "expand control");
        RequireNoCallerAttrs(call, "expand_dynamic");
        if (control.elements.size() != data.dims.size()) {
            Reject("expand target length must equal the data rank");
        }
        for (size_t axis = 0; axis < control.elements.size(); ++axis) {
            const DimExpr& data_dim = data.dims[axis];
            const DimExpr& target_dim = control.elements[axis];
            if (data_dim == target_dim || data_dim == DimExpr::Const(1)) {
                continue;
            }
            Reject("expand cannot broadcast axis " + std::to_string(axis) +
                   " under the restricted proof rules");
        }
        const EncodedExpr encoded =
            EncodeElements(control.elements, data.dims, "expand target");
        rewritten_[node] = kxc::Call(
            kxc::relay::Op::Get("expand_dynamic"),
            {rewritten_.at(call->args[0].get()),
             rewritten_.at(call->args[1].get())},
            kxc::relay::ExpandDynamicAttrs::Create(encoded.kinds, encoded.values,
                                            encoded.axes));
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = control.elements;
        return info;
    }

    NodeInfo ResolveSqueeze(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& input = Resolve(call->args[0]);
        RequireData(input, "squeeze input");
        const auto* squeeze_attrs = call->attrs.As<kxc::relay::SqueezeAttrsNode>();
        if (!squeeze_attrs || squeeze_attrs->axes.empty()) {
            Reject("squeeze requires explicit axes");
        }
        const int64_t rank = static_cast<int64_t>(input.dims.size());
        if (rank < 1) {
            Reject("squeeze requires data rank >= 1");
        }
        std::vector<bool> removed(static_cast<size_t>(rank), false);
        for (int64_t raw_axis : squeeze_attrs->axes) {
            const int axis = NormalizeAxis("squeeze", raw_axis, rank);
            if (removed[static_cast<size_t>(axis)]) {
                Reject("squeeze duplicate axis");
            }
            removed[static_cast<size_t>(axis)] = true;
            const DimExpr& dim = input.dims[static_cast<size_t>(axis)];
            if (!(dim == DimExpr::Const(1))) {
                Reject("squeeze requires each removed axis to be provably 1");
            }
        }
        std::vector<DimExpr> dims;
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (!removed[static_cast<size_t>(axis)]) {
                dims.push_back(input.dims[static_cast<size_t>(axis)]);
            }
        }
        if (dims.empty()) {
            Reject("squeeze must keep at least one axis");
        }
        rewritten_[node] = RebuildCall(
            "squeeze", call, call->attrs,
            {rewritten_.at(call->args[0].get())});
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(dims);
        return info;
    }

    NodeInfo ResolveUnsqueeze(const kxc::CallNode* call, const Object* node) {
        const NodeInfo& input = Resolve(call->args[0]);
        RequireData(input, "unsqueeze input");
        const auto* unsqueeze_attrs =
            call->attrs.As<kxc::relay::UnsqueezeAttrsNode>();
        if (!unsqueeze_attrs || unsqueeze_attrs->axes.empty()) {
            Reject("unsqueeze requires explicit axes");
        }
        const int64_t input_rank = static_cast<int64_t>(input.dims.size());
        const int64_t new_rank =
            input_rank + static_cast<int64_t>(unsqueeze_attrs->axes.size());
        if (new_rank > 8) {
            Reject("unsqueeze output rank exceeds the declared subset");
        }
        std::vector<bool> placed(static_cast<size_t>(new_rank), false);
        for (int64_t raw_axis : unsqueeze_attrs->axes) {
            const int axis = NormalizeAxis("unsqueeze", raw_axis, new_rank);
            if (placed[static_cast<size_t>(axis)]) {
                Reject("unsqueeze duplicate axis");
            }
            placed[static_cast<size_t>(axis)] = true;
        }
        std::vector<DimExpr> dims(static_cast<size_t>(new_rank), DimExpr::Const(1));
        size_t next = 0;
        for (size_t axis = 0; axis < dims.size(); ++axis) {
            if (placed[axis]) continue;
            dims[axis] = input.dims[next++];
        }
        rewritten_[node] = RebuildCall(
            "unsqueeze", call, call->attrs,
            {rewritten_.at(call->args[0].get())});
        NodeInfo info;
        info.kind = ValueKind::kData;
        info.dims = std::move(dims);
        return info;
    }

    // 阶段二：post-order 走折叠后树，按 BuildValueGraph 顺序分配 value id，
    // 并收集单元算子名、attrs 与 value 表达式覆盖。
    void WalkRewritten(const Expr& expr, Resolution* result) {
        if (!expr.defined()) Reject("rewritten expression is undefined");
        const Object* node = expr.get();
        if (expr.As<kxc::VarNode>()) {
            const NodeInfo info = rewritten_info_.at(node);
            result->value_dimensions["value." +
                                     std::to_string(param_index_.at(node))] =
                info.dims;
            return;
        }
        if (expr.As<kxc::ConstantNode>()) {
            Reject("the rewritten restricted graph cannot contain constants");
        }
        const auto* call = expr.As<kxc::CallNode>();
        if (!call) {
            Reject("the rewritten restricted graph must stay a call tree");
        }
        for (const Expr& argument : call->args) {
            WalkRewritten(argument, result);
        }
        const size_t value_id =
            param_index_.size() + result->registry_operations.size();
        const std::string name = OpName(call);
        result->registry_operations.push_back(name);
        result->unit_attrs.push_back(call->attrs);
        const auto override_found = value_expr_overrides_.find(node);
        if (override_found != value_expr_overrides_.end()) {
            result->unit_value_expressions.push_back(override_found->second);
        } else {
            result->unit_value_expressions.push_back(std::nullopt);
        }
        result->value_dimensions["value." + std::to_string(value_id)] =
            ValueTensorDims(rewritten_info_.at(node));
    }

    // 单元产出的张量维度：数据单元取其符号维；形状值单元是
    // int64[元素数] 行向量，长度静态已知。
    static std::vector<DimExpr> ValueTensorDims(const NodeInfo& info) {
        if (info.kind == ValueKind::kShapeValue) {
            return {DimExpr::Const(
                static_cast<int64_t>(info.elements.size()))};
        }
        return info.dims;
    }
};

}  // namespace

Resolution ResolveShapeValues(
    const Function& snapshot,
    const std::vector<InputAxisSymbol>& parameter_symbols) {
    Resolver resolver(snapshot, parameter_symbols);
    return resolver.Run();
}

}  // namespace shape_resolution
}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
