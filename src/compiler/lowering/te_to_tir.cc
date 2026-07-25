/*! \file src/compiler/lowering/te_to_tir.cc
 * \brief Private TE DAG to TIR primitive lowering.
 */

#include "../internal/te_to_tir.h"
#include "kxc/pass/context.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/te/te.h"
#include "kxc/tir/expr.h"
#include "kxc/tir/visitor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kxc {
namespace relay {

namespace {

tir::PrimExpr FlattenIndex(const Array<tir::PrimExpr>& indices,
                           const Array<tir::PrimExpr>& shape) {
    if (shape.empty()) return tir::IntImm(0);
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Index rank mismatch during flattening");
    }
    tir::PrimExpr linear = indices[0];
    for (size_t index = 1; index < indices.size(); ++index) {
        linear = linear * shape[index] + indices[index];
    }
    return linear;
}

tir::PrimExpr MakeIdentityForReduce(te::ReduceType type,
                                    tir::DataType dtype) {
    if (type == te::ReduceType::kSum) {
        if (dtype.code == 2) {
            return dtype.bits == 64 ? tir::FloatImm(0.0, dtype)
                                    : tir::FloatImm(0.0f, dtype);
        }
        return tir::IntImm(0, dtype);
    }
    if (type == te::ReduceType::kMax) {
        if (dtype.code == 2) {
            return dtype.bits == 64
                       ? tir::FloatImm(
                             -std::numeric_limits<double>::infinity(), dtype)
                       : tir::FloatImm(
                             -std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::min(), dtype);
    }
    if (type == te::ReduceType::kMin) {
        if (dtype.code == 2) {
            return dtype.bits == 64
                       ? tir::FloatImm(
                             std::numeric_limits<double>::infinity(), dtype)
                       : tir::FloatImm(
                             std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::max(), dtype);
    }
    throw std::runtime_error("Unsupported reduce type");
}

// 递归收集 TE 表达式中的 ProducerLoad 依赖。
void FindProducerLoads(const tir::PrimExpr& expr, std::vector<te::Tensor>* deps) {
    if (!expr.defined()) return;
    if (auto* pl = expr.As<te::ProducerLoadNode>()) {
        deps->push_back(pl->tensor);
        for (const auto& idx : pl->indices) FindProducerLoads(idx, deps);
        return;
    }
    if (auto* red = expr.As<te::ReduceNode>()) {
        for (const auto& src : red->source) FindProducerLoads(src, deps);
        return;
    }
    if (auto* n = expr.As<tir::BinaryOpNode>()) {
        FindProducerLoads(n->a, deps);
        FindProducerLoads(n->b, deps);
        return;
    }
    if (auto* n = expr.As<tir::CallNode>()) {
        for (const auto& arg : n->args) FindProducerLoads(arg, deps);
        return;
    }
    if (auto* n = expr.As<tir::SelectNode>()) {
        FindProducerLoads(n->condition, deps);
        FindProducerLoads(n->true_value, deps);
        FindProducerLoads(n->false_value, deps);
        return;
    }
    if (auto* n = expr.As<tir::NotNode>()) {
        FindProducerLoads(n->value, deps);
        return;
    }
    if (auto* n = expr.As<tir::LoadNode>()) {
        FindProducerLoads(n->index, deps);
        if (n->predicate.defined()) FindProducerLoads(n->predicate, deps);
    }
}

// 对 TE 张量依赖图执行 DFS，生成生产者在前的拓扑顺序。
void CollectOpsDFS(const te::Tensor& t,
                   std::unordered_set<const Object*>* visited,
                   std::unordered_map<const Object*, std::vector<te::Tensor>>* op_outputs,
                   std::vector<te::Operation>* topo) {
    if (!t.defined() || !t->op.defined()) return;
    const Object* op_ptr = t->op.get();
    auto& outputs = (*op_outputs)[op_ptr];
    bool recorded = false;
    for (const auto& existing : outputs) {
        if (existing.get() == t.get()) {
            recorded = true;
            break;
        }
    }
    if (!recorded) outputs.push_back(t);
    if (visited->count(op_ptr)) return;

    if (auto* cop = t->op.As<te::ComputeOpNode>()) {
        for (const auto& body_expr : cop->body) {
            std::vector<te::Tensor> deps;
            FindProducerLoads(body_expr, &deps);
            for (const auto& dep : deps) {
                CollectOpsDFS(dep, visited, op_outputs, topo);
            }
        }
    }

    visited->insert(op_ptr);
    topo->push_back(t->op);
}

// 把 TE 表达式中的张量访问改写为 TIR buffer Load。
class ExprLowerer {
public:
    // 绑定张量到 TIR buffer 变量的映射。
    explicit ExprLowerer(const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor)
        : buffer_var_by_tensor_(buffer_var_by_tensor) {}

    // 递归降低纯表达式节点；Reduce 留给语句级 lowering。
    tir::PrimExpr Lower(const tir::PrimExpr& expr) const {
        if (!expr.defined()) return expr;

        if (auto* n = expr.As<te::ProducerLoadNode>()) {
            auto it = buffer_var_by_tensor_.find(n->tensor.get());
            if (it == buffer_var_by_tensor_.end()) {
                throw std::runtime_error("Missing buffer var for tensor in ProducerLoad");
            }
            Array<tir::PrimExpr> lowered_indices;
            for (const auto& idx : n->indices) lowered_indices.push_back(Lower(idx));
            tir::PrimExpr linear = FlattenIndex(lowered_indices, n->tensor->shape);
            return tir::Load(it->second, linear);
        }

        if (auto* n = expr.As<tir::AddNode>()) return tir::Add(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::SubNode>()) return tir::Sub(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MulNode>()) return tir::Mul(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::DivNode>()) return tir::Div(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::ModNode>()) return tir::Mod(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MinNode>()) return tir::Min(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::MaxNode>()) return tir::Max(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::EQNode>()) return tir::EQ(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::LTNode>()) return tir::LT(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::AndNode>()) return tir::And(Lower(n->a), Lower(n->b));
        if (auto* n = expr.As<tir::OrNode>()) return tir::Or(Lower(n->a), Lower(n->b));

        if (auto* n = expr.As<tir::NotNode>()) return tir::Not(Lower(n->value));
        if (auto* n = expr.As<tir::SelectNode>()) return tir::Select(Lower(n->condition), Lower(n->true_value), Lower(n->false_value));

        if (auto* n = expr.As<tir::CallNode>()) {
            Array<tir::PrimExpr> args;
            for (const auto& arg : n->args) args.push_back(Lower(arg));
            return tir::Call(n->dtype, n->name, args);
        }

        if (auto* n = expr.As<tir::LoadNode>()) {
            tir::PrimExpr pred = n->predicate.defined() ? Lower(n->predicate) : tir::PrimExpr();
            return tir::Load(n->buffer_var, Lower(n->index), pred);
        }

        if (expr.As<te::ReduceNode>()) {
            throw std::runtime_error("Reduce must be lowered at statement level");
        }

        return expr;
    }

private:
    const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor_;
};

// 用 ComputeOp 数据轴从内到外包裹串行循环。
tir::Stmt WrapDataLoops(const te::ComputeOpNode* op, tir::Stmt body) {
    for (int i = static_cast<int>(op->axis.size()) - 1; i >= 0; --i) {
        body = tir::For(op->axis[i], tir::IntImm(0), op->shape[i], tir::ForType::Serial, body);
    }
    return body;
}

// 将 TE ComputeOp 的指定 value_index 降为 TIR Store、数据循环和可选归约循环。
tir::Stmt LowerComputeStmt(const te::Tensor& out_tensor,
                           const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor,
                           const ExprLowerer& expr_lowerer) {
    auto* op = out_tensor->op.As<te::ComputeOpNode>();
    if (!op) {
        throw std::runtime_error("LowerComputeStmt expects ComputeOpNode");
    }
    if (op->body.empty()) {
        throw std::runtime_error("ComputeOp body is empty");
    }
    if (out_tensor->value_index < 0 ||
        static_cast<size_t>(out_tensor->value_index) >= op->body.size()) {
        throw std::runtime_error(
            "LowerComputeStmt tensor value_index is outside ComputeOp body");
    }

    auto out_it = buffer_var_by_tensor.find(out_tensor.get());
    if (out_it == buffer_var_by_tensor.end()) {
        throw std::runtime_error("Missing output buffer var for compute tensor");
    }
    tir::Var out_var = out_it->second;

    Array<tir::PrimExpr> data_indices;
    for (const auto& ax : op->axis) data_indices.push_back(ax);
    tir::PrimExpr out_index = FlattenIndex(data_indices, op->shape);

    tir::PrimExpr body_expr = op->body[static_cast<size_t>(out_tensor->value_index)];
    if (auto* red = body_expr.As<te::ReduceNode>()) {
        if (red->source.size() != 1) {
            throw std::runtime_error("Only single-source reduce is supported");
        }
        tir::PrimExpr init_value = MakeIdentityForReduce(red->reduce_type, out_tensor->dtype);
        tir::Stmt init_store = tir::Store(out_var, init_value, out_index);

        tir::PrimExpr src = expr_lowerer.Lower(red->source[0]);
        tir::PrimExpr old = tir::Load(out_var, out_index);
        tir::PrimExpr update_value;
        if (red->reduce_type == te::ReduceType::kSum) {
            update_value = old + src;
        } else if (red->reduce_type == te::ReduceType::kMax) {
            update_value = tir::Max(old, src);
        } else if (red->reduce_type == te::ReduceType::kMin) {
            update_value = tir::Min(old, src);
        } else {
            throw std::runtime_error("Unsupported reduce type in update");
        }
        tir::Stmt update_store = tir::Store(out_var, update_value, out_index);

        for (int i = static_cast<int>(red->axis.size()) - 1; i >= 0; --i) {
            const auto& rax = red->axis[i];
            update_store = tir::For(rax->var, rax->dom_min, rax->dom_extent, tir::ForType::Serial, update_store);
        }

        tir::Stmt seq = tir::SeqStmt({init_store, update_store});
        return WrapDataLoops(op, seq);
    }

    tir::PrimExpr lowered = expr_lowerer.Lower(body_expr);
    tir::Stmt store = tir::Store(out_var, lowered, out_index);
    return WrapDataLoops(op, store);
}

bool CheckedAddInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool CheckedSubInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
        (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs)) {
        return false;
    }
    *result = lhs - rhs;
    return true;
}

bool CheckedMulInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if (lhs == 0 || rhs == 0) {
        *result = 0;
        return true;
    }
    if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
        (rhs == -1 && lhs == std::numeric_limits<int64_t>::min())) {
        return false;
    }
    if (lhs > 0) {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs)) {
            return false;
        }
    } else if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
               (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool EvaluateStaticInt64(const tir::PrimExpr& expression, int64_t* result) {
    if (const auto* value = expression.As<tir::IntImmNode>()) {
        if (value->dtype.code != 0 && value->dtype.code != 1) return false;
        *result = value->value;
        return true;
    }
    const auto* binary = expression.As<tir::BinaryOpNode>();
    if (!binary) return false;
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!EvaluateStaticInt64(binary->a, &lhs) ||
        !EvaluateStaticInt64(binary->b, &rhs)) {
        return false;
    }
    if (expression.As<tir::AddNode>()) return CheckedAddInt64(lhs, rhs, result);
    if (expression.As<tir::SubNode>()) return CheckedSubInt64(lhs, rhs, result);
    if (expression.As<tir::MulNode>()) return CheckedMulInt64(lhs, rhs, result);
    if (expression.As<tir::DivNode>()) {
        if (rhs == 0 ||
            (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) {
            return false;
        }
        *result = lhs / rhs;
        return true;
    }
    if (expression.As<tir::ModNode>()) {
        if (rhs == 0 ||
            (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) {
            return false;
        }
        *result = lhs % rhs;
        return true;
    }
    if (expression.As<tir::MinNode>()) {
        *result = std::min(lhs, rhs);
        return true;
    }
    if (expression.As<tir::MaxNode>()) {
        *result = std::max(lhs, rhs);
        return true;
    }
    return false;
}

// 为公开输出生成唯一且可读的 TIR 参数名。
std::string MakeOutputVarName(const te::Tensor& tensor,
                              size_t output_index,
                              std::unordered_set<std::string>* used_names) {
    std::string base = tensor->name.empty() ? "output" : tensor->name;
    std::string name = base + "_out";
    if (output_index != 0) {
        name += "_" + std::to_string(output_index);
    }
    while (used_names && used_names->count(name)) {
        name += "_";
    }
    if (used_names) {
        used_names->insert(name);
    }
    return name;
}

}  // namespace

namespace internal {

bool EvaluateStaticLoweringInt64(const tir::PrimExpr& expression,
                                 int64_t* result) {
    return EvaluateStaticInt64(expression, result);
}

void ValidateStaticLoweringTensor(const Array<tir::PrimExpr>& shape,
                                  tir::DataType dtype,
                                  const std::string& context) {
    if (dtype.bits == 0 || dtype.lanes == 0) {
        throw std::invalid_argument(context +
                                    " has an invalid zero-width tensor dtype");
    }
    const size_t scalar_bytes =
        (static_cast<size_t>(dtype.bits) + 7U) / 8U;
    if (scalar_bytes > std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::overflow_error(context + " dtype byte width overflows size_t");
    }
    const size_t element_bytes = scalar_bytes * dtype.lanes;

    bool has_zero_extent = false;
    std::vector<int64_t> extents;
    extents.reserve(shape.size());
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        int64_t extent = 0;
        if (!EvaluateStaticInt64(shape[axis], &extent)) {
            throw std::invalid_argument(
                context + " requires an int64-representable static integer extent at axis " +
                std::to_string(axis));
        }
        if (extent < 0) {
            throw std::invalid_argument(
                context + " requires non-negative extent at axis " +
                std::to_string(axis));
        }
        if (extent > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(
                context + " extent exceeds the int32 iteration domain at axis " +
                std::to_string(axis));
        }
        has_zero_extent = has_zero_extent || extent == 0;
        extents.push_back(extent);
    }

    int64_t elements = has_zero_extent ? 0 : 1;
    if (!has_zero_extent) {
        for (int64_t extent : extents) {
            if (elements > std::numeric_limits<int64_t>::max() / extent) {
                throw std::overflow_error(
                    context + " row-major element count/flatten index overflows int64");
            }
            elements *= extent;
        }
    }
    if (elements != 0 &&
        elements > std::numeric_limits<int64_t>::max() /
                       static_cast<int64_t>(element_bytes)) {
        throw std::overflow_error(context + " tensor byte count overflows int64");
    }
    if (static_cast<uint64_t>(elements) >
        std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::overflow_error(context + " tensor byte count overflows size_t");
    }
}

LoweredFunction LowerTensorGraphToTIR(
    const Array<te::Tensor>& inputs,
    const std::vector<ConstantTensor>& constants,
    const Array<te::Tensor>& outputs,
    const PrimFuncIdentity& identity) {
    if (std::string(identity.symbol).empty()) {
        throw std::invalid_argument("PrimFunc identity requires a non-empty symbol");
    }
    if (identity.unit_id >= 0 &&
        (std::string(identity.operator_name).empty() ||
         identity.operator_schema_version <= 0 ||
         std::string(identity.structural_hash).empty())) {
        throw std::invalid_argument(
            "Per-unit PrimFunc identity requires operator and structural metadata");
    }
    if (outputs.empty()) {
        throw std::invalid_argument("TE-to-TIR lowering requires output tensors");
    }

    std::unordered_map<const Object*, size_t> output_index_by_tensor;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const te::Tensor& output = outputs[i];
        if (!output.defined()) {
            throw std::invalid_argument("TE-to-TIR output tensor is undefined");
        }
        if (!output_index_by_tensor.emplace(output.get(), i).second) {
            throw std::invalid_argument(
                "TE-to-TIR output tensors must be distinct logical values");
        }
        ValidateStaticLoweringTensor(
            output->shape, output->dtype,
            "TE-to-TIR output tensor '" + output->name + "'");
        if (!output->op.As<te::ComputeOpNode>()) {
            throw std::invalid_argument(
                "TE-to-TIR public outputs must be produced by ComputeOp");
        }
    }

    std::unordered_set<const Object*> visited_ops;
    std::unordered_map<const Object*, std::vector<te::Tensor>> op_output_tensors;
    std::vector<te::Operation> topo_ops;
    for (const auto& output : outputs) {
        CollectOpsDFS(output, &visited_ops, &op_output_tensors, &topo_ops);
    }
    for (auto& entry : op_output_tensors) {
        std::sort(entry.second.begin(), entry.second.end(),
                  [](const te::Tensor& lhs, const te::Tensor& rhs) {
                      return lhs->value_index < rhs->value_index;
                  });
        for (const auto& tensor : entry.second) {
            ValidateStaticLoweringTensor(
                tensor->shape, tensor->dtype,
                "TE-to-TIR tensor '" + tensor->name + "'");
        }
    }
    for (const auto& operation : topo_ops) {
        const auto* compute = operation.As<te::ComputeOpNode>();
        if (!compute || compute->reduce_axis.empty()) continue;
        Array<tir::PrimExpr> reduction_shape;
        for (const auto& axis : compute->reduce_axis) {
            reduction_shape.push_back(axis->dom_extent);
        }
        ValidateStaticLoweringTensor(
            reduction_shape, tir::DataType::UInt(8),
            "TE-to-TIR reduction iteration domain for '" + operation->name + "'");
    }

    Array<tir::Var> params;
    Map<tir::Var, tir::Buffer> buffer_map;
    std::unordered_map<const Object*, tir::Var> buffer_var_by_tensor;
    for (const auto& tensor : inputs) {
        if (!tensor.defined()) {
            throw std::invalid_argument("TE-to-TIR input tensor is undefined");
        }
        ValidateStaticLoweringTensor(
            tensor->shape, tensor->dtype,
            "TE-to-TIR input tensor '" + tensor->name + "'");
        tir::Var data_var(tensor->name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype, tensor->shape, {},
                           tir::IntImm(0), tensor->name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
    }
    const int64_t input_count = static_cast<int64_t>(inputs.size());

    Array<ConstantBinding> constant_bindings;
    Array<String> constant_keys;
    for (const auto& record : constants) {
        if (!record.tensor.defined() || !record.value.defined() ||
            std::string(record.key).empty()) {
            throw std::invalid_argument(
                "TE-to-TIR constant tensor record is incomplete");
        }
        const te::Tensor& tensor = record.tensor;
        ValidateStaticLoweringTensor(
            tensor->shape, tensor->dtype,
            "TE-to-TIR constant tensor '" + tensor->name + "'");
        tir::Var data_var(tensor->name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype, tensor->shape, {},
                           tir::IntImm(0), tensor->name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
        const int64_t param_index = static_cast<int64_t>(params.size() - 1);
        constant_bindings.push_back(
            ConstantBinding(record.key, record.value, param_index));
        constant_keys.push_back(record.key);
    }
    const int64_t constant_count = static_cast<int64_t>(constants.size());
    const int64_t output_param_start = input_count + constant_count;

    std::unordered_set<const Object*> public_outputs;
    std::unordered_set<std::string> used_output_names;
    for (size_t index = 0; index < outputs.size(); ++index) {
        const te::Tensor& tensor = outputs[index];
        public_outputs.insert(tensor.get());
        const std::string name =
            MakeOutputVarName(tensor, index, &used_output_names);
        tir::Var data_var(name, tensor->dtype);
        tir::Buffer buffer(data_var, tensor->dtype, tensor->shape, {},
                           tir::IntImm(0), name, 0, 0);
        params.push_back(data_var);
        buffer_map.Set(data_var, buffer);
        buffer_var_by_tensor[tensor.get()] = data_var;
    }

    std::vector<te::Tensor> intermediates;
    for (const auto& operation : topo_ops) {
        if (!operation.As<te::ComputeOpNode>()) continue;
        const auto tensors_it = op_output_tensors.find(operation.get());
        if (tensors_it == op_output_tensors.end()) continue;
        for (const auto& tensor : tensors_it->second) {
            if (public_outputs.count(tensor.get()) != 0) continue;
            tir::Var local_var(tensor->name, tensor->dtype);
            buffer_var_by_tensor[tensor.get()] = local_var;
            intermediates.push_back(tensor);
        }
    }

    ExprLowerer expr_lowerer(buffer_var_by_tensor);
    Array<tir::Stmt> compute_sequence;
    for (const auto& operation : topo_ops) {
        if (operation.As<te::PlaceholderOpNode>()) continue;
        const auto tensors_it = op_output_tensors.find(operation.get());
        if (tensors_it == op_output_tensors.end() || tensors_it->second.empty()) {
            throw std::runtime_error(
                "Missing tensor outputs during TE statement lowering");
        }
        for (const auto& tensor : tensors_it->second) {
            if (!tensor->op.As<te::ComputeOpNode>()) {
                throw std::runtime_error(
                    "Unsupported non-compute operation in TIR lowering");
            }
            try {
                compute_sequence.push_back(
                    LowerComputeStmt(tensor, buffer_var_by_tensor, expr_lowerer));
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "LowerComputeStmt failed for tensor '" + tensor->name +
                    "': " + error.what());
            }
        }
    }

    tir::Stmt body;
    if (compute_sequence.size() == 1) {
        body = compute_sequence[0];
    } else if (!compute_sequence.empty()) {
        body = tir::SeqStmt(compute_sequence);
    }
    for (auto it = intermediates.rbegin(); it != intermediates.rend(); ++it) {
        const te::Tensor& tensor = *it;
        body = tir::Allocate(buffer_var_by_tensor.at(tensor.get()), tensor->dtype,
                             tensor->shape,
                             tir::IntImm(1, tir::DataType::Bool()), body);
    }

    Map<String, ObjectRef> attrs;
    attrs.Set("global_symbol", identity.symbol);
    attrs.Set("tir.noalias", tir::IntImm(1, tir::DataType::Bool()));
    attrs.Set("kxc.input_count",
              tir::IntImm(input_count, tir::DataType::Int(64)));
    attrs.Set("kxc.constant_count",
              tir::IntImm(constant_count, tir::DataType::Int(64)));
    attrs.Set("kxc.output_count",
              tir::IntImm(static_cast<int64_t>(outputs.size()),
                          tir::DataType::Int(64)));
    attrs.Set("kxc.output_param_start",
              tir::IntImm(output_param_start, tir::DataType::Int(64)));
    attrs.Set("kxc.constant_keys", codegen::KernelConstantKeys(constant_keys));
    if (identity.unit_id >= 0) {
        const String operator_identity(
            std::string(identity.operator_name) + "@v" +
            std::to_string(identity.operator_schema_version));
        attrs.Set("kxc.unit_id",
                  tir::IntImm(identity.unit_id, tir::DataType::Int(64)));
        attrs.Set("kxc.operator_name", identity.operator_name);
        attrs.Set("kxc.operator_schema_version",
                  tir::IntImm(identity.operator_schema_version,
                              tir::DataType::Int(64)));
        attrs.Set("kxc.operator_identity", operator_identity);
        attrs.Set("kxc.structural_hash", identity.structural_hash);
    }
    attrs = tir::AttachPassContextAttrs(attrs, PassContext::Current());
    return LoweredFunction(tir::PrimFunc(params, body, buffer_map, attrs),
                           constant_bindings);
}

}  // namespace internal

}  // namespace relay
}  // namespace kxc
