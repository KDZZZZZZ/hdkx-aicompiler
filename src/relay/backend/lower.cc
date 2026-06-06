/*! \file src/relay/backend/lower.cc
 * \brief 实现 Relay/TE 到 TIR 的 lowering 主流程。
 */

#include "relay/transforms/lower.h"
#include "relay/transforms/infer_type.h"
#include "relay/op_attr_types.h"
#include "relay/op.h"
#include "base/pass.h"
#include "base/profiling.h"
#include "te/te.h"
#include "relay/pass/print_ir.h"
#include "tir/pass/print_ir.h"
#include "tir/expr.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kxc {
namespace relay {

namespace {

bool IsDeviceCommunicationOpName(const std::string& op_name) {
    return op_name.rfind("device.", 0) == 0;
}

tir::DataType DTypeFromString(const std::string& dtype) {
    if (dtype == "float32") return tir::DataType::Float(32);
    if (dtype == "float64") return tir::DataType::Float(64);
    if (dtype == "int32") return tir::DataType::Int(32);
    if (dtype == "int64") return tir::DataType::Int(64);
    if (dtype == "int8") return tir::DataType::Int(8);
    if (dtype == "uint8") return tir::DataType::UInt(8);
    if (dtype == "bool") return tir::DataType::Bool();
    throw std::runtime_error("Unsupported dtype string: " + dtype);
}

tir::DataType DTypeFromDL(const DLDataType& dl_dtype) {
    if (dl_dtype.code == kDLFloat) return tir::DataType::Float(dl_dtype.bits, dl_dtype.lanes);
    if (dl_dtype.code == kDLInt) return tir::DataType::Int(dl_dtype.bits, dl_dtype.lanes);
    if (dl_dtype.code == kDLUint) return tir::DataType::UInt(dl_dtype.bits, dl_dtype.lanes);
    throw std::runtime_error("Unsupported DLDataType code in constant");
}

Array<tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type) {
    Array<tir::PrimExpr> shape;
    for (const auto dim : type->shape) {
        shape.push_back(tir::IntImm(dim, tir::DataType::Int(64)));
    }
    return shape;
}

tir::PrimExpr FlattenIndex(const Array<tir::PrimExpr>& indices, const Array<tir::PrimExpr>& shape) {
    if (shape.empty()) {
        return tir::IntImm(0);
    }
    if (indices.size() != shape.size()) {
        throw std::runtime_error("Index rank mismatch during flattening");
    }
    tir::PrimExpr linear = indices[0];
    for (size_t i = 1; i < indices.size(); ++i) {
        linear = linear * shape[i] + indices[i];
    }
    return linear;
}

tir::PrimExpr MakeIdentityForReduce(te::ReduceType rtype, tir::DataType dtype) {
    if (rtype == te::ReduceType::kSum) {
        if (dtype.code == 2) {
            return dtype.bits == 64 ? tir::FloatImm(0.0, dtype) : tir::FloatImm(0.0f, dtype);
        }
        return tir::IntImm(0, dtype);
    }
    if (rtype == te::ReduceType::kMax) {
        if (dtype.code == 2) {
            if (dtype.bits == 64) return tir::FloatImm(-std::numeric_limits<double>::infinity(), dtype);
            return tir::FloatImm(-std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::min(), dtype);
    }
    if (rtype == te::ReduceType::kMin) {
        if (dtype.code == 2) {
            if (dtype.bits == 64) return tir::FloatImm(std::numeric_limits<double>::infinity(), dtype);
            return tir::FloatImm(std::numeric_limits<float>::infinity(), dtype);
        }
        return tir::IntImm(std::numeric_limits<int64_t>::max(), dtype);
    }
    throw std::runtime_error("Unsupported reduce type");
}

std::string RelayNodeKind(const Expr& expr) {
    if (!expr.defined()) return "<undefined>";
    if (expr.As<VarNode>()) return "Var";
    if (expr.As<ConstantNode>()) return "Constant";
    if (expr.As<CallNode>()) return "Call";
    if (expr.As<FunctionNode>()) return "Function";
    if (expr.As<TupleNode>()) return "Tuple";
    if (expr.As<TupleGetItemNode>()) return "TupleGetItem";
    if (expr.As<IfNode>()) return "If";
    if (expr.As<LetNode>()) return "Let";
    if (expr.As<OpNode>()) return "Op";
    return "<unknown>";
}

Array<te::Tensor> InvokeRelayToTE(const OpNode* op_node,
                                  const Attrs& attrs,
                                  const Array<te::Tensor>& inputs,
                                  const kxc::Type& out_type) {
    if (!out_type.defined()) {
        throw std::runtime_error("LowerToTIR requires checked_type for op: " +
                                 op_node->name);
    }
    if (out_type.As<TensorTypeNode>()) {
        auto it = op_node->attrs.find("FRelayToTE");
        if (it == op_node->attrs.end()) {
            throw std::runtime_error("No FRelayToTE registered for op: " + op_node->name);
        }
        auto* lower_ptr = std::any_cast<FRelayToTE>(&it->second);
        if (!lower_ptr) {
            throw std::runtime_error("Bad FRelayToTE type for op: " + op_node->name);
        }
        te::Tensor out = (*lower_ptr)(attrs, inputs, out_type);
        return {out};
    }

    if (const auto* tuple = out_type.As<TupleTypeNode>()) {
        auto it = op_node->attrs.find("FRelayToTEMulti");
        if (it == op_node->attrs.end()) {
            throw std::runtime_error("No FRelayToTEMulti registered for tuple-output op: " +
                                     op_node->name + ", got " + TypeToString(out_type));
        }
        auto* lower_ptr = std::any_cast<FRelayToTEMulti>(&it->second);
        if (!lower_ptr) {
            throw std::runtime_error("Bad FRelayToTEMulti type for op: " + op_node->name);
        }
        Array<te::Tensor> outputs = (*lower_ptr)(attrs, inputs, out_type);
        if (outputs.size() != tuple->fields.size()) {
            throw std::runtime_error("FRelayToTEMulti output count mismatch for op: " +
                                     op_node->name + ", expected " +
                                     std::to_string(tuple->fields.size()) + ", got " +
                                     std::to_string(outputs.size()));
        }
        return outputs;
    }

    throw std::runtime_error("LowerToTIR requires TensorType or TupleType call output for op: " +
                             op_node->name + ", got " + TypeToString(out_type));
}

class RelayToTEConverter : public RelayPassFunctor<Array<te::Tensor>> {
public:
    explicit RelayToTEConverter(const Function& func) {
        for (const auto& param : func->params) {
            const TensorTypeNode* ttype = param->type_annotation.As<TensorTypeNode>();
            if (!ttype) {
                throw std::runtime_error("LowerToTIR requires TensorType on function parameters: " +
                                         param->vid->name_hint);
            }
            tir::DataType dtype = DTypeFromString(ttype->dtype);
            Array<tir::PrimExpr> shape = ShapeFromTensorType(ttype);
            te::Tensor tensor = te::placeholder(shape, dtype, param->vid->name_hint);
            var_map_[param.get()] = tensor;
            input_tensors_.push_back(tensor);
        }
    }

    Array<te::Tensor> Convert(const Expr& expr) { return Visit(expr); }

    const Array<te::Tensor>& input_tensors() const { return input_tensors_; }
    const Array<te::Tensor>& constant_tensors() const { return constant_tensors_; }

protected:
    std::unordered_map<const Object*, Array<te::Tensor>> memo_;
    std::unordered_map<const Object*, te::Tensor> var_map_;
    Array<te::Tensor> input_tensors_;
    Array<te::Tensor> constant_tensors_;
    int constant_counter_ = 0;

    Array<te::Tensor> Visit(const Expr& expr) override {
        auto it = memo_.find(expr.get());
        if (it != memo_.end()) return it->second;
        auto res = RelayPassFunctor::Visit(expr);
        memo_[expr.get()] = res;
        return res;
    }

    Array<te::Tensor> VisitVar(const VarNode* op, const Expr& ref) override {
        auto it = var_map_.find(ref.get());
        if (it == var_map_.end()) {
            throw std::runtime_error("Unexpected free var in LowerToTIR: " + op->vid->name_hint);
        }
        return {it->second};
    }

    Array<te::Tensor> VisitConstant(const ConstantNode* op, const Expr& ref) override {
        Array<tir::PrimExpr> shape;
        for (const auto dim : op->data->shape) {
            shape.push_back(tir::IntImm(dim, tir::DataType::Int(64)));
        }
        tir::DataType dtype = DTypeFromDL(op->data->dl_tensor.dtype);
        std::string name = "const_" + std::to_string(constant_counter_++);
        te::Tensor t = te::placeholder(shape, dtype, name);
        constant_tensors_.push_back(t);
        return {t};
    }

    Array<te::Tensor> VisitCall(const CallNode* op, const Expr& ref) override {
        Array<te::Tensor> inputs;
        for (const auto& arg : op->args) {
            auto arg_tensors = Visit(arg);
            for (const auto& t : arg_tensors) inputs.push_back(t);
        }

        auto* op_node = op->op.As<OpNode>();
        if (!op_node) {
            throw std::runtime_error("Call.op is not OpNode in LowerToTIR");
        }
        if (IsDeviceCommunicationOpName(op_node->name)) {
            throw std::runtime_error(
                "LowerToTIR does not lower device communication ops. "
                "Run LowerRelayToExecPlanPass before LowerToTIR.");
        }

        Attrs attrs = op->attrs.defined() ? Attrs(op->attrs) : Attrs();
        kxc::Type out_type = ref.checked_type();
        return InvokeRelayToTE(op_node, attrs, inputs, out_type);
    }

    Array<te::Tensor> VisitTuple(const TupleNode* op, const Expr& ref) override {
        (void)ref;
        Array<te::Tensor> outputs;
        for (size_t i = 0; i < op->fields.size(); ++i) {
            Array<te::Tensor> field_outputs = Visit(op->fields[i]);
            if (field_outputs.size() != 1) {
                throw std::runtime_error("LowerToTIR does not support nested tuple field " +
                                         std::to_string(i) + "; field produced " +
                                         std::to_string(field_outputs.size()) + " tensors");
            }
            outputs.push_back(field_outputs[0]);
        }
        return outputs;
    }

    Array<te::Tensor> VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override {
        (void)ref;
        Array<te::Tensor> tuple_outputs = Visit(op->tuple);
        if (op->index < 0 || static_cast<size_t>(op->index) >= tuple_outputs.size()) {
            throw std::runtime_error("TupleGetItem index out of range during LowerToTIR: " +
                                     std::to_string(op->index) + ", tuple size " +
                                     std::to_string(tuple_outputs.size()));
        }
        return {tuple_outputs[static_cast<size_t>(op->index)]};
    }

    Array<te::Tensor> VisitDefault(const Expr& expr) override {
        throw std::runtime_error("Unsupported Relay node in LowerToTIR: " +
                                 RelayNodeKind(expr));
    }
};

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

void CollectOpsDFS(const te::Tensor& t,
                   std::unordered_set<const Object*>* visited,
                   std::unordered_map<const Object*, te::Tensor>* op_output,
                   std::vector<te::Operation>* topo) {
    if (!t.defined() || !t->op.defined()) return;
    const Object* op_ptr = t->op.get();
    if (visited->count(op_ptr)) return;

    if (auto* cop = t->op.As<te::ComputeOpNode>()) {
        for (const auto& body_expr : cop->body) {
            std::vector<te::Tensor> deps;
            FindProducerLoads(body_expr, &deps);
            for (const auto& dep : deps) {
                CollectOpsDFS(dep, visited, op_output, topo);
            }
        }
    }

    visited->insert(op_ptr);
    (*op_output)[op_ptr] = t;
    topo->push_back(t->op);
}

class ExprLowerer {
public:
    explicit ExprLowerer(const std::unordered_map<const Object*, tir::Var>& buffer_var_by_tensor)
        : buffer_var_by_tensor_(buffer_var_by_tensor) {}

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

tir::Stmt WrapDataLoops(const te::ComputeOpNode* op, tir::Stmt body) {
    for (int i = static_cast<int>(op->axis.size()) - 1; i >= 0; --i) {
        body = tir::For(op->axis[i], tir::IntImm(0), op->shape[i], tir::ForType::Serial, body);
    }
    return body;
}

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
    if (op->body.size() != 1) {
        throw std::runtime_error("LowerComputeStmt does not support multi-body TE compute for tensor '" +
                                 out_tensor->name + "': body_count=" +
                                 std::to_string(op->body.size()));
    }

    auto out_it = buffer_var_by_tensor.find(out_tensor.get());
    if (out_it == buffer_var_by_tensor.end()) {
        throw std::runtime_error("Missing output buffer var for compute tensor");
    }
    tir::Var out_var = out_it->second;

    Array<tir::PrimExpr> data_indices;
    for (const auto& ax : op->axis) data_indices.push_back(ax);
    tir::PrimExpr out_index = FlattenIndex(data_indices, op->shape);

    tir::PrimExpr body_expr = op->body[0];
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

tir::PrimFunc LowerToTIR(Function func) {
    if (!func.defined()) {
        throw std::runtime_error("LowerToTIR expects a defined function");
    }
    func = InferTypePass(func);
    if (!func->body.defined()) {
        throw std::runtime_error("LowerToTIR expects function body to be defined");
    }

    auto profile_context = profiling::CurrentContext();
    profiling::EventSpec spec;
    spec.component = "lowering";
    spec.event_type = "lower_to_tir";
    profiling::ScopedSpan span(profile_context, std::move(spec));
    const std::string relay_text = relay::pass::ToText(func);
    const std::string relay_hash = profiling::HashText(relay_text);
    span.AddField("relay_ir_hash", relay_hash);
    span.AddMetric("relay_ir_bytes", static_cast<double>(relay_text.size()));

    PassContext inferred_pass_ctx = PassContext::Current();
    if (!inferred_pass_ctx.defined()) {
        inferred_pass_ctx = PassContext::FromRelay(func);
    }
    try {
        PassContext::Scope pass_scope(inferred_pass_ctx);

        RelayToTEConverter converter(func);
        Array<te::Tensor> outputs = converter.Convert(func->body);
        if (outputs.empty()) {
            throw std::runtime_error("LowerToTIR produced no output tensors");
        }
        std::unordered_map<const Object*, size_t> output_index_by_tensor;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const te::Tensor& out_tensor = outputs[i];
            if (!out_tensor.defined()) {
                throw std::runtime_error("LowerToTIR output tensor " +
                                         std::to_string(i) + " is undefined");
            }
            auto duplicate = output_index_by_tensor.find(out_tensor.get());
            if (duplicate != output_index_by_tensor.end()) {
                throw std::runtime_error(
                    "LowerToTIR does not support duplicate output tensor '" +
                    out_tensor->name + "' at output " + std::to_string(i) +
                    "; first seen at output " + std::to_string(duplicate->second));
            }
            output_index_by_tensor[out_tensor.get()] = i;
            if (!out_tensor->op.As<te::ComputeOpNode>()) {
                throw std::runtime_error(
                    "LowerToTIR requires output tensor " + std::to_string(i) +
                    " ('" + out_tensor->name + "') to lower to a compute tensor");
            }
        }

        std::unordered_set<const Object*> visited_ops;
        std::unordered_map<const Object*, te::Tensor> op_output_tensor;
        std::vector<te::Operation> topo_ops;
        for (const auto& out_tensor : outputs) {
            CollectOpsDFS(out_tensor, &visited_ops, &op_output_tensor, &topo_ops);
        }

        Array<tir::Var> params;
        Map<tir::Var, tir::Buffer> buffer_map;
        std::unordered_map<const Object*, tir::Var> buffer_var_by_tensor;

        for (const auto& t : converter.input_tensors()) {
            tir::Var data_var(t->name, t->dtype);
            tir::Buffer buf(data_var, t->dtype, t->shape, {}, tir::IntImm(0), t->name, 0, 0);
            params.push_back(data_var);
            buffer_map.Set(data_var, buf);
            buffer_var_by_tensor[t.get()] = data_var;
        }
        const int64_t input_count = static_cast<int64_t>(converter.input_tensors().size());

        for (const auto& t : converter.constant_tensors()) {
            tir::Var data_var(t->name, t->dtype);
            tir::Buffer buf(data_var, t->dtype, t->shape, {}, tir::IntImm(0), t->name, 0, 0);
            params.push_back(data_var);
            buffer_map.Set(data_var, buf);
            buffer_var_by_tensor[t.get()] = data_var;
        }
        const int64_t constant_count = static_cast<int64_t>(converter.constant_tensors().size());
        const int64_t output_param_start = input_count + constant_count;

        std::unordered_set<const Object*> output_tensor_set;
        std::unordered_set<std::string> used_output_names;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const te::Tensor& out_tensor = outputs[i];
            output_tensor_set.insert(out_tensor.get());
            std::string out_name = MakeOutputVarName(out_tensor, i, &used_output_names);
            tir::Var out_var(out_name, out_tensor->dtype);
            tir::Buffer out_buf(out_var, out_tensor->dtype, out_tensor->shape, {},
                                tir::IntImm(0), out_name, 0, 0);
            params.push_back(out_var);
            buffer_map.Set(out_var, out_buf);
            buffer_var_by_tensor[out_tensor.get()] = out_var;
        }

        std::vector<te::Tensor> intermediates;
        for (const auto& op : topo_ops) {
            if (!op.As<te::ComputeOpNode>()) continue;
            auto t_it = op_output_tensor.find(op.get());
            if (t_it == op_output_tensor.end()) continue;
            te::Tensor t = t_it->second;
            if (output_tensor_set.count(t.get()) != 0) continue;
            tir::Var local_var(t->name, t->dtype);
            buffer_var_by_tensor[t.get()] = local_var;
            intermediates.push_back(t);
        }

        ExprLowerer expr_lowerer(buffer_var_by_tensor);

        Array<tir::Stmt> compute_seq;
        for (const auto& op : topo_ops) {
            if (op.As<te::PlaceholderOpNode>()) {
                continue;
            }
            auto t_it = op_output_tensor.find(op.get());
            if (t_it == op_output_tensor.end()) {
                throw std::runtime_error(
                    "Missing tensor for operation during statement lowering");
            }
            te::Tensor t = t_it->second;
            if (!t->op.As<te::ComputeOpNode>()) {
                throw std::runtime_error("Unsupported non-compute operation in TIR lowering");
            }
            try {
                compute_seq.push_back(LowerComputeStmt(t, buffer_var_by_tensor, expr_lowerer));
            } catch (const std::exception& e) {
                throw std::runtime_error("LowerComputeStmt failed for tensor '" + t->name +
                                         "': " + e.what());
            }
        }

        tir::Stmt body;
        if (compute_seq.empty()) {
            body = tir::Stmt();
        } else if (compute_seq.size() == 1) {
            body = compute_seq[0];
        } else {
            body = tir::SeqStmt(compute_seq);
        }

        for (int i = static_cast<int>(intermediates.size()) - 1; i >= 0; --i) {
            const auto& t = intermediates[i];
            tir::Var data_var = buffer_var_by_tensor[t.get()];
            body = tir::Allocate(data_var, t->dtype, t->shape,
                                 tir::IntImm(1, tir::DataType::Bool()), body);
        }

        Map<String, ObjectRef> attrs;
        attrs.Set(String("global_symbol"), String("main"));
        attrs.Set(String("tir.noalias"), tir::IntImm(1, tir::DataType::Bool()));
        attrs.Set(String("kxc.input_count"), tir::IntImm(input_count, tir::DataType::Int(64)));
        attrs.Set(String("kxc.constant_count"),
                  tir::IntImm(constant_count, tir::DataType::Int(64)));
        attrs.Set(String("kxc.output_count"),
                  tir::IntImm(static_cast<int64_t>(outputs.size()), tir::DataType::Int(64)));
        attrs.Set(String("kxc.output_param_start"),
                  tir::IntImm(output_param_start, tir::DataType::Int(64)));
        attrs = AttachPassContextAttrs(attrs, PassContext::Current());

        tir::PrimFunc lowered = tir::PrimFunc(params, body, buffer_map, attrs);
        std::ostringstream tir_os;
        tir::pass::DumpPrimFunc(lowered, tir_os);
        const std::string tir_text = tir_os.str();
        const std::string tir_hash = profiling::HashText(tir_text);
        const bool changed = relay_hash != tir_hash;
        span.AddField("tir_ir_hash", tir_hash);
        span.AddField("ir_changed", changed ? "true" : "false");
        span.AddMetric("tir_ir_bytes", static_cast<double>(tir_text.size()));

        if (profiling::ShouldCaptureIR(profile_context, changed, false)) {
            const std::string prefix = profiling::CurrentRunId() + "/lower/lower_to_tir";
            profile_context->WriteArtifact(prefix + ".before.relay.txt", relay_text);
            profile_context->WriteArtifact(prefix + ".after.tir.txt", tir_text);
        }
        return lowered;
    } catch (const std::exception& e) {
        span.SetStatus("error");
        span.SetMessage(e.what());
        if (profile_context) {
            const std::string prefix = profiling::CurrentRunId() + "/lower/lower_to_tir";
            if (profiling::ShouldCaptureIR(profile_context, true, true)) {
                profile_context->WriteArtifact(prefix + ".failed.relay.txt", relay_text);
            }
            profile_context->RecordLog(profiling::LogSeverity::kError, "lowering", e.what(),
                                       profiling::MakeFields({
                                           {"relay_ir_hash", relay_hash},
                                           {"stage", "lower_to_tir"},
                                       }));
        }
        throw;
    }
}

}  // namespace relay
}  // namespace kxc
