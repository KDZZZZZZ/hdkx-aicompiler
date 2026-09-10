#include "internal/module_invocation_contract.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace kxc::api {
namespace {
void Put(std::string& out, const std::string& value) { out += std::to_string(value.size()) + ":" + value + ";"; }
void Put(std::string& out, ModuleExtent value) { Put(out, std::to_string(value)); }
ModuleExtent CheckedAdd(ModuleExtent a, ModuleExtent b) { if (b > std::numeric_limits<ModuleExtent>::max() - a) throw std::overflow_error("module shape addition overflow"); return a + b; }
ModuleExtent CheckedMul(ModuleExtent a, ModuleExtent b) { if (a && b > std::numeric_limits<ModuleExtent>::max() / a) throw std::overflow_error("module shape multiplication overflow"); return a * b; }
}

struct ModuleShapeExpr::Node { Kind kind; ModuleExtent value{}; std::size_t input{}, axis{}; std::shared_ptr<const Node> lhs, rhs; };
ModuleShapeExpr::ModuleShapeExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
ModuleShapeExpr ModuleShapeExpr::Const(ModuleExtent value) { auto n=std::make_shared<Node>(); n->kind=Kind::kConst; n->value=value; return ModuleShapeExpr(n); }
ModuleShapeExpr ModuleShapeExpr::InputAxis(std::size_t input, std::size_t axis) { auto n=std::make_shared<Node>(); n->kind=Kind::kInputAxis; n->input=input; n->axis=axis; return ModuleShapeExpr(n); }
ModuleShapeExpr ModuleShapeExpr::Binary(Kind kind, ModuleShapeExpr lhs, ModuleShapeExpr rhs) { if (!lhs.defined() || !rhs.defined()) throw std::invalid_argument("module shape expression operand is undefined"); auto n=std::make_shared<Node>(); n->kind=kind; n->lhs=lhs.node_; n->rhs=rhs.node_; return ModuleShapeExpr(n); }
ModuleShapeExpr ModuleShapeExpr::Add(ModuleShapeExpr a, ModuleShapeExpr b) { return Binary(Kind::kAdd,a,b); }
ModuleShapeExpr ModuleShapeExpr::Mul(ModuleShapeExpr a, ModuleShapeExpr b) { return Binary(Kind::kMul,a,b); }
ModuleShapeExpr ModuleShapeExpr::FloorDiv(ModuleShapeExpr a, ModuleShapeExpr b) { return Binary(Kind::kFloorDiv,a,b); }
ModuleShapeExpr ModuleShapeExpr::Min(ModuleShapeExpr a, ModuleShapeExpr b) { return Binary(Kind::kMin,a,b); }
ModuleShapeExpr ModuleShapeExpr::Max(ModuleShapeExpr a, ModuleShapeExpr b) { return Binary(Kind::kMax,a,b); }

void ModuleShapeExpr::Validate(const std::vector<std::size_t>& input_ranks) const {
    if (!node_) throw std::invalid_argument("module shape expression is undefined");
    std::vector<std::pair<std::shared_ptr<const Node>, std::size_t>> stack{{node_, 1}};
    std::unordered_set<const Node*> seen;
    while (!stack.empty()) {
        const auto [node, depth] = stack.back(); stack.pop_back();
        if (depth > kMaxDepth) throw std::invalid_argument("module shape expression depth exceeds limit");
        if (!seen.insert(node.get()).second) continue;
        if (seen.size() > kMaxUniqueNodes) throw std::invalid_argument("module shape expression unique-node limit exceeded");
        if (node->kind == Kind::kInputAxis) {
            if (node->input >= input_ranks.size() || node->axis >= input_ranks[node->input])
                throw std::invalid_argument("module shape expression references missing input axis");
        } else if (node->kind != Kind::kConst) {
            if (!node->lhs || !node->rhs) throw std::invalid_argument("module shape expression binary operand is undefined");
            stack.emplace_back(node->lhs, depth + 1); stack.emplace_back(node->rhs, depth + 1);
        }
    }
}
std::pair<ModuleExtent, ModuleExtent> ModuleShapeExpr::ValidateRanges(
    const std::vector<std::vector<ModuleExtent>>& input_lowers,
    const std::vector<std::vector<ModuleExtent>>& input_uppers) const {
    if (!node_ || input_lowers.size() != input_uppers.size()) throw std::invalid_argument("module shape range bounds are invalid");
    std::vector<std::size_t> ranks; ranks.reserve(input_lowers.size());
    for (std::size_t input=0; input<input_lowers.size(); ++input) {
        if (input_lowers[input].size() != input_uppers[input].size()) throw std::invalid_argument("module shape range rank differs");
        for (std::size_t axis=0; axis<input_lowers[input].size(); ++axis) if (input_lowers[input][axis] > input_uppers[input][axis]) throw std::invalid_argument("module shape range is invalid");
        ranks.push_back(input_lowers[input].size());
    }
    Validate(ranks);
    struct Range { ModuleExtent lower; ModuleExtent upper; };
    std::unordered_map<const Node*, Range> ranges;
    std::vector<std::pair<std::shared_ptr<const Node>, bool>> stack{{node_, false}};
    while (!stack.empty()) {
        const auto [node, visited]=stack.back(); stack.pop_back();
        if (ranges.count(node.get())) continue;
        if (!visited) {
            stack.emplace_back(node, true);
            if (node->kind != Kind::kConst && node->kind != Kind::kInputAxis) {
                stack.emplace_back(node->rhs, false); stack.emplace_back(node->lhs, false);
            }
            continue;
        }
        Range range{};
        if (node->kind == Kind::kConst) range={node->value,node->value};
        else if (node->kind == Kind::kInputAxis) range={input_lowers[node->input][node->axis],input_uppers[node->input][node->axis]};
        else {
            const Range lhs=ranges.at(node->lhs.get()), rhs=ranges.at(node->rhs.get());
            switch (node->kind) {
                case Kind::kAdd:
                    if (lhs.upper > std::numeric_limits<ModuleExtent>::max()-rhs.upper) throw std::invalid_argument("module shape addition may overflow");
                    range={lhs.lower+rhs.lower,lhs.upper+rhs.upper}; break;
                case Kind::kMul:
                    if (lhs.upper && rhs.upper > std::numeric_limits<ModuleExtent>::max()/lhs.upper) throw std::invalid_argument("module shape multiplication may overflow");
                    range={lhs.lower*rhs.lower,lhs.upper*rhs.upper}; break;
                case Kind::kFloorDiv:
                    if (!rhs.lower) throw std::invalid_argument("module shape floor divisor is not provably positive");
                    range={lhs.lower/rhs.upper,lhs.upper/rhs.lower}; break;
                case Kind::kMin: range={std::min(lhs.lower,rhs.lower),std::min(lhs.upper,rhs.upper)}; break;
                case Kind::kMax: range={std::max(lhs.lower,rhs.lower),std::max(lhs.upper,rhs.upper)}; break;
                case Kind::kConst:
                case Kind::kInputAxis: throw std::logic_error("unexpected module shape node");
            }
        }
        ranges.emplace(node.get(), range);
    }
    const Range root=ranges.at(node_.get());
    return {root.lower,root.upper};
}
std::size_t ModuleShapeExpr::unique_node_count() const {
    if (!node_) throw std::invalid_argument("module shape expression is undefined");
    std::unordered_set<const Node*> seen; std::vector<std::shared_ptr<const Node>> stack{node_};
    while (!stack.empty()) { auto node=stack.back(); stack.pop_back(); if (!seen.insert(node.get()).second) continue; if (node->lhs) stack.push_back(node->lhs); if (node->rhs) stack.push_back(node->rhs); }
    return seen.size();
}
ModuleExtent ModuleShapeExpr::Evaluate(const std::vector<std::vector<ModuleExtent>>& inputs) const {
    std::vector<std::size_t> ranks; ranks.reserve(inputs.size()); for (const auto& input : inputs) ranks.push_back(input.size()); Validate(ranks);
    std::unordered_map<const Node*, ModuleExtent> values;
    std::vector<std::pair<std::shared_ptr<const Node>, bool>> stack{{node_, false}};
    while (!stack.empty()) {
        const auto [node, visited] = stack.back(); stack.pop_back();
        if (values.count(node.get())) continue;
        if (!visited) { stack.emplace_back(node, true); if (node->kind != Kind::kConst && node->kind != Kind::kInputAxis) { stack.emplace_back(node->rhs, false); stack.emplace_back(node->lhs, false); } continue; }
        switch (node->kind) {
            case Kind::kConst: values.emplace(node.get(), node->value); break;
            case Kind::kInputAxis: values.emplace(node.get(), inputs[node->input][node->axis]); break;
            case Kind::kAdd: values.emplace(node.get(), CheckedAdd(values.at(node->lhs.get()), values.at(node->rhs.get()))); break;
            case Kind::kMul: values.emplace(node.get(), CheckedMul(values.at(node->lhs.get()), values.at(node->rhs.get()))); break;
            case Kind::kFloorDiv: { const auto divisor=values.at(node->rhs.get()); if (!divisor) throw std::invalid_argument("module shape floor division by zero"); values.emplace(node.get(), values.at(node->lhs.get()) / divisor); break; }
            case Kind::kMin: values.emplace(node.get(), std::min(values.at(node->lhs.get()), values.at(node->rhs.get()))); break;
            case Kind::kMax: values.emplace(node.get(), std::max(values.at(node->lhs.get()), values.at(node->rhs.get()))); break;
        }
    }
    return values.at(node_.get());
}
bool ModuleShapeExpr::IsConstant() const noexcept {
    if (!node_) return false;
    std::vector<std::shared_ptr<const Node>> stack{node_}; std::unordered_set<const Node*> seen;
    while (!stack.empty()) { auto node=stack.back(); stack.pop_back(); if (!seen.insert(node.get()).second) continue; if (node->kind == Kind::kInputAxis) return false; if (node->lhs) stack.push_back(node->lhs); if (node->rhs) stack.push_back(node->rhs); }
    return true;
}
bool ModuleShapeExpr::defined() const noexcept { return static_cast<bool>(node_); }
void ModuleShapeExpr::AppendCanonical(std::string& out) const {
    if (!node_) throw std::invalid_argument("module shape expression is undefined");
    // Ordered preorder AST is explicitly ABI-significant; sharing does not alter bytes.
    std::vector<std::shared_ptr<const Node>> stack{node_}; std::size_t nodes=0;
    while (!stack.empty()) { auto node=stack.back(); stack.pop_back(); if (++nodes > kMaxUniqueNodes * 2) throw std::invalid_argument("module shape expression serialization limit exceeded"); Put(out,static_cast<ModuleExtent>(node->kind)); if(node->kind==Kind::kConst) Put(out,node->value); else if(node->kind==Kind::kInputAxis){Put(out,node->input);Put(out,node->axis);} else { stack.push_back(node->rhs); stack.push_back(node->lhs); } }
}

ModuleInvocationContract::ModuleInvocationContract(std::vector<ModuleInputContract> inputs, std::vector<ModuleTensorContract> outputs, std::vector<ModuleRuntimeExtentScalar> scalars, std::size_t budget, std::uint32_t version)
    : abi_version_(version), inputs_(std::move(inputs)), outputs_(std::move(outputs)), runtime_extent_scalars_(std::move(scalars)), run_byte_budget_(budget) {
    if (version != kAbiVersion || outputs_.empty()) throw std::invalid_argument("invalid module invocation contract header");
    for (const auto& output : outputs_) {
        if (output.logical.size() != output.physical.size() || output.logical.size() != output.valid.size()) throw std::invalid_argument("module invocation output extents differ in rank");
        for (const auto* expressions : {&output.logical, &output.physical, &output.valid}) for (const auto& expression : *expressions) if (!expression.defined()) throw std::invalid_argument("module invocation output expression is undefined");
    }
    for (const auto& scalar : runtime_extent_scalars_) {
        if (scalar.source == ModuleRuntimeExtentScalar::Source::kStateExtent) {
            if (scalar.expression.has_value()) {
                throw std::invalid_argument(
                    "state-sourced runtime extent scalar must not carry an input-axis expression");
            }
            continue;
        }
        if (!scalar.expression.has_value()) throw std::invalid_argument("module runtime extent scalar expression is undefined");
    }
}
const std::vector<ModuleInputContract>& ModuleInvocationContract::inputs() const noexcept{return inputs_;}
const std::vector<ModuleTensorContract>& ModuleInvocationContract::outputs() const noexcept{return outputs_;}
const std::vector<ModuleRuntimeExtentScalar>& ModuleInvocationContract::runtime_extent_scalars() const noexcept{return runtime_extent_scalars_;}
std::size_t ModuleInvocationContract::run_byte_budget() const noexcept{return run_byte_budget_;}
std::uint32_t ModuleInvocationContract::abi_version() const noexcept{return abi_version_;}
bool ModuleInvocationContract::IsConstantShape(const codegen::KernelSignature& signature) const noexcept {
    try {
        Validate(signature);
        if (!runtime_extent_scalars_.empty()) return false;
        const auto args = signature.arguments(); size_t input = 0;
        for (const auto& arg : args) if (arg->role == codegen::KernelArgRole::kInput && input < inputs_.size()) {
            const auto shape = arg.shape(); const auto& guards = inputs_[input++].axis_guards;
            if (guards.size() != shape.size()) return false;
            for (size_t axis = 0; axis < shape.size(); ++axis) if (shape[axis] == codegen::kDynamicDimension || guards[axis].axis != axis || !guards[axis].exact || guards[axis].equal_to) return false;
        }
        for (const auto& output : outputs_) for (const auto* expressions : {&output.logical, &output.physical, &output.valid}) for (const auto& expression : *expressions) if (!expression.IsConstant()) return false;
        return true;
    } catch (...) { return false; }
}
std::string ModuleInvocationContract::CanonicalBytes() const {
    std::string b="KXC_MODULE_INVOKE_V4;"; Put(b,abi_version_);Put(b,inputs_.size());Put(b,outputs_.size());Put(b,runtime_extent_scalars_.size());Put(b,run_byte_budget_);
    for(const auto& i:inputs_){Put(b,i.axis_guards.size());for(const auto& g:i.axis_guards){Put(b,g.axis);Put(b,g.lower);Put(b,g.upper);Put(b,g.divisible_by);Put(b,g.exact?1:0);if(g.exact)Put(b,*g.exact);Put(b,g.equal_to?1:0);if(g.equal_to){Put(b,g.equal_to->input_index);Put(b,g.equal_to->axis);}}}
    for(const auto&o:outputs_){Put(b,o.max_bytes);for(const auto* v:{&o.logical,&o.physical,&o.valid}){Put(b,v->size());for(const auto&e:*v)e.AppendCanonical(b);}}
    for(const auto& scalar:runtime_extent_scalars_){Put(b,static_cast<ModuleExtent>(scalar.source));if(scalar.source==ModuleRuntimeExtentScalar::Source::kInputAxis)scalar.expression->AppendCanonical(b);} return b;
}
void ModuleInvocationContract::Validate(const codegen::KernelSignature& signature) const {
    signature.Validate(); if(abi_version_!=kAbiVersion) throw std::invalid_argument("module invocation ABI version is unsupported");
    const auto args=signature.arguments(); std::vector<std::size_t> input_ranks; input_ranks.reserve(inputs_.size()); std::vector<std::vector<ModuleExtent>> input_lowers, input_uppers; input_lowers.reserve(inputs_.size()); input_uppers.reserve(inputs_.size()); size_t input=0, scalar=0, output=0;
    for(const auto& arg:args) {
        if(arg->role==codegen::KernelArgRole::kInput) {
            const auto shape=arg.shape();
            if(input<inputs_.size()) {
                const auto& contract_input=inputs_[input]; input_ranks.push_back(shape.size()); input_lowers.emplace_back(shape.size(), 0); input_uppers.emplace_back(shape.size(), std::numeric_limits<ModuleExtent>::max());
                for(size_t axis=0;axis<shape.size();++axis) if(shape[axis]!=codegen::kDynamicDimension) input_lowers.back()[axis]=input_uppers.back()[axis]=static_cast<ModuleExtent>(shape[axis]);
                std::size_t prior=0; bool first=true;
                for(const auto& guard:contract_input.axis_guards) {
                    if((!first && guard.axis<=prior) || guard.axis>=shape.size() || guard.lower>guard.upper || !guard.divisible_by || (guard.exact && (*guard.exact<guard.lower || *guard.exact>guard.upper || *guard.exact%guard.divisible_by))) throw std::invalid_argument("module invocation guards must be sorted, unique, and internally consistent");
                    if (guard.equal_to && (guard.equal_to->input_index > input ||
                        (guard.equal_to->input_index == input && guard.equal_to->axis >= guard.axis) ||
                        guard.equal_to->axis >= input_ranks[guard.equal_to->input_index])) {
                        throw std::invalid_argument("module invocation equality guard must target a prior physical input axis");
                    }
                    input_lowers.back()[guard.axis]=guard.exact ? *guard.exact : guard.lower;
                    input_uppers.back()[guard.axis]=guard.exact ? *guard.exact : guard.upper;
                    prior=guard.axis; first=false;
                }
                if(contract_input.axis_guards.size()!=shape.size()) throw std::invalid_argument("module invocation requires exactly one guard for every caller input axis");
                for(size_t axis=0;axis<shape.size();++axis) {
                    const auto& guard=contract_input.axis_guards[axis];
                    if(guard.axis!=axis) throw std::invalid_argument("module invocation guards must cover caller input axes in order");
                    if(shape[axis]==codegen::kDynamicDimension) {
                        if(guard.upper>static_cast<ModuleExtent>(std::numeric_limits<int64_t>::max()) || (guard.exact && *guard.exact>static_cast<ModuleExtent>(std::numeric_limits<int64_t>::max()))) throw std::invalid_argument("dynamic module input guard exceeds the int64 shape domain");
                    } else if(!guard.exact || *guard.exact!=static_cast<ModuleExtent>(shape[axis])) {
                        throw std::invalid_argument("static physical input ABI requires an exact guard on every axis");
                    }
                }
                ++input;
            } else {
                throw std::invalid_argument("module invocation caller input arity differs from physical ABI");
            }
        } else if(arg->role==codegen::KernelArgRole::kRuntimeExtent) {
            const auto shape=arg.shape();
            if(scalar>=runtime_extent_scalars_.size() || arg->dtype.code!=kDLUInt || arg->dtype.bits!=64 || arg->dtype.lanes!=1 || shape.size()!=1 || shape[0]!=1) throw std::invalid_argument("runtime extent ABI must be a uint64[1] buffer");
            ++scalar;
        } else if(arg->role==codegen::KernelArgRole::kConstant) {
            continue;
        } else if(arg->role==codegen::KernelArgRole::kOutput) {
            if(output>=outputs_.size()) throw std::invalid_argument("module invocation output arity differs from physical ABI");
            const auto& contract_output=outputs_[output++]; const auto shape=arg.shape();
            if(contract_output.logical.size()!=shape.size()) throw std::invalid_argument("module invocation output rank differs from physical ABI");
        }
    }
    if(input!=inputs_.size()||scalar!=runtime_extent_scalars_.size()||output!=outputs_.size()) throw std::invalid_argument("module invocation ABI arity differs from physical ABI");
    std::size_t count=0;
    const auto validate_expression=[&](const ModuleShapeExpr& expression) {
        expression.Validate(input_ranks); const auto range=expression.ValidateRanges(input_lowers,input_uppers);
        if((count+=expression.unique_node_count())>kMaxExpressions) throw std::invalid_argument("module invocation expression aggregate limit exceeded");
        return range;
    };
    const auto same_expression=[](const ModuleShapeExpr& lhs, const ModuleShapeExpr& rhs) {
        std::string left, right; lhs.AppendCanonical(left); rhs.AppendCanonical(right); return left==right;
    };
    for(const auto& contract_output:outputs_) for(size_t axis=0;axis<contract_output.logical.size();++axis) {
        const auto logical=validate_expression(contract_output.logical[axis]);
        const auto physical=validate_expression(contract_output.physical[axis]);
        const auto valid=validate_expression(contract_output.valid[axis]);
        if(!same_expression(contract_output.valid[axis],contract_output.logical[axis]) && valid.second>logical.first) throw std::invalid_argument("module output valid extent is not provably <= logical extent");
        if(!same_expression(contract_output.logical[axis],contract_output.physical[axis]) && logical.second>physical.first) throw std::invalid_argument("module output logical extent is not provably <= physical extent");
    }
    for(const auto& scalar_contract:runtime_extent_scalars_) if(scalar_contract.source!=ModuleRuntimeExtentScalar::Source::kStateExtent) (void)validate_expression(*scalar_contract.expression);
    size_t physical_output=0; for(const auto& arg:args) if(arg->role==codegen::KernelArgRole::kOutput) { const auto shape=arg.shape(); const auto& contract_output=outputs_[physical_output++]; for(size_t axis=0;axis<shape.size();++axis) if(shape[axis]!=codegen::kDynamicDimension && (!contract_output.physical[axis].IsConstant() || contract_output.physical[axis].Evaluate({})!=static_cast<ModuleExtent>(shape[axis]))) throw std::invalid_argument("module physical output ABI does not match KernelSignature"); }
}
}  // namespace kxc::api
