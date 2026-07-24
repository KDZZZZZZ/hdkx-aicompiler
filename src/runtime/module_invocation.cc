#include "kxc/runtime/module_invocation.h"

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
bool Pow2(std::size_t value) { return value && !(value & (value - 1)); }
bool ValidDType(DLDataType d) {
    return d.bits != 0 && d.bits % 8 == 0 && d.lanes != 0 &&
        (d.code == kDLInt || d.code == kDLUInt || d.code == kDLFloat ||
         d.code == kDLBfloat || d.code == kDLComplex || d.code == kDLBool);
}
std::string DType(DLDataType d) { return std::to_string(d.code) + ":" + std::to_string(d.bits) + ":" + std::to_string(d.lanes); }
void PutDevice(std::string& out, const Device& device) { Put(out, static_cast<ModuleExtent>(device.device_type())); Put(out, static_cast<ModuleExtent>(device.device_id())); }
bool Same(DLDataType a, DLDataType b) { return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes; }
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

ModuleInvocationContract::ModuleInvocationContract(std::vector<ModuleInputContract> inputs, std::vector<ModuleTensorContract> outputs, std::size_t budget, std::uint32_t version)
    : ModuleInvocationContract(std::move(inputs), std::move(outputs), {}, budget, version) {}
ModuleInvocationContract::ModuleInvocationContract(std::vector<ModuleInputContract> inputs, std::vector<ModuleTensorContract> outputs, std::vector<ModuleRuntimeExtentScalar> scalars, std::size_t budget, std::uint32_t version)
    : abi_version_(version), inputs_(std::move(inputs)), outputs_(std::move(outputs)), runtime_extent_scalars_(std::move(scalars)), run_byte_budget_(budget) {
    if(version!=kAbiVersion || outputs_.empty()) throw std::invalid_argument("invalid module invocation contract header");
    std::vector<std::size_t> ranks; ranks.reserve(inputs_.size());
    for (std::size_t i=0;i<inputs_.size();++i) {
        const auto& input=inputs_[i]; if(!ValidDType(input.dtype)||!input.device.defined()) throw std::invalid_argument("module invocation input dtype or device is invalid"); ranks.push_back(input.rank);
        std::size_t prior=0; bool first=true;
        for(const auto& guard:input.axis_guards) { if((!first && guard.axis<=prior) || guard.axis>=input.rank || guard.lower>guard.upper || !guard.divisible_by || (guard.exact && (*guard.exact<guard.lower || *guard.exact>guard.upper || *guard.exact%guard.divisible_by))) throw std::invalid_argument("module invocation guards must be sorted, unique, and internally consistent"); if(guard.equal_to && (guard.equal_to->input_index>=i || guard.equal_to->axis>=inputs_[guard.equal_to->input_index].rank)) throw std::invalid_argument("module invocation equality guard must target a strictly prior valid input"); prior=guard.axis; first=false; }
    }
    std::size_t count=0;
    for(const auto& output:outputs_) { if(!ValidDType(output.dtype)||!output.device.defined()||!Pow2(output.alignment)||output.layout!="contiguous.row_major"||output.scope!="global"||output.logical.size()!=output.physical.size()||output.logical.size()!=output.valid.size()) throw std::invalid_argument("module invocation output contract is invalid"); for(const auto* expressions:{&output.logical,&output.physical,&output.valid}) for(const auto& expression:*expressions) { expression.Validate(ranks); if((count += expression.unique_node_count()) > kMaxExpressions) throw std::invalid_argument("module invocation expression aggregate limit exceeded"); } }
    for(const auto& scalar:runtime_extent_scalars_) { if(!scalar.expression.defined()||scalar.dtype.code!=kDLUInt||scalar.dtype.bits!=64||scalar.dtype.lanes!=1||!scalar.device.defined()||!Pow2(scalar.alignment)) throw std::invalid_argument("module runtime extent scalar must be a valid uint64[1] ABI value"); scalar.expression.Validate(ranks); if((count += scalar.expression.unique_node_count()) > kMaxExpressions) throw std::invalid_argument("module invocation expression aggregate limit exceeded"); }
}
const std::vector<ModuleInputContract>& ModuleInvocationContract::inputs() const noexcept{return inputs_;}
const std::vector<ModuleTensorContract>& ModuleInvocationContract::outputs() const noexcept{return outputs_;}
const std::vector<ModuleRuntimeExtentScalar>& ModuleInvocationContract::runtime_extent_scalars() const noexcept{return runtime_extent_scalars_;}
std::size_t ModuleInvocationContract::run_byte_budget() const noexcept{return run_byte_budget_;}
std::uint32_t ModuleInvocationContract::abi_version() const noexcept{return abi_version_;}
bool ModuleInvocationContract::IsConstantShape() const noexcept {
    if (!runtime_extent_scalars_.empty()) return false;
    for(const auto& input:inputs_) { if(input.axis_guards.size()!=input.rank) return false; for(std::size_t axis=0;axis<input.rank;++axis) if(input.axis_guards[axis].axis!=axis || !input.axis_guards[axis].exact || input.axis_guards[axis].equal_to) return false; }
    for(const auto& output:outputs_) for(const auto* expressions:{&output.logical,&output.physical,&output.valid}) for(const auto& expression:*expressions) if(!expression.IsConstant()) return false;
    return true;
}
std::string ModuleInvocationContract::CanonicalBytes() const {
    std::string b="KXC_MODULE_INVOKE_V2;"; Put(b,abi_version_);Put(b,inputs_.size());Put(b,outputs_.size());Put(b,runtime_extent_scalars_.size());Put(b,run_byte_budget_);
    for(const auto& i:inputs_){Put(b,DType(i.dtype));PutDevice(b,i.device);Put(b,i.rank);Put(b,i.axis_guards.size());for(const auto& g:i.axis_guards){Put(b,g.axis);Put(b,g.lower);Put(b,g.upper);Put(b,g.divisible_by);Put(b,g.exact?1:0);if(g.exact)Put(b,*g.exact);Put(b,g.equal_to?1:0);if(g.equal_to){Put(b,g.equal_to->input_index);Put(b,g.equal_to->axis);}}}
    for(const auto&o:outputs_){Put(b,DType(o.dtype));PutDevice(b,o.device);Put(b,o.alignment);Put(b,o.layout);Put(b,o.scope);Put(b,o.max_bytes);for(const auto* v:{&o.logical,&o.physical,&o.valid}){Put(b,v->size());for(const auto&e:*v)e.AppendCanonical(b);}}
    for(const auto& scalar:runtime_extent_scalars_){Put(b,DType(scalar.dtype));PutDevice(b,scalar.device);Put(b,scalar.alignment);scalar.expression.AppendCanonical(b);} return b;
}
void ModuleInvocationContract::Validate(const codegen::KernelSignature& signature) const {
    signature.Validate(); if(abi_version_!=kAbiVersion) throw std::invalid_argument("module invocation ABI version is unsupported"); const auto args=signature.arguments(); size_t in=0,out=0,scalar=0;
    for(const auto&a:args){
        if(a->role==codegen::KernelArgRole::kInput){
            if(in<inputs_.size()){ const auto&i=inputs_[in++]; if(!Same(i.dtype,a->dtype)||i.device!=a->device||i.rank!=a.shape().size())throw std::invalid_argument("module invocation input differs from physical ABI"); const auto shape=a.shape(); for(std::size_t axis=0;axis<shape.size();++axis) if(shape[axis]!=codegen::kDynamicDimension) { const auto guard=std::find_if(i.axis_guards.begin(),i.axis_guards.end(),[&](const ModuleAxisGuard& candidate){return candidate.axis==axis;}); if(guard==i.axis_guards.end() || !guard->exact || *guard->exact!=static_cast<ModuleExtent>(shape[axis])) throw std::invalid_argument("static physical input ABI requires an exact guard on every axis"); }
            } else { if(scalar>=runtime_extent_scalars_.size()) throw std::invalid_argument("module invocation input arity differs from physical ABI"); const auto&s=runtime_extent_scalars_[scalar++]; const auto shape=a.shape(); if(!Same(s.dtype,a->dtype)||s.device!=a->device||s.alignment!=a->alignment||shape.size()!=1||shape[0]!=1) throw std::invalid_argument("runtime extent scalar ABI must be a one-element matching input buffer"); }
        } else if(a->role==codegen::KernelArgRole::kOutput){ if(out>=outputs_.size())throw std::invalid_argument("module invocation output arity differs from physical ABI"); const auto&o=outputs_[out++]; if(!Same(o.dtype,a->dtype)||o.device!=a->device||o.alignment!=a->alignment||o.logical.size()!=a.shape().size())throw std::invalid_argument("module invocation output differs from physical ABI"); const auto shape=a.shape(); for(size_t x=0;x<shape.size();++x) if(shape[x]!=codegen::kDynamicDimension && (!o.physical[x].IsConstant() || o.physical[x].Evaluate({})!=static_cast<ModuleExtent>(shape[x])))throw std::invalid_argument("module physical output ABI does not match KernelSignature"); }
    }
    if(in!=inputs_.size()||out!=outputs_.size()||scalar!=runtime_extent_scalars_.size())throw std::invalid_argument("module invocation ABI arity differs from physical ABI");
}
}  // namespace kxc::api
