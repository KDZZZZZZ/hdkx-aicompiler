#include "kxc/runtime/module_invocation.h"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace kxc::api {
namespace {
void Put(std::string& out, const std::string& value) { out += std::to_string(value.size()) + ":" + value + ";"; }
void Put(std::string& out, ModuleExtent value) { out += std::to_string(value) + ";"; }
ModuleExtent CheckedAdd(ModuleExtent a, ModuleExtent b) { if (b > std::numeric_limits<ModuleExtent>::max() - a) throw std::overflow_error("module shape addition overflow"); return a + b; }
ModuleExtent CheckedMul(ModuleExtent a, ModuleExtent b) { if (a && b > std::numeric_limits<ModuleExtent>::max() / a) throw std::overflow_error("module shape multiplication overflow"); return a * b; }
bool Pow2(std::size_t value) { return value && !(value & (value - 1)); }
std::string DType(DLDataType d) { return std::to_string(d.code) + ":" + std::to_string(d.bits) + ":" + std::to_string(d.lanes); }
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
ModuleExtent ModuleShapeExpr::Evaluate(const std::vector<std::vector<ModuleExtent>>& inputs) const {
    if (!node_) throw std::invalid_argument("module shape expression is undefined");
    const auto eval = [&](const auto& self, const std::shared_ptr<const Node>& n) -> ModuleExtent {
        switch(n->kind) { case Kind::kConst:return n->value; case Kind::kInputAxis: if(n->input>=inputs.size()||n->axis>=inputs[n->input].size()) throw std::invalid_argument("module shape expression references missing input axis"); return inputs[n->input][n->axis]; default: break; }
        const ModuleExtent a=self(self,n->lhs), b=self(self,n->rhs);
        switch(n->kind) { case Kind::kAdd:return CheckedAdd(a,b); case Kind::kMul:return CheckedMul(a,b); case Kind::kFloorDiv: if(!b) throw std::invalid_argument("module shape floor division by zero"); return a/b; case Kind::kMin:return a<b?a:b; case Kind::kMax:return a>b?a:b; default: throw std::logic_error("invalid module shape expression"); }
    }; return eval(eval,node_);
}
bool ModuleShapeExpr::IsConstant() const noexcept { return node_ && node_->kind == Kind::kConst; }
bool ModuleShapeExpr::defined() const noexcept { return static_cast<bool>(node_); }
void ModuleShapeExpr::AppendCanonical(std::string& out) const { if(!node_) throw std::invalid_argument("module shape expression is undefined"); Put(out,static_cast<ModuleExtent>(node_->kind)); if(node_->kind==Kind::kConst) Put(out,node_->value); else if(node_->kind==Kind::kInputAxis){Put(out,node_->input);Put(out,node_->axis);} else { ModuleShapeExpr(node_->lhs).AppendCanonical(out); ModuleShapeExpr(node_->rhs).AppendCanonical(out); } }

ModuleInvocationContract::ModuleInvocationContract(std::vector<ModuleInputContract> inputs, std::vector<ModuleTensorContract> outputs, std::size_t budget, std::uint32_t version) : abi_version_(version), inputs_(std::move(inputs)), outputs_(std::move(outputs)), run_byte_budget_(budget) { if(version!=kAbiVersion || outputs_.empty()) throw std::invalid_argument("invalid module invocation contract header"); }
const std::vector<ModuleInputContract>& ModuleInvocationContract::inputs() const noexcept{return inputs_;}
const std::vector<ModuleTensorContract>& ModuleInvocationContract::outputs() const noexcept{return outputs_;}
std::size_t ModuleInvocationContract::run_byte_budget() const noexcept{return run_byte_budget_;}
std::uint32_t ModuleInvocationContract::abi_version() const noexcept{return abi_version_;}
bool ModuleInvocationContract::IsConstantShape() const noexcept { for(const auto& o:outputs_) for(const auto& e:o.logical) if(!e.IsConstant()) return false; return true; }
std::string ModuleInvocationContract::CanonicalBytes() const { std::string b="KXC_MODULE_INVOKE_V1;"; Put(b,abi_version_);Put(b,inputs_.size());Put(b,outputs_.size());Put(b,run_byte_budget_); for(const auto& i:inputs_){Put(b,DType(i.dtype));Put(b,i.device.ToString());Put(b,i.rank);Put(b,i.axis_guards.size());for(const auto& g:i.axis_guards){Put(b,g.axis);Put(b,g.lower);Put(b,g.upper);Put(b,g.divisible_by);Put(b,g.exact?1:0);if(g.exact)Put(b,*g.exact);Put(b,g.equal_to?1:0);if(g.equal_to){Put(b,g.equal_to->input_index);Put(b,g.equal_to->axis);}}} for(const auto&o:outputs_){Put(b,DType(o.dtype));Put(b,o.device.ToString());Put(b,o.alignment);Put(b,o.layout);Put(b,o.scope);Put(b,o.max_bytes);for(const auto* v:{&o.logical,&o.physical,&o.valid}){Put(b,v->size());for(const auto&e:*v)e.AppendCanonical(b);}} return b; }
void ModuleInvocationContract::Validate(const codegen::KernelSignature& signature) const { signature.Validate(); if(abi_version_!=kAbiVersion) throw std::invalid_argument("module invocation ABI version is unsupported"); const auto args=signature.arguments(); size_t in=0,out=0; for(const auto&a:args){ if(a->role==codegen::KernelArgRole::kInput){if(in>=inputs_.size())throw std::invalid_argument("module invocation input arity differs from physical ABI"); const auto&i=inputs_[in++]; if(!Same(i.dtype,a->dtype)||i.device!=a->device||i.rank!=a.shape().size())throw std::invalid_argument("module invocation input differs from physical ABI"); for(const auto&g:i.axis_guards)if(g.axis>=i.rank||g.lower>g.upper||!g.divisible_by)throw std::invalid_argument("module invocation input guard is invalid");} else if(a->role==codegen::KernelArgRole::kOutput){if(out>=outputs_.size())throw std::invalid_argument("module invocation output arity differs from physical ABI"); const auto&o=outputs_[out++]; if(!Same(o.dtype,a->dtype)||o.device!=a->device||!Pow2(o.alignment)||o.layout!="contiguous.row_major"||o.scope!="global"||o.logical.size()!=o.physical.size()||o.logical.size()!=o.valid.size()||o.logical.size()!=a.shape().size())throw std::invalid_argument("module invocation output contract is invalid"); for(size_t x=0;x<o.physical.size();++x){ if(!o.logical[x].defined()||!o.physical[x].defined()||!o.valid[x].defined())throw std::invalid_argument("module invocation output shape expression is undefined"); const auto s=a.shape()[x]; if(s!=codegen::kDynamicDimension && (!o.physical[x].IsConstant() || o.physical[x].Evaluate({})!=static_cast<ModuleExtent>(s)))throw std::invalid_argument("module physical output ABI does not match KernelSignature"); }} } if(in!=inputs_.size()||out!=outputs_.size())throw std::invalid_argument("module invocation ABI arity differs from physical ABI"); }
}  // namespace kxc::api
