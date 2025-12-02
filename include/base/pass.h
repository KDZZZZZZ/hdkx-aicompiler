#pragma once
#include "relay.h"
namespace kxc {
template <typename R>
class RelayPassFunctor {
public:
    virtual ~RelayPassFunctor() = default;
    virtual R Visit(const Relay& expr) {
        if (!expr.defined()) return R()
        if (auto* n = expr.As<ConstantNode>()) return VisitConstant(n, expr);
        if (auto* n = expr.As<VarNode>())      return VisitVar(n, expr);
        if (auto* n = expr.As<CallNode>())     return VisitCall(n, expr);
        if (auto* n = expr.As<FunctionNode>()) return VisitFunction(n, expr);
        if (auto* n = expr.As<IdNode>())       return VisitId(n, expr);
        return VisitDefault(expr);
    }
private:
    virtual R VisitConstant(const ConstantNode* op, const Relay& ref) { return VisitDefault(ref); }
    virtual R VisitVar(const VarNode* op, const Relay& ref)           { return VisitDefault(ref); }
    virtual R VisitCall(const CallNode* op, const Relay& ref)         { return VisitDefault(ref); }
    virtual R VisitFunction(const FunctionNode* op, const Relay& ref) { return VisitDefault(ref); }
    virtual R VisitId(const IdNode* op, const Relay& ref)             { return VisitDefault(ref); }

    virtual R VisitDefault(const Relay& expr) {
        // 默认报错或返回空
        return R(); 
    }
};
class RelayPass : public RelayPassFunctor<Relay> {
public:
    
};

}