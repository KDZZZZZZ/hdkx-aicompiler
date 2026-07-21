/*! \file src/relay/transforms/fold_constant.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "relay/transforms/fold_constant.h"

#include <cmath>
#include <string>

#include "base/pass.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

// 重写可在编译期求值的二元标量调用，并保留原表达式的设备放置信息。
class FoldConstantRewriter : public RelayPass {
protected:
    // 递归改写调用，在两个参数均为标量常量时执行受支持的算术或比较。
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitCall(op, ref);
        const auto* call = rewritten.As<CallNode>();
        if (!call || call->args.size() != 2) {
            return rewritten;
        }

        const std::string op_name = pass_utils::GetCallOpName(call);
        if (op_name.empty()) {
            return rewritten;
        }

        double lhs = 0.0;
        double rhs = 0.0;
        DLDataType lhs_dtype{0, 0, 0};
        DLDataType rhs_dtype{0, 0, 0};
        if (!pass_utils::TryGetScalarConstantValueWithDType(call->args[0], &lhs, &lhs_dtype) ||
            !pass_utils::TryGetScalarConstantValueWithDType(call->args[1], &rhs, &rhs_dtype)) {
            return rewritten;
        }

        bool matched = true;
        double out_value = 0.0;
        DLDataType out_dtype = lhs_dtype;

        if (op_name == "add") {
            out_value = lhs + rhs;
        } else if (op_name == "subtract") {
            out_value = lhs - rhs;
        } else if (op_name == "mul") {
            out_value = lhs * rhs;
        } else if (op_name == "divide") {
            if (rhs == 0.0) {
                return rewritten;
            }
            out_value = lhs / rhs;
        } else if (op_name == "equal") {
            out_value = lhs == rhs ? 1.0 : 0.0;
            out_dtype = DLDataType{kDLBool, 8, 1};
        } else if (op_name == "greater") {
            out_value = lhs > rhs ? 1.0 : 0.0;
            out_dtype = DLDataType{kDLBool, 8, 1};
        } else {
            matched = false;
        }

        if (!matched || !std::isfinite(out_value)) {
            return rewritten;
        }

        Expr constant = pass_utils::MakeScalarConstant(out_value, out_dtype);
        return pass_utils::CopyVirtualDevice(rewritten, constant);
    }
};

}  // namespace

// 对函数执行一次标量常量折叠并返回新 Relay 函数。
Function FoldConstantPass(const Function& func) {
    FoldConstantRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc

