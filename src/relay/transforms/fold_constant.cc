#include "relay/transforms/fold_constant.h"

#include <cmath>
#include <string>

#include "base/pass.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class FoldConstantRewriter : public RelayPass {
protected:
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
        } else if (op_name == "subtract" || op_name == "sub") {
            out_value = lhs - rhs;
        } else if (op_name == "mul" || op_name == "multiply") {
            out_value = lhs * rhs;
        } else if (op_name == "divide" || op_name == "div") {
            if (rhs == 0.0) {
                return rewritten;
            }
            out_value = lhs / rhs;
        } else if (op_name == "equal") {
            out_value = lhs == rhs ? 1.0 : 0.0;
            out_dtype = DLDataType{kDLUint, 1, 1};
        } else if (op_name == "greater") {
            out_value = lhs > rhs ? 1.0 : 0.0;
            out_dtype = DLDataType{kDLUint, 1, 1};
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

Function FoldConstantPass(const Function& func) {
    FoldConstantRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc

