/*! \file test/relay_anf_test.cpp
 * \brief Standalone checks for deterministic Relay ANF normalization.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/relay/op.h"
#include "kxc/relay/printer/print_ir.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/normalize_to_anf.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

kxc::Call Multiply(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
}

void CollectCallNames(const kxc::Expr& expr, std::vector<std::string>* names) {
    if (const auto* call = expr.As<kxc::CallNode>()) {
        const auto* op = call->op.As<kxc::relay::OpNode>();
        names->push_back(op ? op->name : "<non-op>");
        for (const kxc::Expr& argument : call->args) CollectCallNames(argument, names);
    } else if (const auto* let = expr.As<kxc::LetNode>()) {
        CollectCallNames(let->value, names);
        CollectCallNames(let->body, names);
    } else if (const auto* tuple = expr.As<kxc::TupleNode>()) {
        for (const kxc::Expr& field : tuple->fields) CollectCallNames(field, names);
    } else if (const auto* item = expr.As<kxc::TupleGetItemNode>()) {
        CollectCallNames(item->tuple, names);
    } else if (const auto* if_node = expr.As<kxc::IfNode>()) {
        CollectCallNames(if_node->cond, names);
        CollectCallNames(if_node->true_branch, names);
        CollectCallNames(if_node->false_branch, names);
    } else if (const auto* while_node = expr.As<kxc::WhileNode>()) {
        CollectCallNames(while_node->initial_state, names);
        CollectCallNames(while_node->condition, names);
        CollectCallNames(while_node->body, names);
    }
}

bool TestNestedSharedTupleIsDeterministicAndIdempotent() {
    using namespace kxc;
    const TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Call shared = Add(x, y);
    const_cast<ExprNode*>(static_cast<const ExprNode*>(shared.get()))->span =
        Span("relay_anf_test", 7, 3);
    Function input({x, y}, Tuple({Add(shared, y), Multiply(shared, x)}));
    input = relay::InferTypePass(input);

    Function normalized = relay::NormalizeToANF(input);
    std::string diagnostic;
    TEST_CHECK(relay::IsANF(normalized, &diagnostic), diagnostic);
    TEST_CHECK(relay::NormalizeToANF(normalized).get() == normalized.get(),
               "normalization must be idempotent by identity");

    std::vector<std::string> call_names;
    CollectCallNames(normalized->body, &call_names);
    TEST_CHECK(call_names == std::vector<std::string>({"add", "add", "mul"}),
               "shared Call must be bound once in left-to-right evaluation order");

    const auto* first_let = normalized->body.As<LetNode>();
    TEST_CHECK(first_let && first_let->value.As<CallNode>() &&
                   first_let->value.As<CallNode>()->span.defined(),
               "generated binding must preserve source Span metadata");
    const auto* span = first_let->value.As<CallNode>()->span.As<SpanNode>();
    TEST_CHECK(span && span->source_name == "relay_anf_test" && span->line == 7,
               "Call Span metadata changed during normalization");

    Var x2("x", type);
    Var y2("y", type);
    Call shared2 = Add(x2, y2);
    Function equivalent({x2, y2}, Tuple({Add(shared2, y2), Multiply(shared2, x2)}));
    equivalent = relay::InferTypePass(equivalent);
    TEST_CHECK(relay::printer::ToText(normalized) ==
                   relay::printer::ToText(relay::NormalizeToANF(equivalent)),
               "equivalent input must receive deterministic ANF names and order");
    return true;
}

bool TestExistingLetAndBranchesStayLexical() {
    using namespace kxc;
    const TensorType type({4}, "float32");
    Var predicate("predicate", TensorType({}, "bool"));
    Var x("x", type);
    Var y("y", type);
    Var shared("shared", type);
    Call value = Add(x, y);
    Expr body = If(predicate, Add(shared, Add(shared, x)),
                   Add(shared, Add(shared, y)));
    Function input({predicate, x, y}, Let(shared, value, body));
    input = relay::InferTypePass(input);

    Function normalized = relay::NormalizeToANF(input);
    std::string diagnostic;
    TEST_CHECK(relay::IsANF(normalized, &diagnostic), diagnostic);
    const auto* outer = normalized->body.As<LetNode>();
    TEST_CHECK(outer && outer->var.get() == shared.get(),
               "an already well-formed Let binder should remain stable");

    const auto* after_shared = outer->body.As<LetNode>();
    TEST_CHECK(after_shared && after_shared->value.As<IfNode>(),
               "If itself must be let-bound after existing Let work");
    const auto* if_node = after_shared->value.As<IfNode>();
    TEST_CHECK(if_node->true_branch.As<LetNode>() && if_node->false_branch.As<LetNode>(),
               "branch-local Calls must remain in their own lexical branches");
    return true;
}

bool TestWhileIsDeterministicAndLexical() {
    using namespace kxc;
    const TensorType type({4}, "float32");
    Var predicate("predicate", TensorType({}, "bool"));
    Var initial("initial", type), increment("increment", type), state("state", type);
    Function input({predicate, initial, increment},
        While(Add(initial, increment), state, predicate, Add(state, increment), 2));
    input = relay::InferTypePass(input);
    Function normalized = relay::NormalizeToANF(input);
    std::string diagnostic;
    TEST_CHECK(relay::IsANF(normalized, &diagnostic), diagnostic);
    TEST_CHECK(relay::NormalizeToANF(normalized).get() == normalized.get(),
               "While ANF normalization must be idempotent by identity");
    TEST_CHECK(relay::printer::ToText(normalized).find("While(max_trip_count=2, var=state)") !=
                   std::string::npos,
               "printer must deterministically include the bounded lexical loop");
    return true;
}

bool TestVerifierDiagnostics() {
    using namespace kxc;
    const TensorType type({4}, "float32");
    Var x("x", type);
    Var y("y", type);
    Function raw({x, y}, Add(x, y));
    raw = relay::InferTypePass(raw);
    std::string diagnostic;
    TEST_CHECK(!relay::IsANF(raw, &diagnostic) &&
                   diagnostic.find("must be let-bound") != std::string::npos,
               "raw executable root needs an actionable ANF diagnostic");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"nested_shared_tuple_deterministic_idempotent",
         TestNestedSharedTupleIsDeterministicAndIdempotent},
        {"existing_let_and_branch_lexicality", TestExistingLetAndBranchesStayLexical},
        {"while_deterministic_lexical", TestWhileIsDeterministicAndLexical},
        {"verifier_diagnostics", TestVerifierDiagnostics},
    };
    for (const auto& test : tests) {
        try {
            if (!test.second()) return 1;
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
