#include "../include/tir/stmt.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;
using namespace kxc::tir;

namespace kxc {
thread_local Arena* current_arena = nullptr;
}

void test_tir_structure() {
    std::cout << "Testing TIR Structure..." << std::endl;

    // 1. Expressions
    Var n("n");
    Var A("A", DataType::Handle());
    Var B("B", DataType::Handle());
    
    // index = i * 2
    Var i("i");
    PrimExpr index = i * 2; 
    assert(index.As<MulNode>() != nullptr);
    std::cout << "Expression created: i * 2" << std::endl;

    // Load: A[index]
    PrimExpr load_a = Load(A, index);
    assert(load_a.As<LoadNode>() != nullptr);
    
    // Add: A[index] + 1.0f
    PrimExpr add_res = load_a + FloatImm(1.0);
    
    // 2. Statements
    
    // Store: B[index] = A[index] + 1.0f
    Stmt store_stmt = Store(B, add_res, index);
    assert(store_stmt.As<StoreNode>() != nullptr);
    std::cout << "Store statement created." << std::endl;

    // For Loop: for (i = 0; i < n; ++i) { ... }
    Stmt for_stmt = For(i, IntImm(0), n, ForType::Serial, store_stmt);
    const ForNode* for_node = for_stmt.As<ForNode>();
    assert(for_node != nullptr);
    assert(for_node->loop_var.get() == i.get());
    std::cout << "For loop created." << std::endl;

    // Allocate: float temp[n];
    Var temp("temp", DataType::Handle());
    Stmt alloc_stmt = Allocate(temp, DataType::Float(32), {n}, IntImm(1, DataType::Bool()), for_stmt);
    assert(alloc_stmt.As<AllocateNode>() != nullptr);
    std::cout << "Allocate statement created." << std::endl;

    // AttrStmt (Thread Binding example)
    Var blockIdx("blockIdx.x");
    Stmt attr_stmt = AttrStmt(blockIdx, "thread_extent", IntImm(64), alloc_stmt);
    assert(attr_stmt.As<AttrStmtNode>() != nullptr);
    std::cout << "AttrStmt created." << std::endl;

    // Block / SeqStmt
    Stmt seq = SeqStmt({attr_stmt});
    assert(seq.As<SeqStmtNode>() != nullptr);
    std::cout << "SeqStmt created." << std::endl;

    std::cout << "PASS: TIR Structure Construction" << std::endl;
}

Stmt VectorizeSerialLoops(const Stmt& stmt) {
    if (!stmt.defined()) return stmt;

    if (auto* n = stmt.As<ForNode>()) {
        Stmt new_body = VectorizeSerialLoops(n->body);
        ForType new_type = n->for_type == ForType::Serial ? ForType::Vectorized : n->for_type;
        if (new_body.get() == n->body.get() && new_type == n->for_type) return stmt;
        return For(n->loop_var, n->min, n->extent, new_type, new_body);
    }

    if (auto* n = stmt.As<AllocateNode>()) {
        Stmt new_body = VectorizeSerialLoops(n->body);
        if (new_body.get() == n->body.get()) return stmt;
        return Allocate(n->buffer_var, n->dtype, n->extents, n->condition, new_body);
    }

    if (auto* n = stmt.As<AttrStmtNode>()) {
        Stmt new_body = VectorizeSerialLoops(n->body);
        if (new_body.get() == n->body.get()) return stmt;
        return AttrStmt(n->node, n->attr_key, n->value, new_body);
    }

    if (auto* n = stmt.As<LetStmtNode>()) {
        Stmt new_body = VectorizeSerialLoops(n->body);
        if (new_body.get() == n->body.get()) return stmt;
        return LetStmt(n->var, n->value, new_body);
    }

    if (auto* n = stmt.As<IfThenElseNode>()) {
        Stmt new_then = VectorizeSerialLoops(n->then_case);
        Stmt new_else = VectorizeSerialLoops(n->else_case);
        if (new_then.get() == n->then_case.get() && new_else.get() == n->else_case.get()) return stmt;
        return IfThenElse(n->condition, new_then, new_else);
    }

    if (auto* n = stmt.As<SeqStmtNode>()) {
        bool changed = false;
        std::vector<Stmt> new_seq;
        new_seq.reserve(n->seq.size());
        for (const auto& s : n->seq) {
            Stmt ns = VectorizeSerialLoops(s);
            if (ns.get() != s.get()) changed = true;
            new_seq.push_back(ns);
        }
        if (!changed) return stmt;
        return SeqStmt(std::move(new_seq));
    }

    return stmt;
}

void test_tir_pass_vectorize_serial_loops() {
    std::cout << "Testing TIR Pass (Vectorize Serial Loops)..." << std::endl;

    Var n("n");
    Var i("i");
    Var A("A", DataType::Handle());
    Var B("B", DataType::Handle());
    PrimExpr index = i;
    Stmt body = Store(B, Load(A, index), index);
    Stmt for_stmt = For(i, IntImm(0), n, ForType::Serial, body);

    Stmt rewritten = VectorizeSerialLoops(for_stmt);
    const ForNode* for_node = rewritten.As<ForNode>();
    assert(for_node != nullptr);
    assert(for_node->for_type == ForType::Vectorized);

    std::cout << "PASS: TIR Pass (Vectorize Serial Loops)" << std::endl;
}

int main() {
    test_tir_structure();
    test_tir_pass_vectorize_serial_loops();
    return 0;
}
