#include "../include/tir/stmt.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;
using namespace kxc::tir;

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

int main() {
    test_tir_structure();
    return 0;
}
