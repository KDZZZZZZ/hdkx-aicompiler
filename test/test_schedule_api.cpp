#include "te/te.h"
#include "te/topi.h"
#include <iostream>
#include <cassert>

using namespace kxc;
using namespace kxc::te;
using namespace kxc::topi;
using namespace kxc::tir;

namespace kxc {
    thread_local Arena* current_arena = nullptr;
}

void test_schedule_split() {
    std::cout << "\nTesting Schedule Split..." << std::endl;
    Var n("n");
    Tensor A = placeholder({n}, DataType::Float(32), "A");
    Tensor B = compute({n}, [&](const std::vector<Var>& i) { return A(i) * 2.0f; }, "B");
    
    Schedule s = create_schedule({B->op});
    Stage stage = s[B];
    
    // Check initial iter vars
    assert(stage->leaf_iter_vars.size() == 1);
    IterVar axis0 = stage->leaf_iter_vars[0];
    
    // Split
    IterVar outer, inner;
    stage.split(axis0, 4, &outer, &inner);
    
    // Check result
    assert(stage->leaf_iter_vars.size() == 2);
    // Explicitly check string find result is not npos
    bool found_outer = stage->leaf_iter_vars[0]->var->name_hint.find(".outer") != std::string::npos;
    bool found_inner = stage->leaf_iter_vars[1]->var->name_hint.find(".inner") != std::string::npos;
    assert(found_outer);
    assert(found_inner);
    
    std::cout << "Split passed." << std::endl;
}

void test_schedule_tile_vectorize() {
    std::cout << "\nTesting Schedule Tile & Vectorize (MatMul)..." << std::endl;
    Var M("M"), N("N"), K("K");
    Tensor A = placeholder({M, K}, DataType::Float(32), "A");
    Tensor B = placeholder({K, N}, DataType::Float(32), "B");
    Tensor C = matmul(A, B);
    
    Schedule s = create_schedule({C->op});
    Stage stage = s[C];
    
    // MatMul has 2 spatial axes (M, N) and 1 reduce axis (K)
    // leaf_iter_vars: [i, j, k]
    assert(stage->leaf_iter_vars.size() == 3);
    
    IterVar y = stage->leaf_iter_vars[0]; // M
    IterVar x = stage->leaf_iter_vars[1]; // N
    IterVar k = stage->leaf_iter_vars[2]; // K
    
    IterVar yo, yi, xo, xi;
    stage.tile(y, x, 32, 32, &yo, &xo, &yi, &xi);
    
    // After tile: [yo, xo, yi, xi, k]
    assert(stage->leaf_iter_vars.size() == 5);
    // Explicit pointer comparison to avoid ambiguity
    assert(stage->leaf_iter_vars[0].get() == yo.get());
    assert(stage->leaf_iter_vars[1].get() == xo.get());
    
    // Vectorize inner loop
    stage.vectorize(xi);
    assert(xi->iter_type == IterVarType::kVectorized);

    // Unroll
    stage.unroll(yi);
    assert(yi->iter_type == IterVarType::kUnrolled);
    
    std::cout << "Tile & Vectorize & Unroll passed." << std::endl;
}

void test_schedule_fuse_parallel() {
    std::cout << "\nTesting Schedule Fuse & Parallel..." << std::endl;
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    Tensor B = compute({n, m}, [&](const std::vector<Var>& idx) { return A(idx); }, "B");
    
    Schedule s = create_schedule({B->op});
    Stage stage = s[B];
    
    IterVar i = stage->leaf_iter_vars[0];
    IterVar j = stage->leaf_iter_vars[1];
    
    // Fuse
    IterVar fused = stage.fuse(i, j);
    assert(stage->leaf_iter_vars.size() == 1);
    assert(stage->leaf_iter_vars[0].get() == fused.get());
    
    // Parallel
    stage.parallel(fused);
    assert(fused->iter_type == IterVarType::kParallel);
    
    std::cout << "Fuse & Parallel passed." << std::endl;
}

void test_schedule_bind_thread_axis() {
    std::cout << "\nTesting Schedule Bind (Thread Axis)..." << std::endl;

    Var n("n");
    Tensor A = placeholder({n}, DataType::Float(32), "A");
    Tensor B = compute({n}, [&](const std::vector<Var>& i) { return A(i); }, "B");

    Schedule s = create_schedule({B->op});
    Stage stage = s[B];

    assert(stage->leaf_iter_vars.size() == 1);
    IterVar axis0 = stage->leaf_iter_vars[0];

    IterVar outer, inner;
    stage.split(axis0, 64, &outer, &inner);

    IterVar tx = thread_axis(IntImm(64), "threadIdx.x");
    stage.bind(inner, tx);

    assert(inner->iter_type == IterVarType::kThreadIndex);
    assert(inner->thread_tag == "threadIdx.x");

    std::cout << "Bind passed." << std::endl;
}

int main() {
    try {
        test_schedule_split();
        test_schedule_tile_vectorize();
        test_schedule_fuse_parallel();
        test_schedule_bind_thread_axis();
        std::cout << "\nAll Schedule API tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
