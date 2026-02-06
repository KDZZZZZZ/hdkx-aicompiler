#include "te/te.h"
#include "base/arena.h"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <sstream>
#include <iomanip>

// ============================================================================
// Minimal Test Framework (mimicking GoogleTest/Catch2)
// ============================================================================

namespace kxc {
    thread_local Arena* current_arena = nullptr;
}

namespace kxc {
namespace test {


class TestRegistry {
public:
    using TestFunc = std::function<void()>;
    struct TestInfo {
        std::string name;
        TestFunc func;
    };

    static TestRegistry& Instance() {
        static TestRegistry instance;
        return instance;
    }

    void Register(const std::string& name, TestFunc func) {
        tests_.push_back({name, func});
    }

    int RunAll() {
        int passed = 0;
        int failed = 0;
        std::cout << "[==========] Running " << tests_.size() << " tests." << std::endl;
        
        for (const auto& test : tests_) {
            std::cout << "[ RUN      ] " << test.name << std::endl;
            try {
                test.func();
                std::cout << "[       OK ] " << test.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                std::cout << "[  FAILED  ] " << test.name << "\n" << e.what() << std::endl;
                failed++;
            } catch (...) {
                std::cout << "[  FAILED  ] " << test.name << "\nUnknown exception" << std::endl;
                failed++;
            }
        }
        
        std::cout << "[==========] " << tests_.size() << " tests ran." << std::endl;
        std::cout << "[  PASSED  ] " << passed << " tests." << std::endl;
        if (failed > 0) {
            std::cout << "[  FAILED  ] " << failed << " tests." << std::endl;
        }
        return failed;
    }

private:
    std::vector<TestInfo> tests_;
};

class TestRegisterer {
public:
    TestRegisterer(const std::string& name, TestRegistry::TestFunc func) {
        TestRegistry::Instance().Register(name, func);
    }
};

class TestException : public std::exception {
public:
    TestException(const std::string& msg) : msg_(msg) {}
    const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

#define TEST(TestSuite, TestName) \
    void TestSuite##_##TestName(); \
    kxc::test::TestRegisterer global_##TestSuite##_##TestName(#TestSuite "." #TestName, TestSuite##_##TestName); \
    void TestSuite##_##TestName()

#define EXPECT_TRUE(condition) \
    if (!(condition)) { \
        std::stringstream ss; \
        ss << "Expected true: " #condition << "\n" << __FILE__ << ":" << __LINE__; \
        std::cerr << ss.str() << std::endl; \
    }

#define EXPECT_EQ(val1, val2) \
    if ((val1) != (val2)) { \
        std::stringstream ss; \
        ss << "Expected equality of these values:\n  " #val1 "\n    Which is: " << (val1) << "\n  " #val2 "\n    Which is: " << (val2) << "\n" << __FILE__ << ":" << __LINE__; \
        std::cerr << ss.str() << std::endl; \
    }

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::stringstream ss; \
        ss << "Assertion failed: " #condition << "\n" << __FILE__ << ":" << __LINE__; \
        throw kxc::test::TestException(ss.str()); \
    }

#define ASSERT_EQ(val1, val2) \
    if ((val1) != (val2)) { \
        std::stringstream ss; \
        ss << "Assertion failed:\n  " #val1 "\n    Which is: " << (val1) << "\n  " #val2 "\n    Which is: " << (val2) << "\n" << __FILE__ << ":" << __LINE__; \
        throw kxc::test::TestException(ss.str()); \
    }

} // namespace test
} // namespace kxc

// ============================================================================
// Test Suite: TE (Tensor Expression)
// ============================================================================

using namespace kxc;
using namespace kxc::te;
using namespace kxc::tir;

// Helper to print DataType
inline std::ostream& operator<<(std::ostream& os, const DataType& t) {
    os << "DataType(";
    if (t.code == 0) os << "Int";
    else if (t.code == 1) os << "UInt";
    else if (t.code == 2) os << "Float";
    else if (t.code == 3) os << "Handle";
    else if (t.code == 4) os << "Void";
    else os << "Unknown";
    os << ", " << (int)t.bits << ", " << t.lanes << ")";
    return os;
}

TEST(Tensor, CreationAndProperties) {

    Var n("n"), m("m");
    std::vector<PrimExpr> shape = {n, m};
    Tensor A = placeholder(shape, DataType::Float(32), "A");

    ASSERT_EQ(A->name, "A");
    ASSERT_EQ(A->shape.size(), 2);
    // PrimExpr equality might not be overloaded for simple comparison, but let's assume checking size is enough for now
    // or we check if they are the same Var.
    // Since Var equality is pointer based in this codebase (ObjectRef), it should work if we reused the vars.
    // Wait, PrimExpr wrapping Var might create new wrapper? No, Var is ObjectRef.
    // PrimExpr is wrapping ObjectRef.
    
    ASSERT_EQ(A->dtype, DataType::Float(32));
    ASSERT_EQ(A->value_index, 0);
    ASSERT_TRUE(A->op.defined());
    ASSERT_TRUE(A->op.As<PlaceholderOpNode>() != nullptr);
}

TEST(Operation, PlaceholderOp) {
    Var n("n");
    Tensor A = placeholder({n}, DataType::Int(32), "Input");
    
    ASSERT_TRUE(A->op.As<PlaceholderOpNode>());
    auto op = A->op.As<PlaceholderOpNode>();
    
    ASSERT_EQ(op->name, "Input");
    ASSERT_EQ(op->num_outputs(), 1);
    ASSERT_EQ(op->output_dtype(0), DataType::Int(32));
    ASSERT_EQ(op->output_shape(0).size(), 1);
}

TEST(Operation, ComputeOpBasic) {
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    
    // B(i, j) = A(i, j) + 1.0
    Tensor B = compute({n, m}, [&](const std::vector<Var>& axis) {
        Var i = axis[0];
        Var j = axis[1];
        return A(i, j) + 1.0f;
    }, "B");

    ASSERT_TRUE(B->op.As<ComputeOpNode>());
    auto op = B->op.As<ComputeOpNode>();
    
    ASSERT_EQ(op->name, "B");
    ASSERT_EQ(op->axis.size(), 2);
    ASSERT_EQ(op->body.size(), 1);
    ASSERT_EQ(op->reduce_axis.size(), 0);
}

TEST(Operation, ComputeOpReduction) {
    Var n("n"), k("k");
    Tensor A = placeholder({n, k}, DataType::Float(32), "A");
    
    // Sum over k
    // C(i) = sum(A(i, k), axis=k)
    IterVar rv = reduce_axis(0, k, "rv");
    
    Tensor C = compute({n}, [&](const std::vector<Var>& axis) {
        Var i = axis[0];
        return sum(A(i, rv), {rv});
    }, "C");
    
    ASSERT_TRUE(C->op.As<ComputeOpNode>());
    auto op = C->op.As<ComputeOpNode>();
    
    ASSERT_EQ(op->name, "C");
    ASSERT_EQ(op->axis.size(), 1);
    // reduce_axis should be detected
    ASSERT_EQ(op->reduce_axis.size(), 1);
    ASSERT_TRUE(op->reduce_axis[0] == rv);
}

TEST(Schedule, CreateSchedule) {
    Var n("n");
    Tensor A = placeholder({n}, DataType::Float(32), "A");
    Tensor B = compute({n}, [&](const std::vector<Var>& axis) {
        return A(axis) * 2.0f;
    }, "B");
    
    Schedule s = create_schedule({B->op});
    
    ASSERT_EQ(s->outputs.size(), 1);
    ASSERT_TRUE(s->outputs[0] == B->op);
    
    Stage stage_b = s[B->op];
    ASSERT_TRUE(stage_b.defined());
    ASSERT_TRUE(stage_b->op == B->op);
}

TEST(Schedule, SplitPrimitive) {
    Var n("n");
    Tensor A = placeholder({n}, DataType::Float(32), "A");
    Tensor B = compute({n}, [&](const std::vector<Var>& axis) {
        return A(axis);
    }, "B");
    
    Schedule s = create_schedule({B->op});
    Stage stage = s[B->op];
    
    ASSERT_EQ(stage->leaf_iter_vars.size(), 1);
    IterVar outer, inner;
    stage.split(stage->leaf_iter_vars[0], 32, &outer, &inner);
    
    ASSERT_EQ(stage->leaf_iter_vars.size(), 2);
    ASSERT_TRUE(stage->leaf_iter_vars[0] == outer);
    ASSERT_TRUE(stage->leaf_iter_vars[1] == inner);
}

TEST(Schedule, FusePrimitive) {
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    Tensor B = compute({n, m}, [&](const std::vector<Var>& axis) {
        return A(axis);
    }, "B");
    
    Schedule s = create_schedule({B->op});
    Stage stage = s[B->op];
    
    ASSERT_EQ(stage->leaf_iter_vars.size(), 2);
    IterVar fused = stage.fuse(stage->leaf_iter_vars[0], stage->leaf_iter_vars[1]);
    
    ASSERT_EQ(stage->leaf_iter_vars.size(), 1);
    ASSERT_TRUE(stage->leaf_iter_vars[0] == fused);
}

int main() {
    // Initialize any global state if necessary
    return kxc::test::TestRegistry::Instance().RunAll();
}
