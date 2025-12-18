#include "../include/base/pass.h"
#include "../include/base/relay.h"
#include "../include/base/op.h"
#include "../include/relay/op_macros.h"
#include <iostream>
#include <cassert>

using namespace kxc;

namespace kxc {
thread_local Arena* current_arena = nullptr;
}

// A simple pass that replaces a specific variable with another expression
class VarSubstitutor : public RelayPass {
public:
    VarSubstitutor(Var target, Expr replacement) 
        : target_(target), replacement_(replacement) {}

    // Override using Expr
    Expr VisitVar(const VarNode* op, const Expr& ref) override {
        if (op->vid->name_hint == target_->vid->name_hint) {
            return replacement_;
        }
        return ref;
    }

private:
    Var target_;
    Expr replacement_;
};

void test_pass_infrastructure() {
    std::cout << "Testing Pass Infrastructure..." << std::endl;

    // 1. Setup Graph: f(x) = x
    Var x("x");
    Var y("y");
    
    // Graph: Call(op, {x})
    // We'll use a dummy Op
    Var op_var("dummy_op"); 
    std::vector<Expr> args = {x};
    Call call(op_var, args);

    // 2. Run Pass: Replace x with y
    VarSubstitutor substitutor(x, y);
    Expr new_graph = substitutor.Mutate(call);

    // 3. Verify
    // The new graph should be Call(op, {y})
    const CallNode* new_call_node = new_graph.As<CallNode>();
    assert(new_call_node != nullptr);
    assert(new_call_node->args.size() == 1);
    
    const VarNode* arg0 = new_call_node->args[0].As<VarNode>();
    assert(arg0 != nullptr);
    assert(arg0->vid->name_hint == "y"); // Should be y now
    
    std::cout << "Variable substitution successful: x -> y" << std::endl;

    // 4. Test Copy-On-Write (No change)
    // Run pass again replacing 'z' (which doesn't exist) -> should return original ref
    Var z("z");
    VarSubstitutor no_op_pass(z, y);
    Expr same_graph = no_op_pass.Mutate(call);
    
    assert(same_graph.get() == call.get()); // Pointers should be identical
    std::cout << "Copy-On-Write verification successful." << std::endl;

    std::cout << "PASS: Pass Infrastructure" << std::endl;
}

class AnnotateDevice : public RelayPass {
public:
    explicit AnnotateDevice(ObjectRef device) : device_(std::move(device)) {}

protected:
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        auto new_op = Mutate(op->op);
        std::vector<Expr> new_args;
        bool changed = (new_op.get() != op->op.get());

        for (const auto& arg : op->args) {
            auto new_arg = Mutate(arg);
            if (new_arg.get() != arg.get()) changed = true;
            new_args.push_back(new_arg);
        }

        ObjectRef new_attrs = op->attrs.defined() ? op->attrs : device_;
        if (new_attrs.get() != op->attrs.get()) changed = true;

        if (!changed) return ref;
        return Call(new_op, new_args, new_attrs);
    }

private:
    ObjectRef device_;
};

class TestTargetNode : public Object {
public:
    std::string kind;
    int id;

    explicit TestTargetNode(std::string kind, int id) : kind(std::move(kind)), id(id) {}

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 901; }
};

class TestTarget : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    TestTarget(std::string kind, int id) {
        object_ = new TestTargetNode(std::move(kind), id);
        if (object_) object_->IncRef();
    }
    const TestTargetNode* operator->() const { return static_cast<const TestTargetNode*>(object_); }
};

void test_device_annotation_pass() {
    std::cout << "Testing Device Annotation Pass..." << std::endl;

    Var x("x");
    Var op_var("add");
    Expr add_call = Call(op_var, {x});

    ObjectRef gpu0 = TestTarget("cuda", 0);
    AnnotateDevice pass(gpu0);
    Expr annotated = pass.Mutate(add_call);

    const CallNode* call_node = annotated.As<CallNode>();
    assert(call_node != nullptr);
    const TestTargetNode* dev = call_node->attrs.As<TestTargetNode>();
    assert(dev != nullptr);
    assert(dev->kind == "cuda");
    assert(dev->id == 0);

    Expr already = Call(op_var, {x}, TestTarget("cpu", 0));
    Expr keep = pass.Mutate(already);
    const CallNode* keep_node = keep.As<CallNode>();
    assert(keep_node != nullptr);
    const TestTargetNode* keep_dev = keep_node->attrs.As<TestTargetNode>();
    assert(keep_dev != nullptr);
    assert(keep_dev->kind == "cpu");
    assert(keep.get() == already.get());

    std::cout << "PASS: Device Annotation Pass" << std::endl;
}

int main() {
    test_pass_infrastructure();
    test_device_annotation_pass();
    return 0;
}
