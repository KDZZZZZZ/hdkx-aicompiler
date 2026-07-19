/*! \file test/object_test.cpp
 * \brief 验证 ObjectRef 生命周期语义和运行时类型元数据。
 */

#include "base/container.h"
#include "relay/op.h"
#include "relay/relay.h"
#include "te/te.h"
#include "tir/stmt.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#ifdef KXC_OBJECT_TEST_STANDALONE
namespace kxc {
thread_local Arena* current_arena = nullptr;
}  // namespace kxc
#endif

namespace {

#define TEST_CHECK(cond, message)                                                   \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::cerr << "[FAIL] " << (message) << "\n";                        \
            return false;                                                           \
        }                                                                           \
    } while (false)

// 记录测试对象的构造和析构次数，用于验证引用生命周期。
struct LifetimeCounters {
    int parent_destroyed{0};
    int child_destroyed{0};
};

// 析构时更新计数器的子对象节点。
class ChildNode final : public kxc::Object {
public:
    explicit ChildNode(LifetimeCounters* counters) : counters_(counters) {}

    ~ChildNode() override { ++counters_->child_destroyed; }

private:
    LifetimeCounters* counters_;
};

// 持有 ChildNode 的父对象，用于覆盖成员引用赋值场景。
class ParentNode final : public kxc::Object {
public:
    ParentNode(kxc::ObjectRef child, LifetimeCounters* counters)
        : child(std::move(child)), counters_(counters) {}

    ~ParentNode() override { ++counters_->parent_destroyed; }

    kxc::ObjectRef child;

private:
    LifetimeCounters* counters_;
};

// 暴露 SetData 的测试句柄，用于验证侵入式引用计数边界。
class TestObjectRef : public kxc::ObjectRef {
public:
    using ObjectRef::ObjectRef;

    void ResetTo(const kxc::Object* object) { SetData(object); }
};

// 要求 128 字节对齐的节点，用于验证 Arena 对齐分配。
class alignas(128) OveralignedNode final : public kxc::Object {};

// 构造时抛出异常的节点，用于验证 Arena 回滚分配。
class ThrowingNode final : public kxc::Object {
public:
    ThrowingNode() { throw std::runtime_error("expected constructor failure"); }
};

// 验证从自身持有成员复制赋值时引用计数不会提前归零。
bool TestCopyAssignmentFromOwnedMember() {
    LifetimeCounters counters;
    {
        kxc::ObjectRef parent(
            new ParentNode(kxc::ObjectRef(new ChildNode(&counters)), &counters));
        const ParentNode* node = parent.As<ParentNode>();
        TEST_CHECK(node != nullptr, "parent should have the expected dynamic type");

        parent = node->child;

        TEST_CHECK(parent.As<ChildNode>() != nullptr,
                   "copy assignment should retain the child object");
        TEST_CHECK(counters.parent_destroyed == 1,
                   "copy assignment should release the previous parent");
        TEST_CHECK(counters.child_destroyed == 0,
                   "child must stay alive after aliasing copy assignment");
    }
    TEST_CHECK(counters.child_destroyed == 1,
               "child should be destroyed exactly once after the destination releases it");
    return true;
}

// 验证从自身持有成员移动赋值时对象仍保持有效。
bool TestMoveAssignmentFromOwnedMember() {
    LifetimeCounters counters;
    {
        kxc::ObjectRef parent(
            new ParentNode(kxc::ObjectRef(new ChildNode(&counters)), &counters));
        auto* node = const_cast<ParentNode*>(parent.As<ParentNode>());
        TEST_CHECK(node != nullptr, "parent should have the expected dynamic type");

        parent = std::move(node->child);

        TEST_CHECK(parent.As<ChildNode>() != nullptr,
                   "move assignment should transfer the child object");
        TEST_CHECK(counters.parent_destroyed == 1,
                   "move assignment should release the previous parent");
        TEST_CHECK(counters.child_destroyed == 0,
                   "child must stay alive after aliasing move assignment");
    }
    TEST_CHECK(counters.child_destroyed == 1,
               "moved child should be destroyed exactly once");
    return true;
}

// 验证用相同裸节点重设 ObjectRef 不改变净引用计数。
bool TestSetDataWithSamePointer() {
    LifetimeCounters counters;
    {
        TestObjectRef ref(new ChildNode(&counters));
        ref.ResetTo(ref.get());
        TEST_CHECK(ref.As<ChildNode>() != nullptr,
                   "SetData with the current pointer should preserve the object");
        TEST_CHECK(counters.child_destroyed == 0,
                   "SetData must not release an object before reacquiring it");
    }
    TEST_CHECK(counters.child_destroyed == 1,
               "SetData object should be destroyed exactly once");
    return true;
}

// 验证 SetData 接收当前对象拥有的成员节点时保留其生命周期。
bool TestSetDataFromOwnedMember() {
    LifetimeCounters counters;
    {
        TestObjectRef parent(
            new ParentNode(kxc::ObjectRef(new ChildNode(&counters)), &counters));
        const ParentNode* node = parent.As<ParentNode>();
        TEST_CHECK(node != nullptr, "parent should have the expected dynamic type");

        parent.ResetTo(node->child.get());

        TEST_CHECK(parent.As<ChildNode>() != nullptr,
                   "SetData should retain a child owned by the old object");
        TEST_CHECK(counters.parent_destroyed == 1,
                   "SetData should release the previous parent");
        TEST_CHECK(counters.child_destroyed == 0,
                   "SetData must acquire the child before releasing its owner");
    }
    TEST_CHECK(counters.child_destroyed == 1,
               "SetData child should be destroyed exactly once");
    return true;
}

// 验证对象类型信息可按运行时编号和类型键稳定查询。
bool TestTypeInfoLookup() {
    kxc::ObjectRef relay_var(new kxc::VarNode());
    const kxc::TypeInfo& info = relay_var.get()->GetTypeInfo();

    TEST_CHECK(info.type_key() == "kxc.relay.VarNode",
               "object should expose its stable type key");
    TEST_CHECK(info.runtime_index() == relay_var.get()->GetTypeId(),
               "TypeInfo and the compatibility type id must agree");
    TEST_CHECK(kxc::TypeRegistry::FindByKey(info.type_key()) == &info,
               "registry key lookup should return the object's TypeInfo");
    TEST_CHECK(kxc::TypeRegistry::FindByIndex(info.runtime_index()) == &info,
               "registry index lookup should return the object's TypeInfo");
    TEST_CHECK(kxc::TypeRegistry::FindByKey("kxc.missing.Type") == nullptr,
               "unknown type keys should not be synthesized by lookup");
    return true;
}

// 验证同名 C++ 节点在不同命名空间中获得独立类型信息。
bool TestCollidingCppNamesHaveDistinctTypeInfo() {
    const kxc::TypeInfo& relay_var = kxc::VarNode::_type_info;
    const kxc::TypeInfo& tir_var = kxc::tir::VarNode::_type_info;
    const kxc::TypeInfo& relay_call = kxc::CallNode::_type_info;
    const kxc::TypeInfo& tir_call = kxc::tir::CallNode::_type_info;
    const kxc::TypeInfo& tir_iter = kxc::tir::IterVarNode::_type_info;
    const kxc::TypeInfo& te_iter = kxc::te::IterVarNode::_type_info;

    TEST_CHECK(relay_var.type_key() == "kxc.relay.VarNode",
               "Relay Var should use a qualified type key");
    TEST_CHECK(tir_var.type_key() == "kxc.tir.VarNode",
               "TIR Var should use a qualified type key");
    TEST_CHECK(relay_call.type_key() == "kxc.relay.CallNode",
               "Relay Call should use a qualified type key");
    TEST_CHECK(tir_call.type_key() == "kxc.tir.CallNode",
               "TIR Call should use a qualified type key");
    TEST_CHECK(tir_iter.type_key() == "kxc.tir.IterVarNode",
               "TIR IterVar should use a qualified type key");
    TEST_CHECK(te_iter.type_key() == "kxc.te.IterVarNode",
               "TE IterVar should use a qualified type key");

    TEST_CHECK(relay_var.runtime_index() != tir_var.runtime_index(),
               "Relay and TIR Var must not share a runtime index");
    TEST_CHECK(relay_call.runtime_index() != tir_call.runtime_index(),
               "Relay and TIR Call must not share a runtime index");
    TEST_CHECK(tir_iter.runtime_index() != te_iter.runtime_index(),
               "TIR and TE IterVar must not share a runtime index");
    return true;
}

// 验证 Array、Map 等容器族均通过对象系统注册类型信息。
bool TestContainerFamilyTypeInfo() {
    kxc::Array<int> ints;
    kxc::Array<kxc::String> strings;
    kxc::Map<int, int> int_map;
    kxc::Map<kxc::String, kxc::ObjectRef> object_map;
    kxc::String string("value");

    TEST_CHECK(&ints.get()->GetTypeInfo() == &strings.get()->GetTypeInfo(),
               "all Array<T> instances should share the Array family TypeInfo");
    TEST_CHECK(ints.get()->GetTypeKey() == "Array", "Array should expose its family key");
    TEST_CHECK(ints.get()->GetTypeId() == ints.get()->GetTypeInfo().runtime_index(),
               "Array TypeInfo and type id must agree");
    TEST_CHECK(&int_map.get()->GetTypeInfo() == &object_map.get()->GetTypeInfo(),
               "all Map<K, V> instances should share the Map family TypeInfo");
    TEST_CHECK(int_map.get()->GetTypeKey() == "Map", "Map should expose its family key");
    TEST_CHECK(int_map.get()->GetTypeId() == int_map.get()->GetTypeInfo().runtime_index(),
               "Map TypeInfo and type id must agree");
    TEST_CHECK(string.get()->GetTypeKey() == "String", "String should expose its type key");
    TEST_CHECK(string.get()->GetTypeId() == string.get()->GetTypeInfo().runtime_index(),
               "String TypeInfo and type id must agree");
    return true;
}

// 验证 Relay attrs 字段使用项目对象容器而非标准库容器。
bool TestRelayAttrsUseObjectContainers() {
    static_assert(std::is_same_v<decltype(kxc::relay::Conv2DAttrsNode::strides),
                                 kxc::Array<int64_t>>);
    static_assert(std::is_same_v<decltype(kxc::relay::MaxPool2DAttrsNode::pool_size),
                                 kxc::Array<int64_t>>);
    static_assert(std::is_same_v<decltype(kxc::relay::ReduceMeanAttrsNode::axes),
                                 kxc::Array<int64_t>>);
    static_assert(std::is_same_v<decltype(kxc::relay::ReshapeAttrsNode::newshape),
                                 kxc::Array<int64_t>>);
    static_assert(std::is_same_v<decltype(kxc::relay::SplitAttrsNode::split),
                                 kxc::Array<int64_t>>);
    static_assert(std::is_same_v<decltype(kxc::relay::TransposeAttrsNode::perm),
                                 kxc::Array<int64_t>>);

    const auto conv = kxc::relay::Conv2DAttrs::Create(
        {2, 2}, {1, 1, 1, 1}, {1, 1}, 1, 64, {3, 3}, "NCHW", "OIHW", "", "");
    TEST_CHECK(conv->strides.size() == 2 && conv->strides[0] == 2,
               "Array-backed Relay attrs should preserve constructor values");

    const auto reshape = kxc::relay::ReshapeAttrs::Create({0, -1});
    TEST_CHECK(reshape->newshape.size() == 2 && reshape->newshape[1] == -1,
               "Array-backed shape attrs should preserve signed dimensions");
    return true;
}

// 验证 ObjectRef::As 依赖 C++ RTTI 拒绝错误的节点转换。
bool TestAsUsesCppRtti() {
    LifetimeCounters counters;
    kxc::ObjectRef child(new ChildNode(&counters));
    TEST_CHECK(child.As<ChildNode>() != nullptr,
               "As<T> should accept the actual C++ dynamic type");
    TEST_CHECK(child.As<ParentNode>() == nullptr,
               "As<T> should reject an unrelated C++ dynamic type");
    return true;
}

// 验证嵌套 ArenaScope 退出后恢复此前线程局部 Arena。
bool TestArenaScopeRestoration() {
    kxc::Arena* initial = kxc::current_arena;
    kxc::Arena outer(1024);
    kxc::Arena inner(1024);
    {
        kxc::ArenaScope outer_scope(outer);
        TEST_CHECK(kxc::current_arena == &outer,
                   "outer ArenaScope should install its Arena");
        {
            kxc::ArenaScope inner_scope(inner);
            TEST_CHECK(kxc::current_arena == &inner,
                       "nested ArenaScope should install the inner Arena");
        }
        TEST_CHECK(kxc::current_arena == &outer,
                   "nested ArenaScope should restore the outer Arena");
    }
    TEST_CHECK(kxc::current_arena == initial,
               "ArenaScope should restore the previous TLS value");
    return true;
}

// 验证 Arena 创建的对象可安全逃逸并由引用计数延长生命周期。
bool TestArenaObjectEscapesOwner() {
    LifetimeCounters counters;
    kxc::ObjectRef escaped;
    {
        kxc::Arena arena(4096);
        kxc::ArenaScope scope(arena);
        escaped = kxc::ObjectRef(new ChildNode(&counters));
    }

    TEST_CHECK(escaped.As<ChildNode>() != nullptr,
               "Arena-backed object should survive its Arena owner");
    TEST_CHECK(counters.child_destroyed == 0,
               "escaped Arena object should remain alive while referenced");
    escaped = kxc::ObjectRef();
    TEST_CHECK(counters.child_destroyed == 1,
               "escaped Arena object should be destroyed exactly once");
    return true;
}

// 验证 Arena 空间耗尽后回退堆分配且不破坏对象生命周期。
bool TestArenaExhaustionFallsBackToHeap() {
    LifetimeCounters counters;
    kxc::ObjectRef escaped;
    {
        kxc::Arena arena(1);
        kxc::ArenaScope scope(arena);
        escaped = kxc::ObjectRef(new ChildNode(&counters));
    }

    TEST_CHECK(escaped.As<ChildNode>() != nullptr,
               "heap fallback object should survive the Arena scope");
    escaped = kxc::ObjectRef();
    TEST_CHECK(counters.child_destroyed == 1,
               "heap fallback object should be released normally");
    return true;
}

// 验证 Arena 对象可在其他线程释放并正确析构。
bool TestArenaObjectDestroyedOnAnotherThread() {
    LifetimeCounters counters;
    kxc::ObjectRef escaped;
    {
        kxc::Arena arena(4096);
        kxc::ArenaScope scope(arena);
        escaped = kxc::ObjectRef(new ChildNode(&counters));
    }

    std::thread worker([object = std::move(escaped)]() mutable {
        object = kxc::ObjectRef();
    });
    worker.join();

    TEST_CHECK(counters.child_destroyed == 1,
               "cross-thread destruction should release the Arena object once");
    return true;
}

// 验证 Arena 满足超过默认值的对象对齐要求。
bool TestOveralignedArenaObject() {
    kxc::ObjectRef object;
    {
        kxc::Arena arena(4096);
        kxc::ArenaScope scope(arena);
        object = kxc::ObjectRef(new OveralignedNode());
    }

    TEST_CHECK(reinterpret_cast<uintptr_t>(object.get()) % alignof(OveralignedNode) == 0,
               "Arena allocation should preserve over-aligned Object requirements");
    object = kxc::ObjectRef();
    return true;
}

// 验证对象构造抛出异常时 Arena 回收预留槽位。
bool TestThrowingArenaConstructorReleasesAllocation() {
    kxc::Arena arena(4096);
    kxc::ArenaScope scope(arena);
    try {
        (void)new ThrowingNode();
    } catch (const std::runtime_error&) {
        return true;
    }
    TEST_CHECK(false, "ThrowingNode construction should throw");
    return false;
}

}  // namespace

// 执行对象系统、容器和 Arena 回归矩阵。
int main() {
    if (!TestCopyAssignmentFromOwnedMember()) return EXIT_FAILURE;
    if (!TestMoveAssignmentFromOwnedMember()) return EXIT_FAILURE;
    if (!TestSetDataWithSamePointer()) return EXIT_FAILURE;
    if (!TestSetDataFromOwnedMember()) return EXIT_FAILURE;
    if (!TestTypeInfoLookup()) return EXIT_FAILURE;
    if (!TestCollidingCppNamesHaveDistinctTypeInfo()) return EXIT_FAILURE;
    if (!TestContainerFamilyTypeInfo()) return EXIT_FAILURE;
    if (!TestRelayAttrsUseObjectContainers()) return EXIT_FAILURE;
    if (!TestAsUsesCppRtti()) return EXIT_FAILURE;
    if (!TestArenaScopeRestoration()) return EXIT_FAILURE;
    if (!TestArenaObjectEscapesOwner()) return EXIT_FAILURE;
    if (!TestArenaExhaustionFallsBackToHeap()) return EXIT_FAILURE;
    if (!TestArenaObjectDestroyedOnAnotherThread()) return EXIT_FAILURE;
    if (!TestOveralignedArenaObject()) return EXIT_FAILURE;
    if (!TestThrowingArenaConstructorReleasesAllocation()) return EXIT_FAILURE;
    std::cout << "All object tests passed.\n";
    return EXIT_SUCCESS;
}
