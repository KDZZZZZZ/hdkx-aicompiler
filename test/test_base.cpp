#include "../include/base/arena.h"
#include "../include/base/object.h"
#include "../include/base/typemanager.h"
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace kxc;

static int g_destruct_count = 0;

class MyObj : public Object {
public:
    MyObj() = default;
    ~MyObj() { ++g_destruct_count; }
    const TypeIndex GetTypeId() const override { return 42; }
};

static Object* MyObjFactory() { return new MyObj(); }
static TypeRegistrar g_myobj_registrar("MyObj", &MyObjFactory);

void test_arena_allocation_basic() {
    std::cout << "Testing Arena basic allocation..." << std::endl;

    Arena arena(256);
    void* p1 = arena.Allocate(16, alignof(void*));
    assert(p1 != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(p1) % alignof(void*) == 0);

    void* p2 = arena.Allocate(32, 16);
    assert(p2 != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(p2) % 16 == 0);

    void* p3 = arena.Allocate(1024, alignof(void*));
    assert(p3 == nullptr);

    std::cout << "PASS: Arena allocation tests" << std::endl;
}

void test_object_allocation_with_arena() {
    std::cout << "Testing Object allocation on Arena..." << std::endl;

    g_destruct_count = 0;
    Arena arena(1024);
    current_arena = &arena;

    {
        ObjectRef ref(new MyObj());
        assert(ref.get() != nullptr);
    }

    current_arena = nullptr;
    assert(g_destruct_count == 1);

    std::cout << "PASS: Object allocation on Arena" << std::endl;
}

void test_objectref_copy_move_lifecycle() {
    std::cout << "Testing ObjectRef copy/move and lifecycle..." << std::endl;

    g_destruct_count = 0;
    {
        ObjectRef r1(new MyObj());
        assert(r1.get() != nullptr);
        {
            ObjectRef r2 = r1;
            ObjectRef r3 = std::move(r2);
            assert(r3.get() != nullptr);
        }
        assert(g_destruct_count == 0);
    }
    assert(g_destruct_count == 1);

    std::cout << "PASS: ObjectRef lifecycle" << std::endl;
}

void test_type_manager_registration_and_creation() {
    std::cout << "Testing TypeManager registration and creation..." << std::endl;

    g_destruct_count = 0;
    ObjectRef obj = TypeManager::Get()->CreateObject("MyObj");
    assert(obj.get() != nullptr);

    {
        ObjectRef obj2 = obj;
        assert(obj2.get() != nullptr);
    }

    assert(g_destruct_count == 0);
    obj = ObjectRef();
    assert(g_destruct_count == 1);

    ObjectRef unknown = TypeManager::Get()->CreateObject("UnknownType");
    assert(unknown.get() == nullptr);

    std::cout << "PASS: TypeManager" << std::endl;
}

int main() {
    try {
        std::cout << "\n=== include/base unit tests ===" << std::endl;
        test_arena_allocation_basic();
        test_object_allocation_with_arena();
        test_objectref_copy_move_lifecycle();
        test_type_manager_registration_and_creation();
        std::cout << "All include/base tests passed!" << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
}
