#include "../include/base/object.h"
#include "../include/base/typemanager.h"
#include <cassert>
#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <typeinfo>

using namespace kxc;

struct BaseObj : Object { const TypeIndex GetTypeId() const override { return 1; } virtual ~BaseObj() = default; };
struct DerivedA : BaseObj { const TypeIndex GetTypeId() const override { return 2; } };
struct DerivedB : BaseObj { const TypeIndex GetTypeId() const override { return 3; } };

using ObserverFn = std::function<void(const std::string&, const Object*)>;
static std::vector<ObserverFn> observers;
static std::vector<std::string> events;

static Object* FactoryA() { auto* p = new DerivedA(); for (auto& fn : observers) fn("A", p); events.push_back("A"); return p; }
static Object* FactoryB() { auto* p = new DerivedB(); for (auto& fn : observers) fn("B", p); events.push_back("B"); return p; }

static TypeRegistrar regA("TypeA", &FactoryA);
static TypeRegistrar regB("TypeB", &FactoryB);

void test_rtti() {
    ObjectRef a(new DerivedA());
    const Object* ao = a.get();
    auto ba = dynamic_cast<const BaseObj*>(ao);
    auto da = dynamic_cast<const DerivedA*>(ao);
    auto db = dynamic_cast<const DerivedB*>(ao);
    assert(ba != nullptr);
    assert(da != nullptr);
    assert(db == nullptr);
    assert(typeid(*ao) == typeid(DerivedA));
    assert(da->GetTypeId() == 2);
}

void test_observer() {
    observers.clear();
    events.clear();
    int c1 = 0, c2 = 0;
    observers.push_back([&](const std::string& key, const Object* obj){ ++c1; });
    observers.push_back([&](const std::string& key, const Object* obj){ ++c2; });
    ObjectRef o1 = TypeManager::Get()->CreateObject("TypeA");
    ObjectRef o2 = TypeManager::Get()->CreateObject("TypeB");
    assert(c1 == 2);
    assert(c2 == 2);
    assert(events.size() == 2);
    assert(events[0] == "A");
    assert(events[1] == "B");
}

int main() {
    test_rtti();
    test_observer();
    std::cout << "PASS: RTTI and observer tests" << std::endl;
    return 0;
}

