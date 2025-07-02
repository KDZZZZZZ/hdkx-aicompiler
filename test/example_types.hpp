#pragma once

#include "../include/base/capi.hpp"
#include <string>
#include <iostream>

namespace kxcomp {
namespace base {

// 示例基类
class Animal : public object {
    SIMPLE_DECLARE_TYPE(Animal, object)
    
public:
    Animal(const std::string& name) : name_(name) {}
    virtual ~Animal() = default;
    
    virtual std::string GetName() const { return name_; }
    virtual std::string MakeSound() const { return "Unknown sound"; }
    virtual void PrintInfo() const {
        std::cout << "Animal: " << name_ << " makes sound: " << MakeSound() << std::endl;
    }
    
protected:
    std::string name_;
};

// 示例派生类 - 狗
class Dog : public Animal {
    SIMPLE_DECLARE_TYPE(Dog, Animal)
    
public:
    Dog(const std::string& name, const std::string& breed) 
        : Animal(name), breed_(breed) {}
    
    std::string MakeSound() const override { return "Woof!"; }
    std::string GetBreed() const { return breed_; }
    
    void PrintInfo() const override {
        std::cout << "Dog: " << name_ << " (" << breed_ << ") makes sound: " << MakeSound() << std::endl;
    }
    
private:
    std::string breed_;
};

// 示例派生类 - 猫
class Cat : public Animal {
    SIMPLE_DECLARE_TYPE(Cat, Animal)
    
public:
    Cat(const std::string& name, int age) 
        : Animal(name), age_(age) {}
    
    std::string MakeSound() const override { return "Meow!"; }
    int GetAge() const { return age_; }
    
    void PrintInfo() const override {
        std::cout << "Cat: " << name_ << " (age " << age_ << ") makes sound: " << MakeSound() << std::endl;
    }
    
private:
    int age_;
};

// 示例工具类
class TypeHelper {
public:
    static void PrintTypeRegistry() {
        std::cout << "\n=== 类型注册表信息 ===" << std::endl;
        TypeRegistry* registry = TypeRegistry::Global();
        std::cout << "已注册类型数量: " << registry->GetTypeCount() << std::endl;
        
        // 尝试获取已知类型的信息
        std::vector<std::string> type_names = {"Animal", "Dog", "Cat"};
        for (const auto& type_name : type_names) {
            int32_t index = registry->GetTypeIndex(type_name);
            if (index != kInvalidIndex) {
                TypeInfo* info = registry->GetTypeInfo(index);
                if (info) {
                    std::cout << "类型: " << info->type_key 
                              << ", 索引: " << index 
                              << ", 哈希: " << info->type_key_hash << std::endl;
                }
            } else {
                std::cout << "类型 " << type_name << " 未注册" << std::endl;
            }
        }
        std::cout << "========================\n" << std::endl;
    }
    
    static void TestObjectCreation() {
        std::cout << "=== 对象创建测试 ===" << std::endl;
        
        // 创建对象
        auto dog = make_object<Dog>("Buddy", "Golden Retriever");
        auto cat = make_object<Cat>("Whiskers", 3);
        
        std::cout << "创建的对象信息:" << std::endl;
        dog->PrintInfo();
        cat->PrintInfo();
        
        std::cout << "对象类型信息:" << std::endl;
        std::cout << "Dog 类型索引: " << dog->GetTypeIndex() 
                  << ", 类型名: " << dog->GetTypeInfo() << std::endl;
        std::cout << "Cat 类型索引: " << cat->GetTypeIndex() 
                  << ", 类型名: " << cat->GetTypeInfo() << std::endl;
        
        std::cout << "引用计数测试:" << std::endl;
        std::cout << "Dog 引用计数: " << dog->GetRefCount() << std::endl;
        std::cout << "Cat 引用计数: " << cat->GetRefCount() << std::endl;
        
        // 测试智能指针复制
        {
            auto dog_copy = dog;
            std::cout << "复制后 Dog 引用计数: " << dog->GetRefCount() << std::endl;
        }
        std::cout << "复制对象销毁后 Dog 引用计数: " << dog->GetRefCount() << std::endl;
        
        std::cout << "==================\n" << std::endl;
    }
};

} // namespace base
} // namespace kxcomp 