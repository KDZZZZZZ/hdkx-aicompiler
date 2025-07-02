#include "../include/base/capi.hpp"
#include "example_types.hpp"
#include <iostream>
#include <cassert>

using namespace kxcomp::base;

// 文件操作测试
void test_read_file() {
    std::cout << "测试 read_file 功能..." << std::endl;
    
    std::string test_content = "Hello, World!\nThis is a test file.\n";
    std::string test_file_path = "test_file.txt";
    
    file_op::write_file(test_file_path, test_content);
    std::string read_content = file_op::read_file(test_file_path);
    
    assert(read_content == test_content);
    std::cout << "✓ read_file 测试通过" << std::endl;
    
    file_op::delete_file(test_file_path);
}

void test_write_file() {
    std::cout << "测试 write_file 功能..." << std::endl;
    
    std::string test_content = "测试写入文件内容\n第二行内容\n";
    std::string test_file_path = "test_write.txt";
    
    file_op::write_file(test_file_path, test_content);
    
    std::ifstream file(test_file_path);
    assert(file.is_open());
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    file.close();
    
    assert(content == test_content);
    std::cout << "✓ write_file 测试通过" << std::endl;
    
    file_op::delete_file(test_file_path);
}

void test_delete_file() {
    std::cout << "测试 delete_file 功能..." << std::endl;
    
    std::string test_file_path = "test_delete.txt";
    std::string test_content = "临时文件内容";
    
    file_op::write_file(test_file_path, test_content);
    
    std::ifstream file(test_file_path);
    assert(file.is_open());
    file.close();
    
    file_op::delete_file(test_file_path);
    
    std::ifstream check_file(test_file_path);
    assert(!check_file.is_open());
    check_file.close();
    
    std::cout << "✓ delete_file 测试通过" << std::endl;
}

void test_file_operations_integration() {
    std::cout << "测试文件操作集成功能..." << std::endl;
    
    std::string test_file_path = "integration_test.txt";
    std::string original_content = "原始内容\n";
    std::string updated_content = "更新后的内容\n";
    
    file_op::write_file(test_file_path, original_content);
    std::string read_content = file_op::read_file(test_file_path);
    assert(read_content == original_content);
    
    file_op::write_file(test_file_path, updated_content);
    read_content = file_op::read_file(test_file_path);
    assert(read_content == updated_content);
    
    file_op::delete_file(test_file_path);
    
    std::ifstream check_file(test_file_path);
    assert(!check_file.is_open());
    check_file.close();
    
    std::cout << "✓ 文件操作集成测试通过" << std::endl;
}

// 类型注册系统测试
void test_type_registration() {
    std::cout << "测试类型注册系统..." << std::endl;
    
    // 触发类型注册
    auto animal = make_object<Animal>("Generic Animal");
    auto dog = make_object<Dog>("Buddy", "Golden Retriever");
    auto cat = make_object<Cat>("Whiskers", 3);
    
    // 验证类型注册
    TypeRegistry* registry = TypeRegistry::Global();
    assert(registry->GetTypeIndex("Animal") != kInvalidIndex);
    assert(registry->GetTypeIndex("Dog") != kInvalidIndex);
    assert(registry->GetTypeIndex("Cat") != kInvalidIndex);
    
    // 验证对象类型信息
    assert(animal->GetTypeInfo() == "Animal");
    assert(dog->GetTypeInfo() == "Dog");
    assert(cat->GetTypeInfo() == "Cat");
    
    std::cout << "✓ 类型注册系统测试通过" << std::endl;
}

void test_object_reference_counting() {
    std::cout << "测试对象引用计数..." << std::endl;
    
    objectPtr<Dog> dog1;
    {
        auto dog = make_object<Dog>("Rex", "German Shepherd");
        assert(dog->GetRefCount() == 1);
        
        dog1 = dog;
        assert(dog->GetRefCount() == 2);
        
        auto dog2 = dog1;
        assert(dog->GetRefCount() == 3);
    }
    
    assert(dog1->GetRefCount() == 1);
    dog1.reset();
    
    std::cout << "✓ 对象引用计数测试通过" << std::endl;
}

void test_smart_pointer_operations() {
    std::cout << "测试智能指针操作..." << std::endl;
    
    auto cat = make_object<Cat>("Fluffy", 2);
    assert(cat);
    assert(cat->GetName() == "Fluffy");
    assert(cat->GetAge() == 2);
    
    // 测试移动语义
    auto cat2 = std::move(cat);
    assert(!cat);
    assert(cat2);
    assert(cat2->GetName() == "Fluffy");
    
    // 测试reset
    cat2.reset();
    assert(!cat2);
    
    std::cout << "✓ 智能指针操作测试通过" << std::endl;
}

int main() {
    std::cout << "开始综合测试..." << std::endl;
    
    try {
        // 文件操作测试
        std::cout << "\n=== 文件操作测试 ===" << std::endl;
        test_read_file();
        test_write_file();
        test_delete_file();
        test_file_operations_integration();
        
        // 类型注册系统测试
        std::cout << "\n=== 类型注册系统测试 ===" << std::endl;
        test_type_registration();
        test_object_reference_counting();
        test_smart_pointer_operations();
        
        // 演示类型注册系统
        std::cout << "\n=== 类型注册系统演示 ===" << std::endl;
        TypeHelper::PrintTypeRegistry();
        TypeHelper::TestObjectCreation();
        
        std::cout << "所有测试通过！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "测试失败: 未知异常" << std::endl;
        return 1;
    }
} 