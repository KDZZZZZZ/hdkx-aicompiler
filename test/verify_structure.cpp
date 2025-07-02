#include "../include/base/capi.hpp"
#include <iostream>

using namespace kxcomp::base;

int main() {
    std::cout << "验证 file_op 类结构..." << std::endl;
    
    // 测试静态方法调用
    try {
        // 这些调用应该能够编译通过
        std::string result = file_op::read_file("nonexistent.txt");
        file_op::write_file("test.txt", "test");
        file_op::delete_file("test.txt");
        
        std::cout << "✓ file_op 类结构正确" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ file_op 类结构有问题: " << e.what() << std::endl;
        return 1;
    }
} 