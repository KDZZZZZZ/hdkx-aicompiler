# kxcomp 基础库测试

本目录包含对 `kxcomp::base` 库的完整测试，包括文件操作和类型注册系统。

## 测试内容

### 文件操作测试
1. **read_file** - 文件读取功能测试
2. **write_file** - 文件写入功能测试  
3. **delete_file** - 文件删除功能测试
4. **集成测试** - 文件操作的完整流程测试

### 类型注册系统测试
1. **类型注册** - 验证类型自动注册功能
2. **对象创建** - 测试 `make_object` 函数
3. **引用计数** - 验证智能指针引用计数
4. **类型查询** - 测试类型信息获取
5. **多态支持** - 验证继承和多态功能

## 文件结构

```
test/
├── test_file_op.cpp          # 主测试文件
├── example_types.hpp         # 示例类型定义
├── example_types.cpp         # 示例类型实现
├── verify_structure.cpp      # 结构验证文件
├── build.bat                 # Windows编译脚本
├── Makefile                  # Linux/macOS编译文件
├── README.md                 # 本文件
├── TYPE_REGISTRATION_GUIDE.md # 类型注册系统详细指南
└── SUMMARY.md                # 测试总结文档
```

## 编译和运行

### Windows 系统

#### 使用批处理文件（推荐）

```cmd
# 运行批处理文件，会自动检测可用的编译器并运行所有测试
build.bat
```

#### 手动编译

**使用 Visual Studio 编译器：**
```cmd
# 编译主测试
cl /std:c++17 /EHsc /I..\include ..\src\base\capi.cpp test_file_op.cpp example_types.cpp /Fe:test_file_op.exe

# 编译结构验证
cl /std:c++17 /EHsc /I..\include ..\src\base\capi.cpp verify_structure.cpp /Fe:verify_structure.exe

# 运行测试
test_file_op.exe
verify_structure.exe
```

**使用 MinGW-w64 编译器：**
```cmd
# 编译主测试
g++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp test_file_op.cpp example_types.cpp -o test_file_op.exe

# 编译结构验证
g++ -std=c++17 -Wall -Wextra -I../include ../src/base/capi.cpp verify_structure.cpp -o verify_structure.exe

# 运行测试
test_file_op.exe
verify_structure.exe
```

### Linux/macOS 系统

#### 使用 Makefile

```bash
# 编译所有测试
make all

# 运行主测试
make test

# 运行结构验证
make verify

# 运行所有测试
make test-all

# 清理编译文件
make clean

# 查看帮助
make help
```

#### 手动编译

```bash
# 编译源文件
g++ -std=c++17 -Wall -Wextra -I../include -c ../src/base/capi.cpp -o capi.o
g++ -std=c++17 -Wall -Wextra -I../include -c test_file_op.cpp -o test_file_op.o
g++ -std=c++17 -Wall -Wextra -I../include -c example_types.cpp -o example_types.o
g++ -std=c++17 -Wall -Wextra -I../include -c verify_structure.cpp -o verify_structure.o

# 链接生成可执行文件
g++ capi.o test_file_op.o example_types.o -o test_file_op
g++ capi.o verify_structure.o -o verify_structure

# 运行测试
./test_file_op
./verify_structure
```

## 测试输出示例

### 主测试输出
```
开始综合测试...

=== 文件操作测试 ===
测试 read_file 功能...
✓ read_file 测试通过
测试 write_file 功能...
✓ write_file 测试通过
测试 delete_file 功能...
✓ delete_file 测试通过
测试文件操作集成功能...
✓ 文件操作集成测试通过

=== 类型注册系统测试 ===
测试类型注册系统...
✓ 类型注册系统测试通过
测试对象引用计数...
✓ 对象引用计数测试通过
测试智能指针操作...
✓ 智能指针操作测试通过

=== 类型注册系统演示 ===

=== 类型注册表信息 ===
已注册类型数量: 3
类型: Animal, 索引: 0, 哈希: 1234567890
类型: Dog, 索引: 1, 哈希: 9876543210
类型: Cat, 索引: 2, 哈希: 5555555555
========================

=== 对象创建测试 ===
创建的对象信息:
Dog: Buddy (Golden Retriever) makes sound: Woof!
Cat: Whiskers (age 3) makes sound: Meow!

对象类型信息:
Dog 类型索引: 1, 类型名: Dog
Cat 类型索引: 2, 类型名: Cat

引用计数测试:
Dog 引用计数: 1
Cat 引用计数: 1
复制后 Dog 引用计数: 2
复制对象销毁后 Dog 引用计数: 1
==================

所有测试通过！
```

## 类型注册系统使用

### 快速开始

1. **定义类**：
```cpp
class MyClass : public object {
    SIMPLE_DECLARE_TYPE(MyClass, object)
public:
    MyClass(int value) : value_(value) {}
    int GetValue() const { return value_; }
private:
    int value_;
};
```

2. **定义静态变量**：
```cpp
// 在.cpp文件中
int32_t MyClass::_type_static_index = kDynamicIndex;
```

3. **创建和使用对象**：
```cpp
auto obj = make_object<MyClass>(42);
std::cout << "Value: " << obj->GetValue() << std::endl;
std::cout << "Type: " << obj->GetTypeInfo() << std::endl;
```

### 详细指南

请参阅 `TYPE_REGISTRATION_GUIDE.md` 获取完整的使用指南，包括：
- 核心组件详解
- 完整使用步骤
- 实际代码示例
- 常见错误和解决方案
- 性能考虑和最佳实践

## 核心特性

### 1. 自动内存管理
- 基于引用计数的智能指针
- 自动对象生命周期管理
- 线程安全的引用计数操作

### 2. 运行时类型信息
- 动态类型查询
- 类型名称获取
- 全局类型注册表

### 3. 多态支持
- 完整的继承层次支持
- 虚函数调用
- 类型安全的转换

### 4. 易用性
- 简单的宏定义
- 类似std::shared_ptr的接口
- 最小化样板代码

## 注意事项

### 编译要求
- C++17兼容的编译器
- 支持原子操作
- 支持函数式编程特性

### 使用约束
1. **必须继承object**: 所有参与类型系统的类必须继承自 `object`
2. **静态变量定义**: 每个类都需要定义 `_type_static_index` 静态变量
3. **使用make_object**: 推荐使用 `make_object` 而不是直接 `new`
4. **避免循环引用**: 注意智能指针的循环引用问题

### 文件权限
- 确保测试目录有读写权限
- 测试会创建和删除临时文件
- Windows用户需要安装C++编译器

## 故障排除

### 编译错误
1. **未找到头文件**: 检查 `-I../include` 路径是否正确
2. **链接错误**: 确保所有源文件都被编译和链接
3. **未定义符号**: 检查静态变量是否在源文件中定义

### 运行时错误
1. **断言失败**: 检查文件权限和磁盘空间
2. **类型未注册**: 确保使用了正确的宏和静态变量定义
3. **内存错误**: 避免手动delete对象，使用智能指针

## 扩展和定制

框架设计为可扩展的，可以添加：
- 自定义类型转换
- 序列化支持
- 反射功能
- 属性系统
- 事件系统

欢迎贡献代码和改进建议！ 