# 类型注册系统使用指南

## 概述

kxcomp框架提供了一个强大的类型注册系统，支持运行时类型信息(RTTI)、自动内存管理和智能指针。

## 核心组件

### 1. 基础类 `object`
所有需要参与类型注册的类都必须继承自 `object` 基类。

```cpp
class object {
public:
    // 获取类型索引
    int32_t GetTypeIndex() const;
    
    // 获取类型名称
    std::string GetTypeInfo() const;
    
    // 引用计数管理
    void IncRef();
    void DecRef();
    int32_t GetRefCount() const;
    
    // 设置类型索引和删除器
    void SetTypeIndex(int32_t index);
    void SetDeleter(std::function<void(void*)> deleter);
};
```

### 2. 类型注册表 `TypeRegistry`
全局单例，管理所有已注册的类型信息。

```cpp
class TypeRegistry {
public:
    static TypeRegistry* Global();
    
    // 注册类型
    int32_t RegisterType(const TypeInfo& type_info);
    
    // 查询类型
    int32_t GetTypeIndex(const std::string& key);
    TypeInfo* GetTypeInfo(int32_t index);
    size_t GetTypeCount() const;
};
```

### 3. 智能指针 `objectPtr<T>`
自动管理对象生命周期的智能指针。

```cpp
template<typename T>
class objectPtr {
public:
    // 构造和析构
    objectPtr();
    explicit objectPtr(T* data);
    objectPtr(const objectPtr& other);
    objectPtr(objectPtr&& other);
    ~objectPtr();
    
    // 赋值操作
    objectPtr& operator=(const objectPtr& other);
    objectPtr& operator=(objectPtr&& other);
    
    // 访问操作
    T* operator->() const;
    T& operator*() const;
    T* get() const;
    explicit operator bool() const;
    
    // 管理操作
    T* release();
    void reset(T* ptr = nullptr);
};
```

## 使用步骤

### 步骤1：定义类并声明类型

使用 `SIMPLE_DECLARE_TYPE` 宏在类中声明类型信息：

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

### 步骤2：在源文件中定义静态变量

```cpp
// 在 .cpp 文件中定义
int32_t MyClass::_type_static_index = kDynamicIndex;
```

### 步骤3：创建对象

使用 `make_object` 函数创建对象：

```cpp
auto obj = make_object<MyClass>(42);
```

### 步骤4：使用对象

```cpp
// 访问对象成员
std::cout << "Value: " << obj->GetValue() << std::endl;

// 获取类型信息
std::cout << "Type: " << obj->GetTypeInfo() << std::endl;
std::cout << "Type Index: " << obj->GetTypeIndex() << std::endl;

// 引用计数
std::cout << "Ref Count: " << obj->GetRefCount() << std::endl;
```

## 完整示例

### 头文件 (example.hpp)
```cpp
#pragma once
#include "capi.hpp"

namespace kxcomp {
namespace base {

// 基类
class Shape : public object {
    SIMPLE_DECLARE_TYPE(Shape, object)
    
public:
    Shape(const std::string& name) : name_(name) {}
    virtual ~Shape() = default;
    
    virtual double GetArea() const = 0;
    virtual std::string GetName() const { return name_; }
    
protected:
    std::string name_;
};

// 派生类
class Circle : public Shape {
    SIMPLE_DECLARE_TYPE(Circle, Shape)
    
public:
    Circle(const std::string& name, double radius) 
        : Shape(name), radius_(radius) {}
    
    double GetArea() const override {
        return 3.14159 * radius_ * radius_;
    }
    
    double GetRadius() const { return radius_; }
    
private:
    double radius_;
};

class Rectangle : public Shape {
    SIMPLE_DECLARE_TYPE(Rectangle, Shape)
    
public:
    Rectangle(const std::string& name, double width, double height)
        : Shape(name), width_(width), height_(height) {}
    
    double GetArea() const override {
        return width_ * height_;
    }
    
    double GetWidth() const { return width_; }
    double GetHeight() const { return height_; }
    
private:
    double width_, height_;
};

} // namespace base
} // namespace kxcomp
```

### 源文件 (example.cpp)
```cpp
#include "example.hpp"

namespace kxcomp {
namespace base {

// 定义静态变量
int32_t Shape::_type_static_index = kDynamicIndex;
int32_t Circle::_type_static_index = kDynamicIndex;
int32_t Rectangle::_type_static_index = kDynamicIndex;

} // namespace base
} // namespace kxcomp
```

### 使用示例 (main.cpp)
```cpp
#include "example.hpp"
#include <iostream>
#include <vector>

using namespace kxcomp::base;

int main() {
    // 创建对象
    auto circle = make_object<Circle>("Circle1", 5.0);
    auto rectangle = make_object<Rectangle>("Rect1", 3.0, 4.0);
    
    // 使用多态
    std::vector<objectPtr<Shape>> shapes;
    shapes.push_back(circle);
    shapes.push_back(rectangle);
    
    // 遍历并使用
    for (const auto& shape : shapes) {
        std::cout << "Shape: " << shape->GetName() 
                  << ", Type: " << shape->GetTypeInfo()
                  << ", Area: " << shape->GetArea() << std::endl;
    }
    
    // 类型查询
    TypeRegistry* registry = TypeRegistry::Global();
    std::cout << "Total registered types: " << registry->GetTypeCount() << std::endl;
    
    return 0;
}
```

## 重要注意事项

### 1. 静态变量定义
每个使用 `SIMPLE_DECLARE_TYPE` 的类都必须在源文件中定义静态变量：
```cpp
int32_t ClassName::_type_static_index = kDynamicIndex;
```

### 2. 类型注册时机
类型注册发生在第一次调用 `RuntimeTypeIndex()` 时，通常是在 `make_object` 调用时。

### 3. 内存管理
- 使用 `objectPtr` 自动管理内存
- 对象通过引用计数自动销毁
- 避免直接使用 `new` 和 `delete`

### 4. 继承关系
- 所有参与类型系统的类必须继承自 `object`
- 支持多级继承
- 虚析构函数确保正确清理

### 5. 线程安全
- 引用计数操作是原子的
- 类型注册使用静态变量，需要注意多线程初始化

## 常见错误和解决方案

### 错误1：未定义静态变量
**错误信息**: 链接错误，未定义符号
**解决方案**: 在源文件中添加静态变量定义

### 错误2：忘记继承object
**错误信息**: 编译错误，缺少基类成员
**解决方案**: 确保类继承自 `object`

### 错误3：直接使用new创建对象
**问题**: 对象不会自动注册类型信息
**解决方案**: 使用 `make_object` 函数

### 错误4：循环引用
**问题**: 对象无法自动销毁
**解决方案**: 使用弱引用或手动break循环

## 性能考虑

1. **类型注册开销**: 每个类型只注册一次，开销很小
2. **引用计数开销**: 原子操作，在多线程环境下有一定开销
3. **虚函数开销**: 基类使用虚函数，有虚函数调用开销
4. **内存开销**: 每个对象额外存储类型信息和引用计数

## 扩展功能

可以扩展类型系统支持：
- 类型转换检查
- 序列化/反序列化
- 反射调用
- 动态属性访问 