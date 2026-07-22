/*! \file src/python/internal/type_manager.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include "kxc/support/object.h"
#include <unordered_map>
#include <string>
#include <functional>
#include <memory>
namespace kxc{
// 类型管理器，使用单例模式
class TypeManager {
public:
    // 获取单例实例
    static TypeManager* Get() {
        static TypeManager instance;
        return &instance;
    }

    // 类型注册函数
    // 接受一个类型键和一个用于创建该类型对象的工厂函数
    void Register(const std::string& type_key, std::function<Object*()> factory) {
        registry_[type_key] = factory;
    }

    // 根据类型键创建对象实例
    ObjectRef CreateObject(const std::string& type_key) {
        auto it = registry_.find(type_key);
        if (it == registry_.end()) {
            // 在实际框架中，这里应该抛出一个更详细的错误
            return ObjectRef(nullptr); 
        }
        return ObjectRef(it->second());
    }

private:
    TypeManager() = default;
    ~TypeManager() = default;
    TypeManager(const TypeManager&) = delete;
    TypeManager& operator=(const TypeManager&) = delete;

    // 存储类型键到工厂函数的映射
    std::unordered_map<std::string, std::function<Object*()>> registry_;
};

// 这是一个辅助结构体，利用其构造函数来执行注册逻辑
// 这是实现自动注册的关键技巧
struct TypeRegistrar {
    TypeRegistrar(const std::string& type_key, std::function<Object*()> factory) {
        TypeManager::Get()->Register(type_key, factory);
    }
};
}
