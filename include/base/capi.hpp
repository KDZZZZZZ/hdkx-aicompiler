#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <cstdio>

namespace kxcomp {
namespace base {

// 前向声明
class object;
template<typename T> class objectPtr;
template<typename T> struct TypeRegistration;

// 常量定义
constexpr int32_t kDynamicIndex = -1;
constexpr int32_t kInvalidIndex = -2;

// 文件操作类
class file_op {
public:
    static std::string read_file(const std::string& path);
    static void write_file(const std::string& path, const std::string& content);
    static void delete_file(const std::string& path);
};

// 类型信息结构
struct TypeInfo {
    std::string type_key;
    std::function<object*()> constructor;
    std::function<void(object*)> deleter;
    int32_t parent_index = kInvalidIndex;
    uint32_t type_key_hash = 0;
    
    TypeInfo(const std::string& key) : type_key(key) {
        type_key_hash = std::hash<std::string>{}(key);
    }
};

// 类型注册表
class TypeRegistry {
public:
    static TypeRegistry* Global() {
        static TypeRegistry instance;
        return &instance;
    }
    
    int32_t RegisterType(const TypeInfo& type_info) {
        int32_t index = static_cast<int32_t>(type_info_list.size());
        type_info_list.push_back(type_info);
        key_to_index[type_info.type_key] = index;
        return index;
    }
    
    int32_t GetTypeIndex(const std::string& key) {
        auto it = key_to_index.find(key);
        if (it == key_to_index.end()) {
            return kInvalidIndex;
        }
        return it->second;
    }
    
    TypeInfo* GetTypeInfo(int32_t index) {
        if (index >= 0 && index < static_cast<int32_t>(type_info_list.size())) {
            return &type_info_list[index];
        }
        return nullptr;
    }
    
    size_t GetTypeCount() const {
        return type_info_list.size();
    }
    
private:
    std::unordered_map<std::string, int32_t> key_to_index;
    std::vector<TypeInfo> type_info_list;
};

// 对象头部信息
struct ObjectHeader {
    std::atomic<int32_t> ref_counter{0};
    std::function<void(void*)> deleter;
    int32_t type_index = kInvalidIndex;
};

// 基础对象类
class object {
protected:
    ObjectHeader header;
    
public:
    object() : header() {
        header.ref_counter = 0;
        header.deleter = nullptr;
        header.type_index = kInvalidIndex;
    }
    
    virtual ~object() = default;
    
    int32_t GetTypeIndex() const {
        return header.type_index;
    }
    
    std::string GetTypeInfo() const {
        if (TypeInfo* info = TypeRegistry::Global()->GetTypeInfo(header.type_index)) {
            return info->type_key;
        }
        return "Unknown";
    }
    
    void IncRef() {
        header.ref_counter.fetch_add(1, std::memory_order_relaxed);
    }

    void DecRef() {
        if (header.ref_counter.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (header.deleter) {
                header.deleter(this);
            }
        }
    }
    
    int32_t GetRefCount() const {
        return header.ref_counter.load(std::memory_order_relaxed);
    }
    
    void SetDeleter(std::function<void(void*)> deleter) {
        header.deleter = deleter;
    }
    
    void SetTypeIndex(int32_t index) {
        header.type_index = index;
    }
};

// 智能指针模板
template<typename T>
class objectPtr {
public:
    objectPtr() : data_(nullptr) {}
    
    explicit objectPtr(T* data) : data_(data) {
        if (data_) data_->IncRef();
    }
    
    objectPtr(const objectPtr& other) : data_(other.data_) {
        if (data_) data_->IncRef();
    }
    
    objectPtr(objectPtr&& other) noexcept : data_(other.data_) {
        other.data_ = nullptr;
    }
    
    ~objectPtr() {
        if (data_) data_->DecRef();
    }
    
    objectPtr& operator=(const objectPtr& other) {
        if (this != &other) {
            if (data_) data_->DecRef();
            data_ = other.data_;
            if (data_) data_->IncRef();
        }
        return *this;
    }
    
    objectPtr& operator=(objectPtr&& other) noexcept {
        if (this != &other) {
            if (data_) data_->DecRef();
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    T* operator->() const { return data_; }
    T& operator*() const { return *data_; }
    explicit operator bool() const { return data_ != nullptr; }
    T* get() const { return data_; }
    
    T* release() {
        T* ptr = data_;
        data_ = nullptr;
        return ptr;
    }

    void reset(T* ptr = nullptr) {
        if (data_) data_->DecRef();
        data_ = ptr;
        if (data_) data_->IncRef();
    }
    
private:
    T* data_;
};

// 类型注册模板
template<typename T>
struct TypeRegistration {
    static int32_t Register() {
        static TypeInfo info(T::_type_key);
        static int32_t index = kInvalidIndex;
        if (index == kInvalidIndex) {
            info.constructor = []() -> object* { return new T(); };
            info.deleter = [](object* obj) { delete static_cast<T*>(obj); };
            index = TypeRegistry::Global()->RegisterType(info);
            T::_type_static_index = index;
        }
        return index;
    }
};

// 对象创建函数
template<typename T, typename... Args>
objectPtr<T> make_object(Args&&... args) {
    T* ptr = new T(std::forward<Args>(args)...);
    const int32_t type_index = T::RuntimeTypeIndex();
    ptr->SetTypeIndex(type_index);
    if (TypeInfo* info = TypeRegistry::Global()->GetTypeInfo(type_index)) {
        ptr->SetDeleter([](void* obj) {
            delete static_cast<T*>(obj);
        });
    }
    return objectPtr<T>(ptr);
}

// 类型注册宏
#define SIMPLE_REGISTER_TYPE(T) \
    template <> struct TypeRegistration<T> { \
        static int32_t Register() { \
            static TypeInfo info(T::_type_key); \
            static int32_t index = kInvalidIndex; \
            if (index == kInvalidIndex) { \
                info.constructor = []() -> object* { return new T(); }; \
                info.deleter = [](object* obj) { delete static_cast<T*>(obj); }; \
                index = TypeRegistry::Global()->RegisterType(info); \
                T::_type_static_index = index; \
            } \
            return index; \
        } \
    };

// 类型声明宏
#define SIMPLE_DECLARE_TYPE(ClassName, ParentType) \
    public: \
        static constexpr const char* _type_key = #ClassName; \
        static int32_t _type_static_index; \
        static int32_t RuntimeTypeIndex() { \
            if (_type_static_index == kDynamicIndex) { \
                return TypeRegistration<ClassName>::Register(); \
            } \
            return _type_static_index; \
        } \
        SIMPLE_REGISTER_TYPE(ClassName)

} // namespace base
} // namespace kxcomp