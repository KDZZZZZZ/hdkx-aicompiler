#pragma once
#include <functional>
#include <atomic>
#include <utility>
#include <iostream>
#include "base/arena.h"
namespace kxc{
extern thread_local Arena* current_arena;
using TypeIndex = uint32_t;
using AttrVisitor = std::function<void(const char* key, void* value)>;
constexpr TypeIndex kKXC_OBJECT_TYPE = 0;
class Object{
public:
    virtual void VisitAttrs(AttrVisitor& visitor) {}
    virtual ~Object() = default;
    static void* operator new(size_t size){
        if (current_arena) {
            std::cout << "Allocating " << size << " bytes from Arena." << std::endl;
            // 默认对齐到void*大小
            return current_arena->Allocate(size, alignof(void*));
        } else {
            std::cout << "Warning: No Arena active. Falling back to global new." << std::endl;
            return ::operator new(size); // 如果没有Arena，则回退到全局new
        }
    }
    static void operator delete(void* ptr, size_t size) {
        // 因为Arena是批量释放，所以单个对象的delete是一个空操作(no-op)
        // 我们什么都不用做，这正是Arena高效的原因之一！
        if (current_arena) {
            // std::cout << "Arena handles deallocation." << std::endl;
        } else {
            ::operator delete(ptr);
        }
    }
	void IncRef() const {
        _refCount.fetch_add(1, std::memory_order_relaxed);
    }
    void DecRef() const {
        if(_refCount.fetch_sub(1, std::memory_order_relaxed)==1){
            std::atomic_thread_fence(std::memory_order_acquire);
            delete this;
        }
    }
    Object() : _refCount(0) {}
    // 允许拷贝构造，但新对象的引用计数初始化为 0
    Object(const Object&) : _refCount(0) {}
    // 允许赋值，但不改变引用计数
    Object& operator=(const Object&) { return *this; }
    
    virtual const TypeIndex GetTypeId() const { return kKXC_OBJECT_TYPE; }
private:
    mutable std::atomic<TypeIndex> _refCount;
};
class ObjectRef {
public:
    ObjectRef() : object_(nullptr) {}
    virtual ~ObjectRef(){
        if(object_) object_->DecRef();
    }
    explicit ObjectRef(const Object* obj) : object_(obj){
        if(object_) object_->IncRef();
    }
    ObjectRef(ObjectRef&& other) noexcept : object_(other.object_) {
        other.object_ = nullptr;
    }
    ObjectRef(const ObjectRef& other) : object_(other.object_) {
        if (object_) object_->IncRef();
    }
    ObjectRef& operator=(ObjectRef&& other) noexcept {
        if (this != &other) {
            if (object_) object_->DecRef();
            object_ = other.object_;
            other.object_ = nullptr;
        }
        return *this;
    }
    ObjectRef& operator=(const ObjectRef& other) {
        if (this != &other) { // 防止自我赋值
            if (object_) object_->DecRef(); // 减少当前对象所持有的引用
            object_ = other.object_;        // 复制指针
            if (object_) object_->IncRef(); // 增加新对象所持有的引用
        }
        return *this;
    }
    const Object* get() const { return object_; }
    const Object* operator->() const { return object_; }
    explicit operator bool() const { return object_ != nullptr; }
    bool defined() const { return object_ != nullptr; }
    template<typename T>
    const T* As() const {
        // dynamic_cast 用于安全地向下转型
        return dynamic_cast<const T*>(object_);
    }
protected:
    const Object* object_;
};

} // namespace base

