/*! \file include/base/object.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include <functional>
#include <atomic>
#include <utility>
#include <mutex>
#include <unordered_map>
#include <string>
#include "base/arena.h"
namespace kxc{
extern thread_local Arena* current_arena;
using TypeIndex = uint32_t;
using AttrVisitor = std::function<void(const char* key, void* value)>;

// Legacy base type index, kept for backward compatibility.
// New code should use the automatic registration mechanism.
constexpr TypeIndex kKXC_OBJECT_TYPE = 0;

/*!
 * \brief 全局对象类型注册表。
 *
 * 每个 Object 派生类通过 KXC_OBJECT_DEFINE 注册一个运行时 type id。
 * 该 id 主要用于 IR 节点识别、序列化和执行计划解释。
 */
class TypeRegistry {
public:
    static uint32_t Register(const std::string& name) {
        static std::mutex mutex;
        static std::unordered_map<std::string, uint32_t> type_map;
        // Start from 1000 to avoid collision with legacy manual IDs (usually < 1000)
        static uint32_t next_index = 1000; 
        
        std::lock_guard<std::mutex> lock(mutex);
        if (type_map.find(name) == type_map.end()) {
            if (name == "Object") {
                type_map[name] = kKXC_OBJECT_TYPE;
            } else {
                type_map[name] = next_index++;
            }
        }
        return type_map[name];
    }
};

/*!
 * \brief 在 Object 派生类中声明静态 type id 和 GetTypeId 覆写。
 */
#define KXC_OBJECT_DECLARE \
    static const uint32_t _type_index; \
    const uint32_t GetTypeId() const override { return _type_index; }

/*!
 * \brief 在类定义可见处定义静态 type id。
 */
#define KXC_OBJECT_DEFINE(TypeName) \
    inline const uint32_t TypeName::_type_index = \
        kxc::TypeRegistry::Register(#TypeName);

/*!
 * \brief 所有 IR 节点和 runtime 对象的引用计数基类。
 *
 * Object 本身不直接暴露给上层长期持有；上层通常通过 ObjectRef 派生句柄
 * 管理生命周期。new 会优先使用当前线程的 Arena，便于批量构造 IR 节点。
 */
class Object{
public:
    virtual void VisitAttrs(AttrVisitor& visitor) {}
    virtual ~Object() = default;
    static void* operator new(size_t size){
        if (current_arena) {
            // Prefer arena allocation; silently fall back to global new on exhaustion.
            if (void* p = current_arena->Allocate(size, alignof(void*))) {
                return p;
            }
        }
        return ::operator new(size);
    }
    static void operator delete(void* ptr, size_t size) {
        // 因为Arena是批量释放，所以单个对象的delete是一个空操作(no-op)
        // 我们什么都不用做，这正是Arena高效的原因之一�?
        if (current_arena) {
            // std::cout << "Arena handles deallocation." << std::endl;
        } else {
            ::operator delete(ptr);
        }
    }
    /*! \brief 增加引用计数。 */
	void IncRef() const {
        _refCount.fetch_add(1, std::memory_order_relaxed);
    }

    /*! \brief 减少引用计数；归零时释放对象。 */
    void DecRef() const {
        if(_refCount.fetch_sub(1, std::memory_order_relaxed)==1){
            std::atomic_thread_fence(std::memory_order_acquire);
            delete this;
        }
    }
    Object() : _refCount(0) {}
    // 允许拷贝构造，但新对象的引用计数初始化�?0
    Object(const Object&) : _refCount(0) {}
    // 允许赋值，但不改变引用计数
    Object& operator=(const Object&) { return *this; }
    
    // Ensure Object has a static type index as well
    static const uint32_t _type_index;
    virtual const TypeIndex GetTypeId() const { return _type_index; }
private:
    mutable std::atomic<TypeIndex> _refCount;
};

// Define Object's type index (should be 0)
KXC_OBJECT_DEFINE(Object)

/*!
 * \brief Object 的引用类型基类。
 *
 * ObjectRef 负责引用计数、拷贝/移动语义和动态类型查询。Relay、TIR、
 * TE、runtime 中的用户可见 handle 大多继承自该类。
 */
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
        if (this != &other) { // 防止自我赋�?
            if (object_) object_->DecRef(); // 减少当前对象所持有的引�?
            object_ = other.object_;        // 复制指针
            if (object_) object_->IncRef(); // 增加新对象所持有的引�?
        }
        return *this;
    }
    const Object* get() const { return object_; }
    const Object* operator->() const { return object_; }
    explicit operator bool() const { return object_ != nullptr; }
    bool defined() const { return object_ != nullptr; }
    /*! \brief 将底层 Object 安全转换为指定节点类型，转换失败返回 nullptr。 */
    template<typename T>
    const T* As() const {
        // dynamic_cast 用于安全地向下转�?
        return dynamic_cast<const T*>(object_);
    }
    
    bool operator==(const ObjectRef& other) const {
        return object_ == other.object_;
    }
    bool operator!=(const ObjectRef& other) const {
        return object_ != other.object_;
    }
    
protected:
    const Object* object_;
    
    // Helper to assign object and increment reference count
    // Used by subclasses to safely adopt new objects
    void SetData(const Object* obj) {
        if (object_) object_->DecRef();
        object_ = obj;
        if (object_) object_->IncRef();
    }
};

/*!
 * \brief 创建 Object 派生对象并返回 ObjectRef。
 * \tparam T Object 派生节点类型。
 * \param args 转发给 T 构造函数的参数。
 */
template<typename T, typename... Args>
inline ObjectRef make_object(Args&&... args) {
    return ObjectRef(new T(std::forward<Args>(args)...));
}

// Special macro for template classes where standard registration might be tricky
// or we just reuse the same logic but need template syntax support in user code.
// Actually KXC_OBJECT_DECLARE works fine for templates if _type_index is static member.
// But defining it for templates requires template<...> syntax.
#define KXC_OBJECT_DECLARE_TEMPLATE_NODE \
    static const uint32_t _type_index; \
    const uint32_t GetTypeId() const override { return _type_index; }

// For template definitions, we can't easily auto-register all instantiations with unique names
// unless we use RTTI typeid(T).name() or similar.
// For simplicity in this project, let's assume specific instantiations or use a generic "Array" type index?
// But Array<Int> and Array<Float> are different C++ types.
// If we use KXC_OBJECT_DECLARE, we need to define _type_index for each T.
// Alternative: ArrayNode<T> returns TypeRegistry::Register("Array") shared?
// No, GetTypeId should ideally distinguish types if we do strict checking.
// But ObjectRef::As<T> uses dynamic_cast, so _type_index is mostly for serialization/reflection.
// Let's implement a simple version where all Arrays share same ID or we use RTTI-based name.

template<typename T>
struct TypeNameTraits {
    static std::string Get() { return "Object"; }
};

// Helper to register template types on demand?
// For now, let's just make KXC_OBJECT_DECLARE_TEMPLATE work by specializing or just
// returning a hash of typeid(T).name() if possible, or just standard Register.

} // namespace base


