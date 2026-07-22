/*! \file include/kxc/support/object.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include <functional>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <string>
#include <string_view>
namespace kxc{
using TypeIndex = uint32_t;
using AttrVisitor = std::function<void(const char* key, void* value)>;

// Legacy base type index, kept for backward compatibility.
// New code should use the automatic registration mechanism.
constexpr TypeIndex kKXC_OBJECT_TYPE = 0;

/*!
 * \brief 单个逻辑对象类型的不可变运行时元数据。
 *
 * type_key 在进程和构建之间保持稳定；runtime_index 仅用于当前进程内
 * 的快速分派，不能作为持久化格式的一部分。
 */
class TypeInfo {
public:
    TypeIndex runtime_index() const noexcept { return runtime_index_; }
    std::string_view type_key() const noexcept { return type_key_; }

private:
    friend class TypeRegistry;

    TypeInfo(TypeIndex runtime_index, std::string type_key);

    TypeIndex runtime_index_;
    std::string type_key_;
};

/*!
 * \brief 全局对象类型注册表。
 *
 * 每个 Object 派生类通过 KXC_OBJECT_DEFINE 注册 TypeInfo。稳定 type key
 * 可用于外部协议；runtime index 只用于当前进程内的快速分派。
 */
class TypeRegistry {
public:
    static const TypeInfo& Register(const std::string& type_key);
    static const TypeInfo* FindByKey(std::string_view type_key);
    static const TypeInfo* FindByIndex(TypeIndex runtime_index);
};

/*!
 * \brief 在 Object 派生类中声明 TypeInfo 和兼容的 type id 接口。
 */
#define KXC_OBJECT_DECLARE \
    static const kxc::TypeInfo& _type_info; \
    static const uint32_t _type_index; \
    const kxc::TypeInfo& GetTypeInfo() const noexcept override { return _type_info; } \
    uint32_t GetTypeId() const noexcept override { return GetTypeInfo().runtime_index(); }

/*!
 * \brief 使用默认短 key 定义 TypeInfo；有重名风险的类型应使用显式 key。
 */
/*!
 * \brief 所有 IR 节点和 runtime 对象的引用计数基类。
 *
 * Object 本身不直接暴露给上层长期持有；上层通常通过 ObjectRef 派生句柄
 * 管理生命周期。new 会优先使用当前线程的 Arena，便于批量构造 IR 节点。
 * ArenaState 由每个 Arena 对象保活，因此 ObjectRef 可以安全离开 Arena scope。
 */
class Object{
public:
    virtual void VisitAttrs(AttrVisitor& visitor) {}
    virtual ~Object();

    static void* operator new(size_t size);
    static void* operator new(size_t size, std::align_val_t alignment);
    static void operator delete(void* ptr) noexcept;
    static void operator delete(void* ptr, size_t) noexcept;
    static void operator delete(void* ptr, std::align_val_t) noexcept;
    static void operator delete(void* ptr, size_t, std::align_val_t) noexcept;
    /*! \brief 增加引用计数。 */
	void IncRef() const;

    /*!
     * \brief 减少引用计数；归零时释放对象。
     *
     * release 发布当前引用持有者对节点的写入，最后一次 decrement 的 acquire
     * 与此前的 release sequence 配对，保证析构线程观察到全部已释放引用的写入。
     */
    void DecRef() const;
    Object();
    // 允许拷贝构造，但新对象的引用计数初始化
    Object(const Object&);
    // 允许赋值，但不改变引用计数
    Object& operator=(const Object&);

    // Ensure Object has a static type index as well
    static const TypeInfo& _type_info;
    static const uint32_t _type_index;
    virtual const TypeInfo& GetTypeInfo() const noexcept { return _type_info; }
    virtual TypeIndex GetTypeId() const noexcept { return GetTypeInfo().runtime_index(); }
    std::string_view GetTypeKey() const noexcept { return GetTypeInfo().type_key(); }
private:
    static void* AllocateStorage(size_t size, size_t requested_alignment);
    static void ReleaseStorage(void* ptr) noexcept;

    mutable std::atomic<TypeIndex> _refCount;
};

/*!
 * \brief Object 的引用类型基类。
 *
 * ObjectRef 负责引用计数、拷贝/移动语义和动态类型查询。Relay、TIR、
 * TE、runtime 中的用户可见 handle 大多继承自该类。
 */
class ObjectRef {
public:
    ObjectRef();
    virtual ~ObjectRef();
    explicit ObjectRef(const Object* obj);
    ObjectRef(ObjectRef&& other) noexcept;
    ObjectRef(const ObjectRef& other);
    ObjectRef& operator=(ObjectRef&& other) noexcept;
    ObjectRef& operator=(const ObjectRef& other);
    const Object* get() const { return object_; }
    const Object* operator->() const { return object_; }
    explicit operator bool() const { return object_ != nullptr; }
    bool defined() const { return object_ != nullptr; }
    /*! \brief 将底层 Object 安全转换为指定节点类型，转换失败返回 nullptr。 */
    template<typename T>
    const T* As() const {
        // dynamic_cast 用于安全地向下转型
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
    void SetData(const Object* obj);
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

// Template nodes use the same metadata contract, but define their static members
// with an explicit template parameter list.
#define KXC_OBJECT_DECLARE_TEMPLATE_NODE \
    static const kxc::TypeInfo& _type_info; \
    static const uint32_t _type_index; \
    const kxc::TypeInfo& GetTypeInfo() const noexcept override { return _type_info; } \
    uint32_t GetTypeId() const noexcept override { return GetTypeInfo().runtime_index(); }

} // namespace kxc
