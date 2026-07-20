/*! \file include/base/object.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include <functional>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <string_view>
#include "base/arena.h"
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

    TypeInfo(TypeIndex runtime_index, std::string type_key)
        : runtime_index_(runtime_index), type_key_(std::move(type_key)) {}

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
    static const TypeInfo& Register(const std::string& type_key) {
        State& state = GlobalState();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.by_key.find(type_key);
        if (it != state.by_key.end()) {
            return *it->second;
        }

        const TypeIndex runtime_index =
            type_key == "Object" ? kKXC_OBJECT_TYPE : state.next_index;
        if (state.by_index.count(runtime_index) != 0) {
            throw std::logic_error("duplicate runtime type index");
        }

        auto info = std::unique_ptr<TypeInfo>(new TypeInfo(runtime_index, type_key));
        auto key_result = state.by_key.emplace(type_key, std::move(info));
        try {
            auto index_result =
                state.by_index.emplace(runtime_index, key_result.first->second.get());
            if (!index_result.second) {
                throw std::logic_error("duplicate runtime type index");
            }
        } catch (...) {
            state.by_key.erase(key_result.first);
            throw;
        }
        if (type_key != "Object") {
            ++state.next_index;
        }
        return *key_result.first->second;
    }

    static const TypeInfo* FindByKey(std::string_view type_key) {
        State& state = GlobalState();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.by_key.find(std::string(type_key));
        return it == state.by_key.end() ? nullptr : it->second.get();
    }

    static const TypeInfo* FindByIndex(TypeIndex runtime_index) {
        State& state = GlobalState();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto it = state.by_index.find(runtime_index);
        return it == state.by_index.end() ? nullptr : it->second;
    }

private:
    struct State {
        std::mutex mutex;
        std::unordered_map<std::string, std::unique_ptr<TypeInfo>> by_key;
        std::unordered_map<TypeIndex, const TypeInfo*> by_index;
        TypeIndex next_index{1000};
    };

    static State& GlobalState() {
        static State state;
        return state;
    }
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
#define KXC_OBJECT_DEFINE_WITH_KEY(TypeName, TypeKey) \
    inline const kxc::TypeInfo& TypeName::_type_info = \
        kxc::TypeRegistry::Register(TypeKey); \
    inline const uint32_t TypeName::_type_index = \
        TypeName::_type_info.runtime_index();

#define KXC_OBJECT_DEFINE(TypeName) \
    KXC_OBJECT_DEFINE_WITH_KEY(TypeName, #TypeName)

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
    virtual ~Object() = default;

    static void* operator new(size_t size) {
        return AllocateStorage(size, alignof(std::max_align_t));
    }

    static void* operator new(size_t size, std::align_val_t alignment) {
        return AllocateStorage(size, static_cast<size_t>(alignment));
    }

    static void operator delete(void* ptr) noexcept {
        ReleaseStorage(ptr);
    }

    static void operator delete(void* ptr, size_t) noexcept {
        ReleaseStorage(ptr);
    }

    static void operator delete(void* ptr, std::align_val_t) noexcept {
        ReleaseStorage(ptr);
    }

    static void operator delete(
        void* ptr, size_t, std::align_val_t) noexcept {
        ReleaseStorage(ptr);
    }
    /*! \brief 增加引用计数。 */
	void IncRef() const {
        _refCount.fetch_add(1, std::memory_order_relaxed);
    }

    /*!
     * \brief 减少引用计数；归零时释放对象。
     *
     * release 发布当前引用持有者对节点的写入，最后一次 decrement 的 acquire
     * 与此前的 release sequence 配对，保证析构线程观察到全部已释放引用的写入。
     */
    void DecRef() const {
        if (_refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
    Object() : _refCount(0) {}
    // 允许拷贝构造，但新对象的引用计数初始化
    Object(const Object&) : _refCount(0) {}
    // 允许赋值，但不改变引用计数
    Object& operator=(const Object&) { return *this; }
    
    // Ensure Object has a static type index as well
    static const TypeInfo& _type_info;
    static const uint32_t _type_index;
    virtual const TypeInfo& GetTypeInfo() const noexcept { return _type_info; }
    virtual TypeIndex GetTypeId() const noexcept { return GetTypeInfo().runtime_index(); }
    std::string_view GetTypeKey() const noexcept { return GetTypeInfo().type_key(); }
private:
    enum class AllocationKind : uint8_t {
        kHeap,
        kArena,
    };

    // 紧邻 Object 之前保存真实分配来源，删除时不依赖当前线程的 TLS 状态。
    struct alignas(std::max_align_t) AllocationHeader {
        uint64_t magic;
        void* heap_base;
        detail::ArenaState* arena_state;
        AllocationKind kind;
    };

    static constexpr uint64_t kAllocationMagic = UINT64_C(0x4b58434f424a4543);

    static void* AllocateStorage(size_t size, size_t requested_alignment) {
        const size_t alignment =
            std::max(requested_alignment, alignof(AllocationHeader));
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::bad_alloc();
        }

        const size_t overhead = sizeof(AllocationHeader) + alignment - 1;
        if (size > std::numeric_limits<size_t>::max() - overhead) {
            throw std::bad_alloc();
        }
        const size_t total = size + overhead;

        void* base = nullptr;
        detail::ArenaState* arena_state = nullptr;
        AllocationKind kind = AllocationKind::kHeap;
        if (current_arena) {
            base = current_arena->Allocate(total, alignof(AllocationHeader));
            if (base) {
                // 每个成功的 Arena 对象各持有一次 state，允许对象跨 scope 逃逸。
                arena_state = current_arena->RetainState();
                kind = AllocationKind::kArena;
            }
        }
        if (!base) {
            // Arena 耗尽时回退 heap；header 会确保之后仍调用全局 delete。
            base = ::operator new(total);
        }

        // 预留 padding，使 header 紧挨对齐后的 Object，同时保留原始 heap base。
        const uintptr_t object_address =
            (reinterpret_cast<uintptr_t>(base) + sizeof(AllocationHeader) +
             alignment - 1) &
            ~(alignment - 1);
        auto* header = reinterpret_cast<AllocationHeader*>(
            object_address - sizeof(AllocationHeader));
        ::new (header) AllocationHeader{
            kAllocationMagic,
            kind == AllocationKind::kHeap ? base : nullptr,
            arena_state,
            kind,
        };
        return reinterpret_cast<void*>(object_address);
    }

    static void ReleaseStorage(void* ptr) noexcept {
        if (!ptr) return;

        auto* header = reinterpret_cast<AllocationHeader*>(
            static_cast<std::byte*>(ptr) - sizeof(AllocationHeader));
        if (header->magic != kAllocationMagic) {
            // 通常表示内存破坏，或混用了没有分配头的旧二进制对象。
            std::abort();
        }

        // 最后一次 ArenaState::Release 可能立即释放 header 所在 block，
        // 所以必须在 Release 前把后续需要的字段复制到栈上。
        const AllocationKind kind = header->kind;
        void* heap_base = header->heap_base;
        detail::ArenaState* arena_state = header->arena_state;
        header->~AllocationHeader();

        if (kind == AllocationKind::kHeap) {
            ::operator delete(heap_base);
        } else {
            arena_state->Release();
        }
    }

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
            const Object* old_object = object_;
            object_ = other.object_;
            other.object_ = nullptr;
            if (old_object) old_object->DecRef();
        }
        return *this;
    }
    ObjectRef& operator=(const ObjectRef& other) {
        if (this != &other) { // 防止自我赋值
            const Object* new_object = other.object_;
            if (new_object) new_object->IncRef();
            const Object* old_object = object_;
            object_ = new_object;
            if (old_object) old_object->DecRef();
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
    void SetData(const Object* obj) {
        if (object_ == obj) return;
        if (obj) obj->IncRef();
        const Object* old_object = object_;
        object_ = obj;
        if (old_object) old_object->DecRef();
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

// Template nodes use the same metadata contract, but define their static members
// with an explicit template parameter list.
#define KXC_OBJECT_DECLARE_TEMPLATE_NODE \
    static const kxc::TypeInfo& _type_info; \
    static const uint32_t _type_index; \
    const kxc::TypeInfo& GetTypeInfo() const noexcept override { return _type_info; } \
    uint32_t GetTypeId() const noexcept override { return GetTypeInfo().runtime_index(); }

} // namespace kxc


