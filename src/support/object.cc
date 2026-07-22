/*! \file src/support/object.cc
 * \brief Implements Object allocation and ObjectRef lifetime.
 */

#include "kxc/support/object.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "kxc/support/arena.h"
#include "kxc/support/object_registration.h"
#include "internal/arena_state.h"

namespace kxc {

namespace {

enum class AllocationKind : uint8_t {
    kHeap,
    kArena,
};

struct alignas(std::max_align_t) AllocationHeader {
    uint64_t magic;
    void* heap_base;
    detail::ArenaState* arena_state;
    AllocationKind kind;
};

constexpr uint64_t kAllocationMagic = UINT64_C(0x4b58434f424a4543);

}  // namespace

Object::Object() : _refCount(0) {}

Object::Object(const Object&) : _refCount(0) {}

Object::~Object() = default;

Object& Object::operator=(const Object&) {
    return *this;
}

void* Object::operator new(size_t size) {
    return AllocateStorage(size, alignof(std::max_align_t));
}

void* Object::operator new(size_t size, std::align_val_t alignment) {
    return AllocateStorage(size, static_cast<size_t>(alignment));
}

void Object::operator delete(void* ptr) noexcept {
    ReleaseStorage(ptr);
}

void Object::operator delete(void* ptr, size_t) noexcept {
    ReleaseStorage(ptr);
}

void Object::operator delete(void* ptr, std::align_val_t) noexcept {
    ReleaseStorage(ptr);
}

void Object::operator delete(void* ptr, size_t, std::align_val_t) noexcept {
    ReleaseStorage(ptr);
}

void Object::IncRef() const {
    _refCount.fetch_add(1, std::memory_order_relaxed);
}

void Object::DecRef() const {
    if (_refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
}

void* Object::AllocateStorage(size_t size, size_t requested_alignment) {
    const size_t alignment = std::max(requested_alignment, alignof(AllocationHeader));
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
            arena_state = current_arena->RetainState();
            kind = AllocationKind::kArena;
        }
    }
    if (!base) base = ::operator new(total);

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

void Object::ReleaseStorage(void* ptr) noexcept {
    if (!ptr) return;

    auto* header = reinterpret_cast<AllocationHeader*>(
        static_cast<std::byte*>(ptr) - sizeof(AllocationHeader));
    if (header->magic != kAllocationMagic) std::abort();

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

KXC_OBJECT_DEFINE(Object)

ObjectRef::ObjectRef() : object_(nullptr) {}

ObjectRef::~ObjectRef() {
    if (object_) object_->DecRef();
}

ObjectRef::ObjectRef(const Object* obj) : object_(obj) {
    if (object_) object_->IncRef();
}

ObjectRef::ObjectRef(ObjectRef&& other) noexcept : object_(other.object_) {
    other.object_ = nullptr;
}

ObjectRef::ObjectRef(const ObjectRef& other) : object_(other.object_) {
    if (object_) object_->IncRef();
}

ObjectRef& ObjectRef::operator=(ObjectRef&& other) noexcept {
    if (this != &other) {
        const Object* old_object = object_;
        object_ = other.object_;
        other.object_ = nullptr;
        if (old_object) old_object->DecRef();
    }
    return *this;
}

ObjectRef& ObjectRef::operator=(const ObjectRef& other) {
    if (this != &other) {
        const Object* new_object = other.object_;
        if (new_object) new_object->IncRef();
        const Object* old_object = object_;
        object_ = new_object;
        if (old_object) old_object->DecRef();
    }
    return *this;
}

void ObjectRef::SetData(const Object* obj) {
    if (object_ == obj) return;
    if (obj) obj->IncRef();
    const Object* old_object = object_;
    object_ = obj;
    if (old_object) old_object->DecRef();
}

}  // namespace kxc
