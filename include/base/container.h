#pragma once
#include "object.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>

namespace kxc {

// ============================================================================
// Array (Reference Counted std::vector)
// ============================================================================

template <typename T>
class ArrayNode : public Object {
public:
    std::vector<T> data;

    ArrayNode() = default;
    ArrayNode(std::vector<T> d) : data(std::move(d)) {}

    static const uint32_t _type_index;
    const uint32_t GetTypeId() const override { return _type_index; }
};

// Initialize static member for each template instantiation
// This will create a unique type index for each T (e.g., Array<int>, Array<float>)
// Note: This relies on TypeRegistry returning unique IDs for "Array" calls. 
// If TypeRegistry implementation simply increments a counter for new calls regardless of name collision (or if we append typeid name),
// it works. The current implementation of TypeRegistry in object.h uses a map based on name.
// So Register("Array") will return the SAME index for all Arrays if we pass "Array".
// To fix this, we should ideally include type info in the name, e.g. "Array<" + typeid(T).name() + ">".
// BUT for this simplified implementation, sharing TypeId "Array" for all arrays might be acceptable 
// if we don't strictly differentiate Array<int> vs Array<float> at runtime type checking level via GetTypeId.
// However, the user hint says: "Compiler will generate a variable...".
// If we want them to have DIFFERENT IDs, we must pass DIFFERENT names to Register.
// If we pass "Array" to all, they get the SAME ID.
// Let's modify it to be generic "Array" for now as per "apply scenario: Array<IntImm>" usually treated as generic Array in dynamic type systems like TVM (which has ArrayNode generic).
template<typename T>
const uint32_t ArrayNode<T>::_type_index = kxc::TypeRegistry::Register("Array");


template <typename T>
class Array : public ObjectRef {
public:
    using value_type = T;

    // Constructors
    Array() : ObjectRef(new ArrayNode<T>()) {}
    
    Array(std::vector<T> data) {
        auto ptr = new ArrayNode<T>(std::move(data));
        SetData(ptr);
    }
    
    Array(std::initializer_list<T> init) {
        auto ptr = new ArrayNode<T>(std::vector<T>(init));
        SetData(ptr);
    }
    
    // Copy/Move constructors handled by ObjectRef
    
    // Accessors
    const T& operator[](size_t i) const {
        return static_cast<const ArrayNode<T>*>(get())->data[i];
    }
    
    T& operator[](size_t i) {
        // COW could be implemented here: if ref_count > 1, clone.
        // For now, mutable access assumes we know what we are doing or it's shared.
        return const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data[i];
    }
    
    size_t size() const {
        return static_cast<const ArrayNode<T>*>(get())->data.size();
    }
    
    bool empty() const {
        return static_cast<const ArrayNode<T>*>(get())->data.empty();
    }
    
    void push_back(const T& val) {
        const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.push_back(val);
    }
    
    void push_back(T&& val) {
        const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.push_back(std::move(val));
    }

    void insert(typename std::vector<T>::const_iterator pos, const T& val) {
        const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.insert(pos, val);
    }
    
    void erase(typename std::vector<T>::const_iterator pos) {
        const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.erase(pos);
    }
    
    // Iterators
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    
    const_iterator begin() const { return static_cast<const ArrayNode<T>*>(get())->data.begin(); }
    const_iterator end() const { return static_cast<const ArrayNode<T>*>(get())->data.end(); }
    
    iterator begin() { return const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.begin(); }
    iterator end() { return const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))->data.end(); }
};


// ============================================================================
// Map (Reference Counted std::unordered_map)
// ============================================================================

template <typename K, typename V>
class MapNode : public Object {
public:
    std::unordered_map<K, V> data;
    
    MapNode() = default;
    MapNode(std::unordered_map<K, V> d) : data(std::move(d)) {}

    static const uint32_t _type_index;
    const uint32_t GetTypeId() const override { return _type_index; }
};

template<typename K, typename V>
const uint32_t MapNode<K, V>::_type_index = kxc::TypeRegistry::Register("Map");

template <typename K, typename V>
class Map : public ObjectRef {
public:
    Map() : ObjectRef(new MapNode<K, V>()) {}
    
    Map(std::unordered_map<K, V> data) {
        SetData(new MapNode<K, V>(std::move(data)));
    }
    
    Map(std::initializer_list<std::pair<const K, V>> init) {
         SetData(new MapNode<K, V>(std::unordered_map<K, V>(init)));
    }
    
    size_t size() const {
         return static_cast<const MapNode<K, V>*>(get())->data.size();
    }
    
    size_t count(const K& key) const {
         return static_cast<const MapNode<K, V>*>(get())->data.count(key);
    }
    
    const V& at(const K& key) const {
         return static_cast<const MapNode<K, V>*>(get())->data.at(key);
    }
    
    const V& operator[](const K& key) const {
        return at(key);
    }
    
    // Mutable access via Set
    void Set(const K& key, const V& val) {
        const_cast<MapNode<K, V>*>(static_cast<const MapNode<K, V>*>(get()))->data[key] = val;
    }
    
    // Iterators
    using iterator = typename std::unordered_map<K, V>::iterator;
    using const_iterator = typename std::unordered_map<K, V>::const_iterator;
    
    const_iterator begin() const { return static_cast<const MapNode<K, V>*>(get())->data.begin(); }
    const_iterator end() const { return static_cast<const MapNode<K, V>*>(get())->data.end(); }
    
    iterator begin() { return const_cast<MapNode<K, V>*>(static_cast<const MapNode<K, V>*>(get()))->data.begin(); }
    iterator end() { return const_cast<MapNode<K, V>*>(static_cast<const MapNode<K, V>*>(get()))->data.end(); }
};

// ============================================================================
// String (Object-based std::string)
// ============================================================================

class StringObj : public Object {
public:
    std::string data;
    
    StringObj() = default;
    StringObj(std::string s) : data(std::move(s)) {}
    
    KXC_OBJECT_DECLARE
};
// Definition macro needs to be in .cc or we assume header-only via inline
// KXC_OBJECT_DEFINE(StringObj) -> This puts inline definition in header.
// But we need to make sure we don't redefine if included multiple times.
// KXC_OBJECT_DEFINE uses 'inline' so it is safe in header.

// To avoid circular dependency or redefinition issues if we put this in header,
// we'll put the define here.
inline const uint32_t StringObj::_type_index = kxc::TypeRegistry::Register("String");

class String : public ObjectRef {
public:
    String() : ObjectRef(new StringObj()) {}
    
    String(const char* s) : ObjectRef(new StringObj(s)) {}
    String(std::string s) : ObjectRef(new StringObj(std::move(s))) {}
    
    // Copy constructor from ObjectRef
    String(const ObjectRef& n) : ObjectRef(n) {}
    
    const StringObj* operator->() const {
        return static_cast<const StringObj*>(object_);
    }
    
    // Implicit conversion to std::string
    operator std::string() const {
        if (!defined()) return "";
        return operator->()->data;
    }
    
    // Comparison
    bool operator==(const String& other) const {
        if (object_ == other.object_) return true;
        if (!defined() || !other.defined()) return false;
        return operator->()->data == other->data;
    }
    
    bool operator==(const std::string& other) const {
        if (!defined()) return false;
        return operator->()->data == other;
    }
    
    bool operator==(const char* other) const {
        if (!defined()) return other == nullptr;
        return operator->()->data == other;
    }
};

// Enable hashing for String to be used in Map
} // namespace kxc

namespace std {
    template <>
    struct hash<kxc::String> {
        size_t operator()(const kxc::String& k) const {
            if (!k.defined()) return 0;
            return std::hash<std::string>()(static_cast<std::string>(k));
        }
    };
}
