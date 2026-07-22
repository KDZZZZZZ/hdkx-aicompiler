/*! \file include/kxc/support/container.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once
#include "kxc/support/object.h"
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

    ArrayNode();
    ArrayNode(std::vector<T> d);

    KXC_OBJECT_DECLARE_TEMPLATE_NODE
};

template <typename T>
class Array : public ObjectRef {
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    Array();
    Array(const ObjectRef& n);
    Array(std::vector<T> data);
    Array(std::initializer_list<T> init);

    const T& operator[](size_t i) const;
    T& operator[](size_t i);
    size_t size() const;
    bool empty() const;
    void push_back(const T& val);
    void push_back(T&& val);
    void insert(typename std::vector<T>::const_iterator pos, const T& val);
    void erase(typename std::vector<T>::const_iterator pos);
    const_iterator begin() const;
    const_iterator end() const;
    iterator begin();
    iterator end();
};


// ============================================================================
// Map (Reference Counted std::unordered_map)
// ============================================================================

template <typename K, typename V>
class MapNode : public Object {
public:
    std::unordered_map<K, V> data;

    MapNode();
    MapNode(std::unordered_map<K, V> d);

    KXC_OBJECT_DECLARE_TEMPLATE_NODE
};

template <typename K, typename V>
class Map : public ObjectRef {
public:
    using iterator = typename std::unordered_map<K, V>::iterator;
    using const_iterator = typename std::unordered_map<K, V>::const_iterator;

    Map();
    Map(const ObjectRef& n);
    Map(std::unordered_map<K, V> data);
    Map(std::initializer_list<std::pair<const K, V>> init);

    size_t size() const;
    size_t count(const K& key) const;
    const V& at(const K& key) const;
    const V& operator[](const K& key) const;
    void Set(const K& key, const V& val);
    const_iterator begin() const;
    const_iterator end() const;
    iterator begin();
    iterator end();
};

// ============================================================================
// String (Object-based std::string)
// ============================================================================

class StringObj : public Object {
public:
    std::string data;

    StringObj();
    explicit StringObj(std::string s);

    KXC_OBJECT_DECLARE
};

class String : public ObjectRef {
public:
    String();
    String(const char* s);
    String(std::string s);
    String(const ObjectRef& n);

    const StringObj* operator->() const;
    operator std::string() const;
    bool operator==(const String& other) const;
    bool operator==(const std::string& other) const;
    bool operator==(const char* other) const;
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

#include "kxc/support/detail/array_inl.h"
#include "kxc/support/detail/map_inl.h"
