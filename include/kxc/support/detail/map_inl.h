/*! \file include/kxc/support/detail/map_inl.h
 * \brief Visible template implementation for Map.
 *
 * This file is included by container.h and is not a stable direct-include API.
 */

#pragma once

namespace kxc {

template <typename K, typename V>
MapNode<K, V>::MapNode() = default;

template <typename K, typename V>
MapNode<K, V>::MapNode(std::unordered_map<K, V> d)
    : data(std::move(d)) {}

template <typename K, typename V>
inline const TypeInfo& MapNode<K, V>::_type_info = TypeRegistry::Register("Map");

template <typename K, typename V>
inline const uint32_t MapNode<K, V>::_type_index =
    MapNode<K, V>::_type_info.runtime_index();

template <typename K, typename V>
Map<K, V>::Map() : ObjectRef(new MapNode<K, V>()) {}

template <typename K, typename V>
Map<K, V>::Map(const ObjectRef& n) : ObjectRef(n) {}

template <typename K, typename V>
Map<K, V>::Map(std::unordered_map<K, V> data) {
    SetData(new MapNode<K, V>(std::move(data)));
}

template <typename K, typename V>
Map<K, V>::Map(std::initializer_list<std::pair<const K, V>> init) {
    SetData(new MapNode<K, V>(std::unordered_map<K, V>(init)));
}

template <typename K, typename V>
size_t Map<K, V>::size() const {
    return static_cast<const MapNode<K, V>*>(get())->data.size();
}

template <typename K, typename V>
size_t Map<K, V>::count(const K& key) const {
    return static_cast<const MapNode<K, V>*>(get())->data.count(key);
}

template <typename K, typename V>
const V& Map<K, V>::at(const K& key) const {
    return static_cast<const MapNode<K, V>*>(get())->data.at(key);
}

template <typename K, typename V>
const V& Map<K, V>::operator[](const K& key) const {
    return at(key);
}

template <typename K, typename V>
void Map<K, V>::Set(const K& key, const V& val) {
    const_cast<MapNode<K, V>*>(static_cast<const MapNode<K, V>*>(get()))
        ->data[key] = val;
}

template <typename K, typename V>
typename Map<K, V>::const_iterator Map<K, V>::begin() const {
    return static_cast<const MapNode<K, V>*>(get())->data.begin();
}

template <typename K, typename V>
typename Map<K, V>::const_iterator Map<K, V>::end() const {
    return static_cast<const MapNode<K, V>*>(get())->data.end();
}

template <typename K, typename V>
typename Map<K, V>::iterator Map<K, V>::begin() {
    return const_cast<MapNode<K, V>*>(
               static_cast<const MapNode<K, V>*>(get()))
        ->data.begin();
}

template <typename K, typename V>
typename Map<K, V>::iterator Map<K, V>::end() {
    return const_cast<MapNode<K, V>*>(
               static_cast<const MapNode<K, V>*>(get()))
        ->data.end();
}

}  // namespace kxc
