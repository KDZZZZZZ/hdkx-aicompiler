/*! \file include/kxc/support/detail/array_inl.h
 * \brief Visible template implementation for Array.
 *
 * This file is included by container.h and is not a stable direct-include API.
 */

#pragma once

namespace kxc {

template <typename T>
ArrayNode<T>::ArrayNode() = default;

template <typename T>
ArrayNode<T>::ArrayNode(std::vector<T> d) : data(std::move(d)) {}

template <typename T>
inline const TypeInfo& ArrayNode<T>::_type_info = TypeRegistry::Register("Array");

template <typename T>
inline const uint32_t ArrayNode<T>::_type_index =
    ArrayNode<T>::_type_info.runtime_index();

template <typename T>
Array<T>::Array() : ObjectRef(new ArrayNode<T>()) {}

template <typename T>
Array<T>::Array(const ObjectRef& n) : ObjectRef(n) {}

template <typename T>
Array<T>::Array(std::vector<T> data) {
    SetData(new ArrayNode<T>(std::move(data)));
}

template <typename T>
Array<T>::Array(std::initializer_list<T> init) {
    SetData(new ArrayNode<T>(std::vector<T>(init)));
}

template <typename T>
const T& Array<T>::operator[](size_t i) const {
    return static_cast<const ArrayNode<T>*>(get())->data[i];
}

template <typename T>
T& Array<T>::operator[](size_t i) {
    return const_cast<ArrayNode<T>*>(
               static_cast<const ArrayNode<T>*>(get()))
        ->data[i];
}

template <typename T>
size_t Array<T>::size() const {
    return static_cast<const ArrayNode<T>*>(get())->data.size();
}

template <typename T>
bool Array<T>::empty() const {
    return static_cast<const ArrayNode<T>*>(get())->data.empty();
}

template <typename T>
void Array<T>::push_back(const T& val) {
    const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))
        ->data.push_back(val);
}

template <typename T>
void Array<T>::push_back(T&& val) {
    const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))
        ->data.push_back(std::move(val));
}

template <typename T>
void Array<T>::insert(typename std::vector<T>::const_iterator pos,
                      const T& val) {
    const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))
        ->data.insert(pos, val);
}

template <typename T>
void Array<T>::erase(typename std::vector<T>::const_iterator pos) {
    const_cast<ArrayNode<T>*>(static_cast<const ArrayNode<T>*>(get()))
        ->data.erase(pos);
}

template <typename T>
typename Array<T>::const_iterator Array<T>::begin() const {
    return static_cast<const ArrayNode<T>*>(get())->data.begin();
}

template <typename T>
typename Array<T>::const_iterator Array<T>::end() const {
    return static_cast<const ArrayNode<T>*>(get())->data.end();
}

template <typename T>
typename Array<T>::iterator Array<T>::begin() {
    return const_cast<ArrayNode<T>*>(
               static_cast<const ArrayNode<T>*>(get()))
        ->data.begin();
}

template <typename T>
typename Array<T>::iterator Array<T>::end() {
    return const_cast<ArrayNode<T>*>(
               static_cast<const ArrayNode<T>*>(get()))
        ->data.end();
}

}  // namespace kxc
