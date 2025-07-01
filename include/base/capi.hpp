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
namespace kxcomp{
namespace base{



class file_op{
public:
    static std::string read_file(const std::string& path){}
    static void write_file(const std::string& path, const std::string& content){}
    static void delete_file(const std::string& path){}
};

constexpr int32_t kDynamicIndex = -1;
constexpr int32_t kInvalidIndex = -2;

struct TypeInfo{
    std::string type_key;
    std::function<object*()> constructor;
    std::function<void(object*)> deleter;
    int32_t parent_index = kInvalidIndex;
    uint32_t type_key_hash = 0;
    TypeInfo(const std::string& key):type_key(key){
        type_key_hash = std::hash<std::string>{}(key);
    }
};
class TypeRegistry{
public:
    static TypeRegistry* Global(){
        static TypeRegistry instance;
        return &instance;
    }
    int32_t RegisterType(const TypeInfo& type_info){
        int32_t index = type_info_list.size();
        type_info_list.push_back(type_info);
        key_to_index[type_info.type_key] = index;
        return index;
    }
    int32_t GetTypeIndex(const std::string& key){
        auto it = key_to_index.find(key);
        if(it == key_to_index.end()){
            return kInvalidIndex;
        }
        return it->second;
    }
    const TypeInfo& GetTypeInfo(int32_t index){
        return type_info_list[index];
    }
private:
    std::unordered_map<std::string, int32_t> key_to_index;
    std::vector<TypeInfo> type_info_list;
};

struct ObjectHeader {
    std::atomic<int32_t> ref_counter = 0;
    std::function<void(void*)> deleter;
    int32_t type_index = kInvalidIndex;
};

class object{
protected:
    ObjectHeader header;
public:
    object():header(){
        header.ref_counter = 0;
        header.deleter = nullptr;
        header.type_index = kInvalidIndex;
    }
    int32_t GetTypeIndex(){
        return header.type_index;
    }
    std::string GetTypeName(){}
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
    
    int GetRefCount(){
        return header.ref_counter;
    }
    void SetDeleter(std::function<void(void*)> deleter){
        header.deleter = deleter;
    }
};

class kstruct{};
class objectPtr{};

}
}