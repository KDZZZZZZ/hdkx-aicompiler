/*! \file include/base/arena.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <vector>
#include <cstddef> // for std::byte, size_t
#include <cstdint> // for uintptr_t

namespace kxc {

class Arena {
public:
    // 默认创建一个4MB的Arena
    explicit Arena(size_t size_bytes = 4 * 1024 * 1024) {
        memory_block_.resize(size_bytes);
        current_ptr_ = memory_block_.data();
    }

    // Arena不可拷贝或移动
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // 核心分配函数：分配指定大小和对齐的内存
    void* Allocate(size_t size, size_t alignment) {
        // 1. 计算对齐后的地址
        //    这是保证返回的地址满足特定对齐要求的标准做法
        uintptr_t current_addr = reinterpret_cast<uintptr_t>(current_ptr_);
        uintptr_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
        
        // 2. 计算对齐所需的额外空间
        size_t padding = aligned_addr - current_addr;

        // 3. 检查是否有足够空间
        if ((current_ptr_ + padding + size) > (memory_block_.data() + memory_block_.size())) {
            // 内存不足，这是Arena最简单的处理方式。
            // 实际项目中可能会扩展Arena（分配新的block）或抛出异常。
            return nullptr; 
        }

        // 4. 移动指针，完成分配
        current_ptr_ += padding + size;
        return reinterpret_cast<void*>(aligned_addr);
    }

private:
    std::vector<std::byte> memory_block_; // 底层内存块
    std::byte* current_ptr_ = nullptr;    // 指针碰撞的"指针"
};

} // namespace my_compiler::runtime
