/*! \file include/kxc/te/topi/window.h
 * \brief 定义滑窗算子 padding 展开与输出尺寸的唯一权威实现。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

/*! \brief 展开后的四边 padding，顺序为 [上, 左, 下, 右]。 */
struct Padding2D {
    int64_t top = 0;
    int64_t left = 0;
    int64_t bottom = 0;
    int64_t right = 0;
};

/*!
 * \brief 将紧凑 padding 属性展开为四边形式。
 *
 * 这是 padding 展开顺序的唯一权威实现。Relay 类型推导与 TOPI compute 必须共同
 * 调用它：两侧一旦各自展开，就可能对同一属性得出不同的上下左右，进而算出不同的
 * 输出 shape。
 *
 * 接受 0、1、2 或 4 个元素：空表示无 padding；1 个表示四边相同；2 个表示
 * [高, 宽] 方向对称；4 个表示 [上, 左, 下, 右]，允许非对称。
 *
 * \tparam Container 提供 size() 与 operator[] 的整数序列，例如 Array<int64_t>。
 */
template <typename Container>
Padding2D ExpandPadding2D(const Container& values) {
    Padding2D out;
    const std::size_t count = values.size();
    if (count == 0) {
        return out;
    }
    if (count == 1) {
        out.top = out.left = out.bottom = out.right = static_cast<int64_t>(values[0]);
        return out;
    }
    if (count == 2) {
        out.top = out.bottom = static_cast<int64_t>(values[0]);
        out.left = out.right = static_cast<int64_t>(values[1]);
        return out;
    }
    if (count >= 4) {
        out.top = static_cast<int64_t>(values[0]);
        out.left = static_cast<int64_t>(values[1]);
        out.bottom = static_cast<int64_t>(values[2]);
        out.right = static_cast<int64_t>(values[3]);
        return out;
    }
    throw std::runtime_error("padding expects 0, 1, 2 or 4 elements");
}

/*!
 * \brief 计算滑窗算子（卷积、池化）在一个空间维度上的输出尺寸。
 *
 * 这是该公式的唯一权威实现。Relay 类型推导与 TOPI compute 必须共同调用它：
 * 两侧一旦各自实现，推导出的 TensorType 就会与实际 compute 的 shape 不一致，
 * 并在 LowerCompilationUnit 的边界校验处以 "Unit TE output shape mismatch" 暴露，
 * 报错指向编译器内部不变量而非真正出错的算子属性。
 *
 * 调用方负责保证 stride 与 dilation 为正、且 numerator 非负；本函数只做算术，
 * 不做校验，因为 PrimExpr 实例化下无法在编译期检查这些条件。
 *
 * \tparam T 维度表达式类型：类型推导实例化为 int64_t，TOPI compute 实例化为 PrimExpr。
 * \param input 输入空间维度。
 * \param kernel 卷积核或池化窗口在该维度上的大小。
 * \param pad_before 该维度起始侧 padding。
 * \param pad_after 该维度结束侧 padding。
 * \param stride 步长。
 * \param dilation 膨胀系数；池化传 1。
 * \param ceil_mode 为 true 时向上取整，否则向下取整。
 */
template <typename T>
T WindowOutputExtent(T input, T kernel, T pad_before, T pad_after, T stride, T dilation,
                     bool ceil_mode) {
    T effective_kernel = (kernel - 1) * dilation + 1;
    T numerator = input + pad_before + pad_after - effective_kernel;
    if (ceil_mode) {
        return (numerator + stride - 1) / stride + 1;
    }
    return numerator / stride + 1;
}

}  // namespace topi
}  // namespace te
}  // namespace kxc
