/*! \file src/relay/op/tensor/shape_ops.cc
 * \brief 注册受限形状值 Relay 算子及其 FRelayToTE compute。
 *
 * 本文件是 M3 S1/S2 形状值算子的唯一落点：
 * - shape_of 把输入张量的各维（静态常量或受限符号维）物化为 int64 行向量。
 * - 形状链的准入、越界与溢出拒绝由 src/compiler/shape/ 的受限准备路径负责；
 *   本层只提供类型规则与真实 TE/TIR 计算，不引入第二套形状解释器。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/te.h"
#include "kxc/tir/expr.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

void RequireInputCount(const char* op_name, const Array<te::Tensor>& inputs, size_t expected) {
    if (inputs.size() != expected) {
        throw std::runtime_error(std::string(op_name) + " expects exactly " +
                                 std::to_string(expected) + " input(s)");
    }
}

const TensorTypeNode* RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
    return tensor_type;
}

te::Tensor RequireDefined(const char* op_name, const te::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::runtime_error(std::string(op_name) + " lowering returned undefined tensor");
    }
    return tensor;
}

Array<kxc::tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type,
                                              const std::string& op_name) {
    Array<kxc::tir::PrimExpr> shape;
    for (int64_t dim : type->shape) {
        if (dim < 0) {
            throw std::runtime_error(op_name + " lowering requires static output shape");
        }
        shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    return shape;
}

}  // namespace

// 把输入张量各维写入 int64[rank] 输出：静态维写入常量，受限符号维在
// bounded 降级中经既有 runtime extent ABI 读取真实运行期数值。
te::Tensor ShapeOfCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("shape_of", inputs, 1);
    const auto* tensor_type = RequireTensorOutput("shape_of", out_type);
    const size_t rank = inputs[0]->shape.size();
    if (rank == 0 || tensor_type->shape.size() != 1 ||
        tensor_type->shape[0] != static_cast<int64_t>(rank)) {
        throw std::runtime_error(
            "shape_of lowering output must be int64[rank(data)] with rank >= 1");
    }
    for (const kxc::tir::PrimExpr& extent : inputs[0]->shape) {
        if (!extent.defined()) {
            throw std::runtime_error("shape_of lowering received an undefined input extent");
        }
    }
    const kxc::tir::DataType index_dtype = kxc::tir::DataType::Int(32);
    const kxc::tir::DataType value_dtype = kxc::tir::DataType::Int(64);
    return RequireDefined("shape_of", te::compute(
        {kxc::tir::IntImm(static_cast<int64_t>(rank), kxc::tir::DataType::Int(64))},
        [input = inputs[0], rank, index_dtype, value_dtype](
            const Array<kxc::tir::Var>& indices) {
            // body(i) = i == 0 ? dim0 : i == 1 ? dim1 : ...；rank 在编译期固定，
            // 静态维为 IntImm，受限符号维为 runtime extent load。
            kxc::tir::PrimExpr selected = kxc::tir::IntImm(0, value_dtype);
            for (size_t axis = rank; axis > 0; --axis) {
                selected = kxc::tir::Select(
                    kxc::tir::EQ(indices[0],
                                 kxc::tir::IntImm(static_cast<int64_t>(axis - 1),
                                                  index_dtype)),
                    input->shape[axis - 1],
                    selected);
            }
            return selected;
        },
        "T_shape_of"));
}

}  // namespace relay
}  // namespace kxc
