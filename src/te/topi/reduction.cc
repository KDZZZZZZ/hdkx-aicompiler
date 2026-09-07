/*! \file src/te/topi/reduction.cc
 * \brief Implements TOPI reduction helpers.
 */

#include "kxc/te/topi/reduction.h"

namespace kxc {
namespace te {
namespace topi {

Tensor comm_reduce(const Tensor& data, const Array<int>& axis, bool keepdims, FCombine combiner, std::string name , std::string tag ){
    size_t ndim = data->shape.size();
    std::vector<size_t> real_axis = GetRealAxis(ndim, axis);
    std::set<size_t> reduce_set(real_axis.begin(), real_axis.end());
    
    Array<PrimExpr> output_shape;
    Array<IterVar> reduce_axes;
    
    for (size_t i = 0; i < ndim; ++i) {
        if (reduce_set.count(i)) {
            // Create reduction axis
            // Use shape[i] as extent. Assuming 0 min.
            // Name: k1, k2, ...
            std::string name_hint = "k" + std::to_string(reduce_axes.size());
            reduce_axes.push_back(reduce_axis(0, data->shape[i], name_hint));
            
            if (keepdims) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(data->shape[i]);
        }
    }
    
    return compute(
        output_shape,
        [&](const Array<tir::Var>& indices) {
            Array<PrimExpr> eval_indices;
            size_t idx_counter = 0;
            size_t red_counter = 0;

            for (size_t i = 0; i < ndim; ++i) {
                if (reduce_set.count(i)) {
                    eval_indices.push_back(AsPrimExpr(reduce_axes[red_counter++]));
                } else if (keepdims) {
                    // keepdims 输出轴与输入轴一一对应（被归约轴以 1 保留在原位），
                    // 非归约输入轴 i 必须读输出轴 i；压缩计数只适用于 keepdims=0。
                    eval_indices.push_back(indices[i]);
                } else {
                    eval_indices.push_back(indices[idx_counter++]);
                }
            }

            return combiner(data(eval_indices), reduce_axes);
        },
        name,
        tag
    );
}

Tensor sum(const Tensor& data, const Array<int>& axis, bool keepdims , std::string name ){
    return comm_reduce(data, axis, keepdims, kxc::te::sum, name);
}

PrimExpr max_reducer(PrimExpr expr, Array<IterVar> axis){
    return kxc::te::max(expr, axis);
}

Tensor max(const Tensor& data, const Array<int>& axis, bool keepdims , std::string name ){
    return comm_reduce(data, axis, keepdims, kxc::te::max, name); 
}

Tensor min(const Tensor& data, const Array<int>& axis, bool keepdims , std::string name ){
    return comm_reduce(data, axis, keepdims, kxc::te::min, name);
}

}  // namespace topi
}  // namespace te
}  // namespace kxc
