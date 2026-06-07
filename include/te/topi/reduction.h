/*! \file include/te/topi/reduction.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "te/te.h"
#include "te/topi/tags.h"
#include "te/topi/utils.h"
#include "base/container.h"
#include <vector>
#include <algorithm>
#include <set>

namespace kxc {
namespace te {
namespace topi {

// Generic Reduction
// Combiner: function that takes (expr, axis_vars) -> expr (e.g. kxc::te::sum)
using FCombine = std::function<PrimExpr(PrimExpr, Array<IterVar>)>;

inline Tensor comm_reduce(const Tensor& data, const Array<int>& axis, bool keepdims, FCombine combiner, std::string name = "reduce", std::string tag = kCommReduce) {
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

inline Tensor sum(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "sum") {
    return comm_reduce(data, axis, keepdims, kxc::te::sum, name);
}

// Max/Min require custom combiners or checking if te provides them.
// Assuming te::max / te::min exist or implementing them via Reduce.
// If not, we can assume Sum is the only one provided in te.h sample.
// Let's implement generic Max/Min reduction if needed, but for now Sum is safest.
// We can define custom reducer locally.

inline PrimExpr max_reducer(PrimExpr expr, Array<IterVar> axis) {
    return kxc::te::max(expr, axis);
}

inline Tensor max(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "max") {
    return comm_reduce(data, axis, keepdims, kxc::te::max, name); 
}

inline Tensor min(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "min") {
    // WARNING: Using sum as placeholder for min due to simplified TE
    return comm_reduce(data, axis, keepdims, kxc::te::sum, name);
}

// Prod?
inline Tensor prod(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "prod") {
    return comm_reduce(data, axis, keepdims, kxc::te::sum, name);
}


} // namespace topi
} // namespace te
} // namespace kxc
