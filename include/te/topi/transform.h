#pragma once
#include "te/te.h"
#include "te/topi/tags.h"
#include "te/topi/utils.h"
#include "base/container.h"
#include <vector>
#include <numeric>
#include <set>

namespace kxc {
namespace te {
namespace topi {

// Transpose
inline Tensor transpose(const Tensor& x, Array<int> axes, std::string name = "transpose", std::string tag = kInjective) {
    size_t ndim = x->shape.size();
    if (axes.empty()) {
        // Default: reverse
        for (int i = (int)ndim - 1; i >= 0; --i) {
            axes.push_back(i);
        }
    }
    
    Array<PrimExpr> output_shape;
    for (int axis : axes) {
        output_shape.push_back(x->shape[axis]);
    }
    
    return compute(
        output_shape,
        [&](const Array<Var>& indices) {
            Array<PrimExpr> input_indices;
            for(size_t i=0; i<ndim; ++i) input_indices.push_back(0);
            
            // Map output indices back to input
            // out[i, j] -> in[j, i] if swap
            // out_idx[k] corresponds to axis axes[k]
            // so in_idx[axes[k]] = out_idx[k]
            for (size_t i = 0; i < ndim; ++i) {
                input_indices[axes[i]] = indices[i];
            }
            return x(input_indices);
        },
        name,
        tag
    );
}

// Expand Dims
inline Tensor expand_dims(const Tensor& x, int axis, int num_newaxis = 1, std::string name = "expand_dims", std::string tag = kBroadcast) {
    size_t ndim = x->shape.size();
    if (axis < 0) axis += (int)ndim + 1;
    
    Array<PrimExpr> output_shape;
    for (size_t i = 0; i < (size_t)axis; ++i) output_shape.push_back(x->shape[i]);
    for (int i = 0; i < num_newaxis; ++i) output_shape.push_back(1);
    for (size_t i = axis; i < ndim; ++i) output_shape.push_back(x->shape[i]);
    
    return compute(
        output_shape,
        [&](const Array<Var>& indices) {
            Array<PrimExpr> input_indices;
            size_t idx_counter = 0;
            for (size_t i = 0; i < output_shape.size(); ++i) {
                // If this is a new axis (between axis and axis+num), skip it
                if (i >= (size_t)axis && i < (size_t)(axis + num_newaxis)) {
                    continue;
                }
                input_indices.push_back(indices[i]);
            }
            return x(input_indices);
        },
        name,
        tag
    );
}

// Squeeze
inline Tensor squeeze(const Tensor& x, Array<int> axes = {}, std::string name = "squeeze", std::string tag = kInjective) {
    size_t ndim = x->shape.size();
    std::vector<size_t> squeeze_axes = GetRealAxis(ndim, axes);
    std::set<size_t> sq_set(squeeze_axes.begin(), squeeze_axes.end());
    
    if (sq_set.empty()) {
        // Squeeze all 1s
        for (size_t i = 0; i < ndim; ++i) {
            int64_t val = 0;
            if (GetConstInt(x->shape[i], &val) && val == 1) {
                sq_set.insert(i);
            }
        }
    }
    
    Array<PrimExpr> output_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (!sq_set.count(i)) {
            output_shape.push_back(x->shape[i]);
        }
    }
    
    return compute(
        output_shape,
        [&](const Array<Var>& indices) {
            Array<PrimExpr> input_indices;
            size_t out_idx = 0;
            for (size_t i = 0; i < ndim; ++i) {
                if (sq_set.count(i)) {
                    input_indices.push_back(0);
                } else {
                    input_indices.push_back(indices[out_idx++]);
                }
            }
            return x(input_indices);
        },
        name,
        tag
    );
}

// Concatenate
inline Tensor concatenate(const Array<Tensor>& inputs, int axis = 0, std::string name = "concatenate", std::string tag = kInjective) {
    if (inputs.empty()) return Tensor(); // Error?
    
    size_t ndim = inputs[0]->shape.size();
    if (axis < 0) axis += (int)ndim;
    
    Array<PrimExpr> output_shape = inputs[0]->shape;
    PrimExpr axis_len = 0;
    for (const auto& t : inputs) {
        axis_len = axis_len + t->shape[axis];
    }
    output_shape[axis] = axis_len;
    
    return compute(
        output_shape,
        [&](const Array<Var>& indices) {
            // Logic: Iterate inputs, check range.
            // Since we can't easily do recursive Select in generic lambda without fold,
            // we'll build the Select chain manually.
            
            PrimExpr current_idx = indices[axis];
            PrimExpr ret; // Initialized to last one or default
            
            PrimExpr offset = 0;
            
            // Reverse iteration to build Select(cond, val, else_val)
            // But we need to know offsets.
            // Let's do forward and accumulate offsets.
            // Actually, best to do it recursively or loop.
            // We want: if (idx < s0) t0(idx) else if (idx < s0+s1) t1(idx-s0) ...
            
            // To build this as expression:
            // Select(idx < s0, t0(...), Select(idx < s0+s1, t1(...), ...))
            
            // We need to construct the expression.
            // Last fallback: last tensor (or 0 if out of bound, but assuming valid).
            
            // Let's pre-calculate offsets if they are constant?
            // Even if symbolic, we can chain Add.
            
            // We need to start from the last one to wrap.
            // inputs[N-1]
            // Unused last_indices removed
            
            // This is getting complicated for symbolic offsets in a loop.
            // Let's build a vector of cumulative offsets first.
            Array<PrimExpr> offsets;
            offsets.push_back(0);
            for (size_t i = 0; i < inputs.size() - 1; ++i) {
                // offsets.back() works on Array? No, need operator[] or back()
                // Array has operator[].
                offsets.push_back(offsets[offsets.size()-1] + inputs[i]->shape[axis]);
            }
            
            // Now build from back
            size_t n = inputs.size();
            // Start with last tensor
            Array<PrimExpr> idx_n; 
            for(auto v : indices) idx_n.push_back(v);
            // Use operator[] for mutable update
            idx_n[axis] = idx_n[axis] - offsets[n-1];
            ret = inputs[n-1](idx_n);
            
            for (int i = (int)n - 2; i >= 0; --i) {
                Array<PrimExpr> idx_i;
                for(auto v : indices) idx_i.push_back(v);
                idx_i[axis] = idx_i[axis] - offsets[i];
                
                PrimExpr cond = indices[axis] < (offsets[i] + inputs[i]->shape[axis]);
                ret = Select(cond, inputs[i](idx_i), ret);
            }
            
            return ret;
        },
        name,
        tag
    );
}

} // namespace topi
} // namespace te
} // namespace kxc
