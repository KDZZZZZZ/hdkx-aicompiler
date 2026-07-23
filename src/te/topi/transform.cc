/*! \file src/te/topi/transform.cc
 * \brief Implements TOPI transform helpers.
 */

#include "kxc/te/topi/transform.h"

#include <limits>
#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

Tensor transpose(const Tensor& x, Array<int> axes, std::string name , std::string tag ){
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
        [&](const Array<tir::Var>& indices) {
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

Tensor expand_dims(const Tensor& x, int axis, int num_newaxis , std::string name , std::string tag ){
    size_t ndim = x->shape.size();
    if (axis < 0) axis += (int)ndim + 1;
    
    Array<PrimExpr> output_shape;
    for (size_t i = 0; i < (size_t)axis; ++i) output_shape.push_back(x->shape[i]);
    for (int i = 0; i < num_newaxis; ++i) output_shape.push_back(1);
    for (size_t i = axis; i < ndim; ++i) output_shape.push_back(x->shape[i]);
    
    return compute(
        output_shape,
        [&](const Array<tir::Var>& indices) {
            Array<PrimExpr> input_indices;
            size_t idx_counter = 0;
            for (size_t i = 0; i < output_shape.size(); ++i) {
                // New axes do not map to input indices.
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

Tensor squeeze(const Tensor& x, Array<int> axes , std::string name , std::string tag ){
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
        [&](const Array<tir::Var>& indices) {
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

Tensor concatenate(const Array<Tensor>& inputs, int axis, std::string name, std::string tag) {
    if (inputs.empty()) {
        throw std::runtime_error("topi::concatenate expects at least one input tensor");
    }
    if (!inputs[0].defined()) {
        throw std::runtime_error("topi::concatenate input tensor is undefined");
    }
    const size_t rank = inputs[0]->shape.size();
    if (rank == 0) {
        throw std::runtime_error("topi::concatenate requires rank >= 1 inputs");
    }
    if (axis < 0) axis += static_cast<int>(rank);
    if (axis < 0 || axis >= static_cast<int>(rank)) {
        throw std::runtime_error("topi::concatenate axis out of range");
    }

    Array<int64_t> axis_extents;
    Array<PrimExpr> output_shape;
    for (const PrimExpr& extent : inputs[0]->shape) output_shape.push_back(extent);
    int64_t axis_total = 0;
    for (const Tensor& input : inputs) {
        if (!input.defined()) {
            throw std::runtime_error("topi::concatenate input tensor is undefined");
        }
        if (input->shape.size() != rank) {
            throw std::runtime_error("topi::concatenate input rank mismatch");
        }
        if (input->dtype != inputs[0]->dtype) {
            throw std::runtime_error("topi::concatenate input dtype mismatch");
        }
        for (size_t index = 0; index < rank; ++index) {
            const auto* extent = input->shape[index].As<tir::IntImmNode>();
            const auto* reference = inputs[0]->shape[index].As<tir::IntImmNode>();
            if (!extent || !reference || extent->value < 0 || reference->value < 0) {
                throw std::runtime_error("topi::concatenate requires static non-negative input dimensions");
            }
            if (static_cast<int>(index) != axis && extent->value != reference->value) {
                throw std::runtime_error("topi::concatenate non-axis dimensions must exactly match");
            }
            if (static_cast<int>(index) == axis) {
                if (axis_total > std::numeric_limits<int64_t>::max() - extent->value) {
                    throw std::runtime_error("topi::concatenate axis extent sum overflows int64");
                }
                axis_total += extent->value;
                axis_extents.push_back(extent->value);
            }
        }
    }
    output_shape[static_cast<size_t>(axis)] =
        tir::IntImm(axis_total, tir::DataType::Int(64));

    return compute(
        output_shape,
        [inputs, axis, axis_extents](const Array<tir::Var>& indices) {
            Array<PrimExpr> offsets;
            offsets.push_back(tir::IntImm(0, tir::DataType::Int(64)));
            for (size_t i = 1; i < inputs.size(); ++i) {
                offsets.push_back(offsets[i - 1] +
                                  tir::IntImm(axis_extents[i - 1], tir::DataType::Int(64)));
            }
            Array<PrimExpr> last_indices;
            for (const tir::Var& value : indices) last_indices.push_back(value);
            last_indices[static_cast<size_t>(axis)] =
                last_indices[static_cast<size_t>(axis)] - offsets[inputs.size() - 1];
            PrimExpr result = inputs[inputs.size() - 1](last_indices);
            for (int i = static_cast<int>(inputs.size()) - 2; i >= 0; --i) {
                Array<PrimExpr> input_indices;
                for (const tir::Var& value : indices) input_indices.push_back(value);
                input_indices[static_cast<size_t>(axis)] =
                    input_indices[static_cast<size_t>(axis)] - offsets[static_cast<size_t>(i)];
                const PrimExpr limit = offsets[static_cast<size_t>(i)] +
                    tir::IntImm(axis_extents[static_cast<size_t>(i)], tir::DataType::Int(64));
                // Select is lowered lazily, so a zero-extent branch never evaluates its Load.
                result = Select(indices[static_cast<size_t>(axis)] < limit,
                                inputs[static_cast<size_t>(i)](input_indices), result);
            }
            return result;
        },
        name, tag);
}

}  // namespace topi
}  // namespace te
}  // namespace kxc
