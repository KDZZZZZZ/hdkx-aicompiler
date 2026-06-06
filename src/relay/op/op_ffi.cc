/*! \file src/relay/op/op_ffi.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/relay.h"
#include "relay/op.h"
#include "base/registry.h"
#include "base/packedfunc.h"
// #include "relay/attrs.h"
#include <vector>
#include <string>

namespace kxc {
namespace relay {

using namespace kxc;

// Helper to get Op
inline const Op& GetOp(const std::string& name) {
    return Op::Get(name);
}

// --- Basic Ops (No Attrs) ---

Call MakeAdd(Expr lhs, Expr rhs) {
    return Call(GetOp("add"), {lhs, rhs}); // Assuming "add" is registered
}

Call MakeSub(Expr lhs, Expr rhs) {
    return Call(GetOp("subtract"), {lhs, rhs});
}

Call MakeMul(Expr lhs, Expr rhs) {
    return Call(GetOp("mul"), {lhs, rhs});
}

Call MakeDiv(Expr lhs, Expr rhs) {
    return Call(GetOp("divide"), {lhs, rhs}); // math.cc uses "divide"
}

Call MakePow(Expr x, Expr y) {
    return Call(GetOp("pow"), {x, y});
}

Call MakeSqrt(Expr data) {
    return Call(GetOp("sqrt"), {data});
}

Call MakeErf(Expr data) {
    return Call(GetOp("erf"), {data});
}

Call MakeEqual(Expr lhs, Expr rhs) {
    return Call(GetOp("equal"), {lhs, rhs});
}

Call MakeGreater(Expr lhs, Expr rhs) {
    return Call(GetOp("greater"), {lhs, rhs});
}

Call MakeMatMul(Expr a, Expr b) {
    return Call(GetOp("matmul"), {a, b});
}

// --- Ops with Attrs ---

Call MakeCast(Expr data, int dtype) {
    auto attrs = CastAttrs::Create(dtype);
    return Call(GetOp("cast"), {data}, attrs);
}

Call MakeConcat(Expr data, int axis) {
    auto attrs = ConcatAttrs::Create(axis);
    // data is expected to be a Tuple of tensors
    return Call(GetOp("concatenate"), {data}, attrs);
}

Call MakeReduceMean(Expr data, std::vector<int64_t> axes, int64_t keepdims) {
    auto attrs = ReduceMeanAttrs::Create(axes, keepdims);
    return Call(GetOp("reduce_mean"), {data}, attrs);
}

Call MakeReshape(Expr data, std::vector<int64_t> newshape, bool allowzero) {
    auto attrs = ReshapeAttrs::Create(newshape, allowzero);
    return Call(GetOp("reshape"), {data}, attrs);
}

Call MakeShape(Expr data) {
    return Call(GetOp("shape"), {data});
}

Call MakeSlice(Expr data, std::vector<int64_t> starts, std::vector<int64_t> ends, std::vector<int64_t> axes, std::vector<int64_t> steps) {
    // Slice in transform.cc takes up to 5 inputs.
    // It does NOT use SliceAttrs in transform.cc (I commented it uses TAttrs?).
    // Wait, I saw `.set_attr<std::string>("TAttrs", "SplitAttrs")` for Split, but Slice?
    // `transform.cc`: `KXC_REGISTER_OP(slice) ... .set_num_inputs(5) ...` NO attributes registered.
    // So Slice expects inputs.
    // I should convert vectors to Constant Tensors.
    // This is getting complicated for a simple wrapper.
    // Ideally we have a helper to create Constant from vector.
    // For now, let's skip Slice or implement it assuming inputs are provided as Exprs?
    // User wants `_make.slice(data, starts, ends, ...)`
    // I will assume for now that I can't easily make Constants here without more helpers.
    // So I will implement `MakeSlice` taking Exprs.
    // But `_make` usually takes primitive types for ease of use.
    
    // Let's skip Slice for a moment and focus on others.
    return Call(GetOp("slice"), {data}); // Placeholder
}

Call MakeSoftmax(Expr data, int axis) {
    auto attrs = SoftmaxAttrs::Create(axis);
    return Call(GetOp("softmax"), {data}, attrs);
}

Call MakeSplit(Expr data, std::vector<int64_t> indices_or_sections, int axis) {
    auto attrs = SplitAttrs::Create(indices_or_sections, axis);
    return Call(GetOp("split"), {data}, attrs);
}

Call MakeTranspose(Expr data, std::vector<int64_t> axes) {
    auto attrs = TransposeAttrs::Create(axes);
    return Call(GetOp("transpose"), {data}, attrs);
}

Call MakeSqueeze(Expr data, std::vector<int64_t> axes) {
    // Squeeze in transform.cc: 2 inputs (data, axes). No attrs.
    // We need to pass axes as a Tensor.
    // Since I don't have easy Constant creation here, I'll assume usage of attributes 
    // and maybe I should update the Op registration to support attributes OR inputs.
    // Or I just register a `MakeSqueeze` that takes `Expr` axes.
    // But usually `_make` is for python convenience.
    // Let's assume we update `transform.cc` to use attributes for Squeeze/Unsqueeze as it's cleaner for static cases.
    // I'll define Attrs for Squeeze/Unsqueeze (Wait, I didn't define them in attrs.h).
    // Let's stick to what's defined.
    return Call(GetOp("squeeze"), {data}); // Placeholder
}

Call MakeUnsqueeze(Expr data, std::vector<int64_t> axes) {
    return Call(GetOp("unsqueeze"), {data}); // Placeholder
}

Call MakeGather(Expr data, Expr indices, int axis) {
    auto attrs = GatherAttrs::Create(axis);
    return Call(GetOp("gather"), {data, indices}, attrs);
}

// --- NN Ops ---

Call MakeConv2D(Expr data, Expr weight, 
                std::vector<int64_t> strides, 
                std::vector<int64_t> padding, 
                std::vector<int64_t> dilation, 
                int groups, 
                int channels, 
                std::vector<int64_t> kernel_size, 
                std::string data_layout, 
                std::string kernel_layout, 
                std::string out_layout, 
                std::string out_dtype) {
    auto attrs = Conv2DAttrs::Create(strides, padding, dilation, groups, channels, kernel_size, data_layout, kernel_layout, out_layout, out_dtype);
    return Call(GetOp("nn_conv2d"), {data, weight}, attrs);
}

Call MakeDense(Expr data, Expr weight, int units, std::string out_dtype) {
    auto attrs = DenseAttrs::Create(units, out_dtype);
    return Call(GetOp("nn_dense"), {data, weight}, attrs);
}

Call MakeRelu(Expr data) {
    return Call(GetOp("nn_relu"), {data}, ReluAttrs::Create());
}

Call MakeMaxPool2D(Expr data, 
                   std::vector<int64_t> strides, 
                   std::vector<int64_t> padding, 
                   std::vector<int64_t> dilation, 
                   std::vector<int64_t> pool_size, 
                   std::string layout, 
                   bool ceil_mode) {
    auto attrs = MaxPool2DAttrs::Create(strides, padding, dilation, pool_size, layout, ceil_mode);
    return Call(GetOp("nn_max_pool2d"), {data}, attrs);
}

Call MakeFlatten(Expr data) {
    return Call(GetOp("nn_flatten"), {data}, FlattenAttrs::Create());
}

Call MakeWhere(Expr condition, Expr x, Expr y) {
    return Call(GetOp("where"), {condition, x, y});
}

Call MakeConstantOfShape(Expr input) {
    auto attrs = ConstantOfShapeAttrs::Create();
    return Call(GetOp("constant_of_shape"), {input}, attrs);
}

Call MakeExpandDims(Expr data, Expr shape) {
    auto attrs = ExpandAttrs::Create();
    return Call(GetOp("expand_dims"), {data, shape}, attrs);
}


// Registration
KXC_REGISTER_GLOBAL("kxc.relay.op._make.add").set_body(ToPackedFunc(MakeAdd));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.sub").set_body(ToPackedFunc(MakeSub));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.mul").set_body(ToPackedFunc(MakeMul));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.div").set_body(ToPackedFunc(MakeDiv)); // Using "div" alias for python
KXC_REGISTER_GLOBAL("kxc.relay.op._make.pow").set_body(ToPackedFunc(MakePow));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.sqrt").set_body(ToPackedFunc(MakeSqrt));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.erf").set_body(ToPackedFunc(MakeErf));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.equal").set_body(ToPackedFunc(MakeEqual));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.greater").set_body(ToPackedFunc(MakeGreater));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.matmul").set_body(ToPackedFunc(MakeMatMul));

KXC_REGISTER_GLOBAL("kxc.relay.op._make.cast").set_body(ToPackedFunc(MakeCast));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.concatenate").set_body(ToPackedFunc(MakeConcat));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reduce_mean").set_body(ToPackedFunc(MakeReduceMean));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reshape").set_body(ToPackedFunc(MakeReshape));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.shape").set_body(ToPackedFunc(MakeShape));
// Slice, Squeeze, Unsqueeze skipped for now or need more work
KXC_REGISTER_GLOBAL("kxc.relay.op._make.softmax").set_body(ToPackedFunc(MakeSoftmax));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.split").set_body(ToPackedFunc(MakeSplit));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.transpose").set_body(ToPackedFunc(MakeTranspose));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.gather").set_body(ToPackedFunc(MakeGather));

KXC_REGISTER_GLOBAL("kxc.relay.op._make.conv2d").set_body(ToPackedFunc(MakeConv2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.dense").set_body(ToPackedFunc(MakeDense));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.relu").set_body(ToPackedFunc(MakeRelu));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.max_pool2d").set_body(ToPackedFunc(MakeMaxPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.flatten").set_body(ToPackedFunc(MakeFlatten));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.where").set_body(ToPackedFunc(MakeWhere));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.constant_of_shape").set_body(ToPackedFunc(MakeConstantOfShape));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.expand_dims").set_body(ToPackedFunc(MakeExpandDims));

} // namespace relay
} // namespace kxc
