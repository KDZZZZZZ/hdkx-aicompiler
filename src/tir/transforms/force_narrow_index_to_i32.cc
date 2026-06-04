/*! \file src/tir/transforms/force_narrow_index_to_i32.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/force_narrow_index_to_i32.h"

#include "base/pass.h"
#include "tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

Buffer NarrowBufferIndices(const Buffer& buffer) {
    if (!buffer.defined()) {
        return buffer;
    }
    bool changed = false;

    Array<PrimExpr> new_shape;
    for (const auto& s : buffer->shape) {
        PrimExpr ns = pass_utils::NarrowIntImmToInt32IfPossible(s);
        if (ns.get() != s.get()) {
            changed = true;
        }
        new_shape.push_back(ns);
    }

    Array<PrimExpr> new_strides;
    for (const auto& s : buffer->strides) {
        PrimExpr ns = pass_utils::NarrowIntImmToInt32IfPossible(s);
        if (ns.get() != s.get()) {
            changed = true;
        }
        new_strides.push_back(ns);
    }

    PrimExpr new_elem_offset = pass_utils::NarrowIntImmToInt32IfPossible(buffer->elem_offset);
    if (new_elem_offset.get() != buffer->elem_offset.get()) {
        changed = true;
    }

    if (!changed) {
        return buffer;
    }
    return Buffer(buffer->data, buffer->dtype, new_shape, new_strides, new_elem_offset,
                  buffer->name, buffer->data_alignment, buffer->offset_factor);
}

class ForceNarrowIndexToI32Rewriter : public TIRPass {
protected:
    PrimFunc VisitPrimFunc(const PrimFunc& func) override {
        PrimFunc rewritten = TIRPass::VisitPrimFunc(func);
        bool changed = false;
        Map<Var, Buffer> new_buffer_map;
        for (const auto& kv : rewritten->buffer_map) {
            Buffer narrowed = NarrowBufferIndices(kv.second);
            if (narrowed.get() != kv.second.get()) {
                changed = true;
            }
            new_buffer_map.Set(kv.first, narrowed);
        }
        if (!changed) {
            return rewritten;
        }
        return PrimFunc(rewritten->params, rewritten->body, new_buffer_map, rewritten->attrs);
    }

    PrimExpr VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitLoad(op, ref);
        const auto* load = rewritten.As<LoadNode>();
        if (!load) {
            return rewritten;
        }
        PrimExpr new_index = pass_utils::NarrowIntImmToInt32IfPossible(load->index);
        if (new_index.get() == load->index.get()) {
            return rewritten;
        }
        return Load(load->buffer_var, new_index, load->predicate);
    }

    Stmt VisitStore(const StoreNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitStore(op, ref);
        const auto* store = rewritten.As<StoreNode>();
        if (!store) {
            return rewritten;
        }
        PrimExpr new_index = pass_utils::NarrowIntImmToInt32IfPossible(store->index);
        if (new_index.get() == store->index.get()) {
            return rewritten;
        }
        return Store(store->buffer_var, store->value, new_index, store->predicate);
    }

    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node) {
            return rewritten;
        }
        PrimExpr new_min = pass_utils::NarrowIntImmToInt32IfPossible(for_node->min);
        PrimExpr new_extent = pass_utils::NarrowIntImmToInt32IfPossible(for_node->extent);
        if (new_min.get() == for_node->min.get() && new_extent.get() == for_node->extent.get()) {
            return rewritten;
        }
        return For(for_node->loop_var, new_min, new_extent, for_node->for_type, for_node->body);
    }
};

}  // namespace

PrimFunc ForceNarrowIndexToI32Pass(const PrimFunc& func) {
    ForceNarrowIndexToI32Rewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc

