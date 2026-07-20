/*! \file include/relay/transforms/lower.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include <cstdint>

#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {
namespace relay {

/*! \brief 保存单个 Relay Constant 与 lowered 参数槽之间的稳定绑定。 */
class ConstantBindingNode final : public Object {
public:
    /*! \brief 当前规范化 Relay Function 内确定性的常量 key。 */
    String key;
    /*! \brief 原始 Storage-backed 常量值，生命周期由对象引用保持。 */
    runtime::NDArray value;
    /*! \brief 该常量在 PrimFunc params 中的绝对参数下标。 */
    int64_t param_index{-1};

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE_WITH_KEY(ConstantBindingNode, "kxc.relay.ConstantBindingNode")

/*! \brief ConstantBindingNode 的类型安全对象句柄。 */
class ConstantBinding : public ObjectRef {
public:
    /*! \brief 构造并校验常量 key、NDArray 和参数下标。 */
    ConstantBinding(String key, runtime::NDArray value, int64_t param_index);
    /*! \brief 从通用对象引用恢复绑定，并验证运行时节点类型。 */
    explicit ConstantBinding(const ObjectRef& ref);

    /*! \brief 校验 key、NDArray 和参数下标，恢复同类型节点时也必须执行。 */
    void Validate() const;
    /*! \brief 返回经过类型检查的只读绑定节点。 */
    const ConstantBindingNode* operator->() const;
};

/*! \brief 保存 TIR 计算和全部常量 payload 的完整 lowering 产物。 */
class LoweredFunctionNode final : public Object {
public:
    /*! \brief Relay/TE lowering 得到的单个 TIR PrimFunc。 */
    tir::PrimFunc prim_func;
    KXC_OBJECT_DECLARE

private:
    friend class LoweredFunction;
    /*! \brief 私有绑定数组，禁止共享 Array 别名改写参数顺序。 */
    Array<ConstantBinding> constants_;
};

KXC_OBJECT_DEFINE_WITH_KEY(LoweredFunctionNode, "kxc.relay.LoweredFunctionNode")

/*! \brief LoweredFunctionNode 的类型安全对象句柄。 */
class LoweredFunction : public ObjectRef {
public:
    /*! \brief 组合并校验 PrimFunc 与常量绑定的参数槽关系。 */
    LoweredFunction(tir::PrimFunc prim_func, Array<ConstantBinding> constants);
    /*! \brief 从通用对象引用恢复 lowering 产物，并验证节点类型。 */
    explicit LoweredFunction(const ObjectRef& ref);

    /*! \brief 返回常量绑定的独立 Array，保持节点内参数顺序不可变。 */
    Array<ConstantBinding> constants() const;
    /*! \brief 校验 attrs、参数槽、Buffer 和常量 payload 的完整一致性。 */
    void Validate() const;
    /*! \brief 返回经过类型检查的只读 lowering 产物节点。 */
    const LoweredFunctionNode* operator->() const;
};

/*!
 * \brief 将 Relay Function 转为 TIR，并保留 codegen 绑定常量所需的 payload。
 *
 * 流程包括类型推导、Relay-to-TE、确定性拓扑遍历和 TIR 组装。返回值中的
 * constants 与 PrimFunc 常量参数段严格一一对应，调用方不得再次扫描 Relay
 * 猜测常量顺序。
 */
LoweredFunction LowerToTIR(Function func);

}
}
