# KXC Relay 算子：从注册到实现端到端教程

本文档将引导您完成在 KXC 编译器中添加一个新算子的完整流程，涵盖从前端注册到后端 TE (Tensor Expression) 计算逻辑实现的各个环节。

## 流程概览

在 KXC 中，添加一个新算子通常涉及以下三个主要步骤：

1.  **定义属性 (Attributes)**: 在 C++ 头文件中定义算子的配置参数结构体。
2.  **前端注册 (Registration)**: 使用宏注册算子接口，声明输入输出和属性类型。
3.  **后端实现 (Implementation)**: 使用 TE/TOPI 定义算子的数学计算逻辑。

---

## 第一步：定义属性 (Attributes)

如果您的算子包含除 Tensor 数据以外的参数（例如卷积的 `stride`，或者 Reduce 操作的 `axis`），您需要在 `include/base/op.h` 中定义一个属性结构体。

**文件**: `include/base/op.h`

**示例 (MyOpAttrs)**:

```cpp
// 1. 定义属性数据节点
class MyOpAttrsNode : public BaseAttrsNode {
public:
    int axis;
    float scale;

    void VisitAttrs(AttrVisitor* v) {
        v->Visit("axis", &axis);
        v->Visit("scale", &scale);
    }
    
    // 分配唯一的 TypeIndex (请确保不冲突，参考现有代码)
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 99; }
};
```
// 2. 定义属性引用类 (RefWrapper)
class MyOpAttrs : public Attrs {
public:
    using Attrs::Attrs;
    
    // 静态工厂方法
    static MyOpAttrs Create(int axis, float scale = 1.0f) {
        MyOpAttrsNode* node = new MyOpAttrsNode();
        node->axis = axis;
        node->scale = scale;
        
        MyOpAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    
    // 指针访问重载
    const MyOpAttrsNode* operator->() const { 
        return static_cast<const MyOpAttrsNode*>(object_); 
    }
};
```

---

## 第二步：前端注册 (Registration)

接下来，告知编译器前端该算子的存在、输入参数以及它使用的属性结构体。

**文件**: `src/relay/op/tensor/my_ops.cc` (或现有分类文件)

```cpp
#include "relay/op_macros.h"
#include "base/relay.h"

namespace kxc {
namespace relay {

KXC_REGISTER_OP(my_op)
    .describe(R"doc(My custom operator description.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input tensor")
    // 绑定刚才定义的属性结构体名
    .set_attr<std::string>("TAttrs", "MyOpAttrs");

} // namespace relay
} // namespace kxc
```

详细注册指南请参考: [OPERATOR_REGISTRATION_GUIDE.md](./OPERATOR_REGISTRATION_GUIDE.md)

---

## 第三步：后端实现 (TE/TOPI)

这是核心的计算实现部分。KXC 使用 **TE (Tensor Expression)** 来描述计算逻辑，类似于 TVM 的 TOPI (Tensor Operator Inventory)。

您需要在 `include/te/topi.h` (或新建头文件) 中实现计算函数。

**文件**: `include/te/topi.h`

### 3.1 简单元素级操作 (Element-wise)

如果算子是简单的逐元素计算（如 Add, Relu）：

```cpp
namespace kxc {
namespace topi {

inline Tensor my_op_compute(const Tensor& input, float scale, std::string name = "T_my_op") {
    // 使用 compute 函数定义计算
    // shape: 输出张量的形状 (通常与输入相同)
    // lambda: 定义每个坐标点 (indices) 的计算逻辑
    return compute(input->shape, [&](const std::vector<Var>& indices) {
        // 使用 input(indices) 访问输入值
        // 返回 PrimExpr (基础表达式)
        return input(indices) * scale;
    }, name);
}

} // namespace topi
} // namespace kxc
```

### 3.2 归约操作 (Reduction)

如果算子涉及归约（如 MatMul, Conv2D, Sum）：

1.  使用 `reduce_axis` 定义归约维度。
2.  使用 `sum` (或其他聚合函数) 进行计算。

**示例：矩阵乘法 (MatMul)**

```cpp
inline Tensor matmul(const Tensor& A, const Tensor& B, std::string name = "T_matmul") {
    // A: [M, K], B: [K, N]
    auto M = A->shape[0];
    auto N = B->shape[1];
    auto K = A->shape[1];
    
    // 定义归约轴 k，范围 [0, K)
    IterVar k = reduce_axis(0, K, "k");
    
    // 输出形状 [M, N]
    return compute({M, N}, [&](const std::vector<Var>& indices) {
        Var i = indices[0];
        Var j = indices[1];
        
        // 计算公式: sum(A[i, k] * B[k, j]) over k
        return sum(A(i, k) * B(k, j), {k});
    }, name);
}
```

---

## 第四步：验证 (Testing)

最后，编写测试用例验证算子能否正确构建计算图。

**文件**: `test/test_my_op.cpp`

```cpp
#include "te/topi.h"
#include <iostream>
#include <cassert>

using namespace kxc;
using namespace kxc::te;
using namespace kxc::topi;

int main() {
    // 1. 创建占位符输入
    Var n("n");
    Tensor A = placeholder({n}, DataType::Float(32), "A");
    
    // 2. 调用 TOPI 实现
    Tensor B = my_op_compute(A, 2.0f);
    
    // 3. 验证
    assert(B->op.defined());
    // 进一步可以检查 B->op->body 等表达式结构
    
    std::cout << "MyOp test passed!" << std::endl;
    return 0;
}
```

## 总结

通过以上步骤，您已经成功地在 KXC 中添加了一个全栈支持的算子：
1.  **Attr**: 定义了配置。
2.  **Relay Op**: 注册了前端接口。
3.  **TOPI**: 实现了后端计算逻辑。
4.  **Test**: 验证了正确性。
