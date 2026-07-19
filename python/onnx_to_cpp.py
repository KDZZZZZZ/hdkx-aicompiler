"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import onnx
import os
import collections

def to_cpp_vector(py_list):
    """将 Python 序列格式化为 C++ std::vector 初始化列表。"""
    if py_list is None:
        return "{}"
    return "{" + ", ".join(str(x) for x in py_list) + "}"

def to_cpp_string(py_str):
    """将 Python 字符串格式化为 C++ 字符串字面量。"""
    return f'"{py_str}"'

class RelayCppCodegen:
    """把 ONNX 图转换为使用强类型 Device/NDArray 接口的 Relay C++ 构图程序。"""

    def __init__(self, model_path, output_path):
        """加载 ONNX 模型并初始化名称映射和输出语句缓冲区。"""
        self.model_path = model_path
        self.output_path = output_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.var_map = {} # Maps ONNX name -> C++ variable name
        self.next_var_id = 0
        self.statements = []

    def get_new_var(self, prefix="v"):
        """生成在单次 C++ 输出中唯一的变量名。"""
        name = f"{prefix}_{self.next_var_id}"
        self.next_var_id += 1
        return name

    def generate(self):
        """生成完整 C++ 构图程序，并将 initializer 放入显式 cpu:0 NDArray。"""
        # Header
        self.statements.append('#include "../include/relay/relay.h"')
        self.statements.append('#include "../include/relay/op.h"')
        self.statements.append('#include <vector>')
        self.statements.append('#include <iostream>')
        self.statements.append('#include <string>')
        self.statements.append('')
        self.statements.append('using namespace kxc;')
        self.statements.append('')
        self.statements.append('Expr build_graph() {')

        # 1. Handle Initializers (Weights) -> Constants
        self.statements.append('    // --- Initializers (Constants) ---')
        for initializer in self.graph.initializer:
            name = initializer.name
            cpp_var = self.get_new_var("const")
            self.var_map[name] = cpp_var
            
            # Extract shape
            shape = [d for d in initializer.dims]
            dtype = "float32" # Simplified, assume float32 for most
            if initializer.data_type == onnx.TensorProto.INT64:
                dtype = "int64"
            
            # 示例生成器不嵌入权重字节，以 CPU 零张量保留 initializer 的 shape 与 dtype。
            shape_str = to_cpp_vector(shape)
            self.statements.append(f'    Tensor {cpp_var}_tensor(runtime::NDArray::Zeros({shape_str}, runtime::DataTypeFromString("{dtype}"), Device::CPU()));')
            self.statements.append(f'    Constant {cpp_var}({cpp_var}_tensor);')

        # 2. Handle Inputs -> Vars
        self.statements.append('')
        self.statements.append('    // --- Inputs (Vars) ---')
        for input_node in self.graph.input:
            name = input_node.name
            if name in self.var_map:
                continue # Already handled as initializer
            
            cpp_var = self.get_new_var("input")
            self.var_map[name] = cpp_var
            self.statements.append(f'    Var {cpp_var}("{name}");')

        # 3. Handle Nodes
        self.statements.append('')
        self.statements.append('    // --- Operators ---')
        for node in self.graph.node:
            self.emit_node(node)

        # 4. Return Output
        self.statements.append('')
        self.statements.append('    // --- Output ---')
        if len(self.graph.output) > 0:
            out_name = self.graph.output[0].name
            if out_name in self.var_map:
                self.statements.append(f'    return {self.var_map[out_name]};')
            else:
                self.statements.append('    return Expr(); // Error: Output not found')
        else:
             self.statements.append('    return Expr();')

        self.statements.append('}')
        self.statements.append('')
        self.statements.append('int main() {')
        self.statements.append('    try {')
        self.statements.append('        Expr graph = build_graph();')
        self.statements.append('        std::cout << "Graph built successfully!" << std::endl;')
        self.statements.append('    } catch (const std::exception& e) {')
        self.statements.append('        std::cerr << "Error: " << e.what() << std::endl;')
        self.statements.append('        return 1;')
        self.statements.append('    }')
        self.statements.append('    return 0;')
        self.statements.append('}')

        # Write to file
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(self.statements))
        print(f"Generated C++ code to: {self.output_path}")

    def emit_node(self, node):
        """把单个 ONNX 节点映射为 Relay Call 或 Constant 构造语句。"""
        op_type = node.op_type
        inputs = []
        for inp in node.input:
            if inp in self.var_map:
                inputs.append(self.var_map[inp])
            else:
                # Handle case where input might be implicit or missing (rare in simplified ONNX)
                print(f"Warning: Input '{inp}' for node '{node.name}' ({op_type}) not found in map.")
                inputs.append("Expr()") 

        inputs_str = ", ".join(inputs)
        cpp_var = self.get_new_var("node")
        # Map output names to this new variable
        # If multiple outputs, we might need TupleGetItem, but for now assume single output or map first one
        for i, out in enumerate(node.output):
            self.var_map[out] = cpp_var # Simplified: all outputs map to the Call node (which returns Tuple if multi-out)
        
        attrs_code = "ObjectRef()" # Default empty attributes
        
        # --- Operator Mapping ---
        op_name = ""
        
        if op_type == "Conv":
            op_name = "nn_conv2d"
            attrs_code = self.convert_conv_attrs(node)
        elif op_type == "Relu":
            op_name = "nn_relu"
        elif op_type == "MaxPool":
            op_name = "nn_max_pool2d"
            attrs_code = self.convert_maxpool_attrs(node)
        elif op_type == "MatMul":
            op_name = "matmul"
        elif op_type == "Mul":
            op_name = "mul"
        elif op_type == "Add":
            op_name = "add"
        elif op_type == "Sub":
            op_name = "sub"
        elif op_type == "Pow":
            op_name = "pow"
        elif op_type == "Sqrt":
            op_name = "sqrt"
        elif op_type == "Reshape":
            op_name = "reshape"
            attrs_code = self.convert_reshape_attrs(node)
        elif op_type == "Transpose":
            op_name = "transpose"
            attrs_code = self.convert_transpose_attrs(node)
        elif op_type == "Softmax":
            op_name = "softmax"
            attrs_code = self.convert_softmax_attrs(node)
        elif op_type == "ReduceMean":
            op_name = "reduce_mean"
            attrs_code = self.convert_reducemean_attrs(node)
        elif op_type == "Split":
            op_name = "split"
            attrs_code = self.convert_split_attrs(node)
        elif op_type == "Slice":
            op_name = "slice"
            # Slice in ONNX has inputs: data, starts, ends, axes, steps
            # No attrs usually in recent opsets (inputs instead)
        elif op_type == "Squeeze":
            op_name = "squeeze"
        elif op_type == "Unsqueeze":
            op_name = "unsqueeze"
        elif op_type == "Shape":
            op_name = "shape"
        elif op_type == "Where":
            op_name = "where"
        elif op_type == "GlobalAveragePool":
             # Map GlobalAveragePool to adaptive_avg_pool2d(1) or similar if available, or custom
             # For now, maybe just map to a placeholder or skip
             print(f"Warning: Unsupported op {op_type}")
             op_name = "unknown_op"
        elif op_type == "Gemm":
             # Complex mapping, treat as matmul for structure
             op_name = "matmul"
        elif op_type == "Gather":
            op_name = "gather"
            attrs_code = self.convert_gather_attrs(node)
        elif op_type == "Concat":
            op_name = "concatenate"
            attrs_code = self.convert_concat_attrs(node)
        elif op_type == "Cast":
            op_name = "cast"
            attrs_code = self.convert_cast_attrs(node)
        elif op_type == "Constant":
            # Constant op in ONNX creates a constant tensor.
            # We can map this to a Constant node, but we need to handle it carefully.
            # If it's a node, it produces a value.
            # We'll generate a Tensor and Constant node inline.
            self.emit_constant_node(node, cpp_var)
            return # Special handling, return early
        elif op_type == "Div":
            op_name = "divide" 
            attrs_code = "DivAttrs::Create()"
        elif op_type == "Erf":
            op_name = "erf"
            attrs_code = "ErfAttrs::Create()"
        elif op_type == "Equal":
            op_name = "equal"
            attrs_code = "EqualAttrs::Create()"
        elif op_type == "Expand":
            op_name = "expand_dims"
            attrs_code = "ExpandAttrs::Create()"
        elif op_type == "ConstantOfShape":
            op_name = "constant_of_shape"
            # Special handling for attribute 'value' which is a Tensor
            # We need to emit code to create this tensor first
            val_attr = self.get_attr_raw(node, "value")
            val_var = f"{cpp_var}_val"
            if val_attr:
                self.emit_tensor_creation(val_attr, val_var)
                attrs_code = f"ConstantOfShapeAttrs::Create({val_var})"
            else:
                # 缺省 value 是零初始化的 float32 标量占位张量。
                self.statements.append(f'    Tensor {val_var}(runtime::NDArray::Zeros({{1}}, runtime::DataTypeFromString("float32"), Device::CPU()));')
                attrs_code = f"ConstantOfShapeAttrs::Create({val_var})"
        else:
             print(f"Warning: Unknown op {op_type}")
             op_name = f"unknown_{op_type}"

        self.statements.append(f'    // Node: {node.name} ({op_type})')
        self.statements.append(f'    std::vector<Expr> args_{cpp_var} = {{{inputs_str}}};')
        self.statements.append(f'    Call {cpp_var}(Op::Get("{op_name}"), args_{cpp_var}, {attrs_code});')

    def emit_constant_node(self, node, cpp_var):
        """生成 ONNX Constant 节点对应的 CPU NDArray 和 Relay Constant。"""
        # ONNX Constant op has 'value' attribute with TensorProto
        attr = self.get_attr_raw(node, "value")
        if attr:
            t = onnx.numpy_helper.to_array(attr)
            shape = list(t.shape)
            dtype = str(t.dtype)
            shape_str = to_cpp_vector(shape)
            # 该示例生成器只验证图结构，因此用显式 CPU 零张量保留 shape/dtype 而不嵌入权重字节。
            self.statements.append(f'    // Node: {node.name} (Constant)')
            self.statements.append(f'    Tensor {cpp_var}_tensor(runtime::NDArray::Zeros({shape_str}, runtime::DataTypeFromString("{dtype}"), Device::CPU()));')
            self.statements.append(f'    Constant {cpp_var}({cpp_var}_tensor);')
        else:
            self.statements.append(f'    // Node: {node.name} (Constant - No Value?)')
            self.statements.append(f'    Constant {cpp_var}(Tensor(runtime::NDArray::Zeros({{}}, runtime::DataTypeFromString("float32"), Device::CPU())));')

    def emit_tensor_creation(self, tensor_proto, var_name):
        """生成保留 TensorProto shape/dtype 的 CPU 零张量占位代码。"""
        t = onnx.numpy_helper.to_array(tensor_proto)
        shape = list(t.shape)
        dtype = str(t.dtype)
        shape_str = to_cpp_vector(shape)
        self.statements.append(f'    Tensor {var_name}(runtime::NDArray::Zeros({shape_str}, runtime::DataTypeFromString("{dtype}"), Device::CPU()));')
        # 真实模型执行必须加载权重；此脚本只生成结构检查程序。

    # --- Attribute Converters ---

    def get_attr_raw(self, node, attr_name):
        """返回指定名称的原始 TensorProto 属性，未找到时返回 None。"""
        for attr in node.attribute:
            if attr.name == attr_name:
                return attr.t # return TensorProto
        return None

    def get_attr(self, node, attr_name, default=None):
        """读取常用 ONNX 标量或列表属性，并在缺失时返回默认值。"""
        for attr in node.attribute:
            if attr.name == attr_name:
                if attr.type == onnx.AttributeProto.INT:
                    return attr.i
                elif attr.type == onnx.AttributeProto.INTS:
                    return list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    return attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    return attr.s.decode('utf-8')
        return default

    def convert_conv_attrs(self, node):
        """将 ONNX Conv 属性转换为 Conv2DAttrs 构造表达式。"""
        strides = self.get_attr(node, "strides", [1, 1])
        pads = self.get_attr(node, "pads", [0, 0, 0, 0]) 
        # ONNX pads: [x1_begin, x2_begin... x1_end, x2_end...]
        # Conv2DAttrs padding: usually just [pad_h, pad_w] if symmetric or [t, l, b, r]
        # Simplified: just pass as is
        
        return f'Conv2DAttrs::Create({to_cpp_vector(strides)}, {to_cpp_vector(pads)})'

    def convert_maxpool_attrs(self, node):
        """将 ONNX MaxPool 属性转换为 MaxPool2DAttrs 构造表达式。"""
        pool_size = self.get_attr(node, "kernel_shape", [2, 2])
        strides = self.get_attr(node, "strides", [1, 1])
        pads = self.get_attr(node, "pads", [0, 0, 0, 0])
        return f'MaxPool2DAttrs::Create({to_cpp_vector(pool_size)}, {to_cpp_vector(strides)}, {to_cpp_vector(pads)})'

    def convert_reshape_attrs(self, node):
        """将 ONNX Reshape 的 allowzero 属性转换为构造表达式。"""
        allowzero = self.get_attr(node, "allowzero", 0)
        return f'ReshapeAttrs::Create({allowzero})'

    def convert_transpose_attrs(self, node):
        """将 ONNX Transpose 的轴排列转换为构造表达式。"""
        perm = self.get_attr(node, "perm", [])
        return f'TransposeAttrs::Create({to_cpp_vector(perm)})'

    def convert_softmax_attrs(self, node):
        """将 ONNX Softmax 的归一化轴转换为构造表达式。"""
        axis = self.get_attr(node, "axis", -1)
        return f'SoftmaxAttrs::Create({axis})'

    def convert_reducemean_attrs(self, node):
        """将 ONNX ReduceMean 的轴和维度保留规则转换为构造表达式。"""
        axes = self.get_attr(node, "axes", [])
        keepdims = self.get_attr(node, "keepdims", 1)
        return f'ReduceMeanAttrs::Create({to_cpp_vector(axes)}, {keepdims})'

    def convert_split_attrs(self, node):
        """将 ONNX Split 的拆分轴转换为构造表达式。"""
        axis = self.get_attr(node, "axis", 0)
        return f'SplitAttrs::Create({axis})'

    def convert_gather_attrs(self, node):
        """将 ONNX Gather 的索引轴转换为构造表达式。"""
        axis = self.get_attr(node, "axis", 0)
        return f'GatherAttrs::Create({axis})'

    def convert_concat_attrs(self, node):
        """将 ONNX Concat 的拼接轴转换为构造表达式。"""
        axis = self.get_attr(node, "axis", 0)
        return f'ConcatAttrs::Create({axis})'

    def convert_cast_attrs(self, node):
        """将 ONNX Cast 的目标类型编号转换为构造表达式。"""
        to_type = self.get_attr(node, "to", 1) # Default float (1)
        return f'CastAttrs::Create({to_type})'


if __name__ == "__main__":
    # Assuming model.onnx is on Desktop
    desktop_path = r"C:\Users\Administrator\Desktop\hdkx-aicompiler-1"
    model_path = os.path.join(desktop_path, "resnet18.onnx")
    output_path = os.path.join(desktop_path, "hdkx-aicompiler-1", "test", "generated_network.cpp")
    
    if os.path.exists(model_path):
        codegen = RelayCppCodegen(model_path, output_path)
        codegen.generate()
    else:
        print(f"Error: {model_path} not found.")

