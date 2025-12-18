import onnx
import os
import collections

def to_cpp_vector(py_list):
    """Converts a python list to C++ std::vector initialization string."""
    if py_list is None:
        return "{}"
    return "{" + ", ".join(str(x) for x in py_list) + "}"

def to_cpp_string(py_str):
    """Converts python string to C++ string literal."""
    return f'"{py_str}"'

class RelayCppCodegen:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.var_map = {} # Maps ONNX name -> C++ variable name
        self.next_var_id = 0
        self.statements = []

    def get_new_var(self, prefix="v"):
        name = f"{prefix}_{self.next_var_id}"
        self.next_var_id += 1
        return name

    def generate(self):
        # Header
        self.statements.append('#include "../include/base/relay.h"')
        self.statements.append('#include "../include/base/op.h"')
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
            
            # Generate code
            shape_str = to_cpp_vector(shape)
            self.statements.append(f'    Tensor {cpp_var}_tensor({shape_str}, "{dtype}");')
            # Note: We skip loading actual data for brevity/performance in this codegen demo.
            # In a real compiler, we'd load from file or embed binary.
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
                # Default 0.0f
                self.statements.append(f'    Tensor {val_var}({{1}}, "float32");') 
                # We would need to fill it with 0, but for codegen we leave it empty/uninitialized data
                attrs_code = f"ConstantOfShapeAttrs::Create({val_var})"
        else:
             print(f"Warning: Unknown op {op_type}")
             op_name = f"unknown_{op_type}"

        self.statements.append(f'    // Node: {node.name} ({op_type})')
        self.statements.append(f'    std::vector<Expr> args_{cpp_var} = {{{inputs_str}}};')
        self.statements.append(f'    Call {cpp_var}(Op::Get("{op_name}"), args_{cpp_var}, {attrs_code});')

    def emit_constant_node(self, node, cpp_var):
        # ONNX Constant op has 'value' attribute with TensorProto
        attr = self.get_attr_raw(node, "value")
        if attr:
            t = onnx.numpy_helper.to_array(attr)
            shape = list(t.shape)
            dtype = str(t.dtype)
            shape_str = to_cpp_vector(shape)
            # For codegen, we skip data.
            self.statements.append(f'    // Node: {node.name} (Constant)')
            self.statements.append(f'    Tensor {cpp_var}_tensor({shape_str}, "{dtype}");')
            self.statements.append(f'    Constant {cpp_var}({cpp_var}_tensor);')
        else:
            self.statements.append(f'    // Node: {node.name} (Constant - No Value?)')
            self.statements.append(f'    Constant {cpp_var}(Tensor({{}}, "float32"));')

    def emit_tensor_creation(self, tensor_proto, var_name):
        t = onnx.numpy_helper.to_array(tensor_proto)
        shape = list(t.shape)
        dtype = str(t.dtype)
        shape_str = to_cpp_vector(shape)
        self.statements.append(f'    Tensor {var_name}({shape_str}, "{dtype}");')
        # In real codegen, we would load data here.

    # --- Attribute Converters ---

    def get_attr_raw(self, node, attr_name):
        for attr in node.attribute:
            if attr.name == attr_name:
                return attr.t # return TensorProto
        return None

    def get_attr(self, node, attr_name, default=None):
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
        strides = self.get_attr(node, "strides", [1, 1])
        pads = self.get_attr(node, "pads", [0, 0, 0, 0]) 
        # ONNX pads: [x1_begin, x2_begin... x1_end, x2_end...]
        # Conv2DAttrs padding: usually just [pad_h, pad_w] if symmetric or [t, l, b, r]
        # Simplified: just pass as is
        
        return f'Conv2DAttrs::Create({to_cpp_vector(strides)}, {to_cpp_vector(pads)})'

    def convert_maxpool_attrs(self, node):
        pool_size = self.get_attr(node, "kernel_shape", [2, 2])
        strides = self.get_attr(node, "strides", [1, 1])
        pads = self.get_attr(node, "pads", [0, 0, 0, 0])
        return f'MaxPool2DAttrs::Create({to_cpp_vector(pool_size)}, {to_cpp_vector(strides)}, {to_cpp_vector(pads)})'

    def convert_reshape_attrs(self, node):
        allowzero = self.get_attr(node, "allowzero", 0)
        return f'ReshapeAttrs::Create({allowzero})'

    def convert_transpose_attrs(self, node):
        perm = self.get_attr(node, "perm", [])
        return f'TransposeAttrs::Create({to_cpp_vector(perm)})'

    def convert_softmax_attrs(self, node):
        axis = self.get_attr(node, "axis", -1)
        return f'SoftmaxAttrs::Create({axis})'

    def convert_reducemean_attrs(self, node):
        axes = self.get_attr(node, "axes", [])
        keepdims = self.get_attr(node, "keepdims", 1)
        return f'ReduceMeanAttrs::Create({to_cpp_vector(axes)}, {keepdims})'

    def convert_split_attrs(self, node):
        axis = self.get_attr(node, "axis", 0)
        return f'SplitAttrs::Create({axis})'

    def convert_gather_attrs(self, node):
        axis = self.get_attr(node, "axis", 0)
        return f'GatherAttrs::Create({axis})'

    def convert_concat_attrs(self, node):
        axis = self.get_attr(node, "axis", 0)
        return f'ConcatAttrs::Create({axis})'

    def convert_cast_attrs(self, node):
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

