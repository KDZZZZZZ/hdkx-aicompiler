"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import onnx
import onnx.numpy_helper
import os
import argparse
import sys
from onnx import TensorProto, helper

def get_data_type_name(data_type):
    """Convert ONNX data type ID to string name."""
    type_mapping = {
        TensorProto.FLOAT: "FLOAT",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT8: "INT8",
        TensorProto.UINT16: "UINT16",
        TensorProto.INT16: "INT16",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.STRING: "STRING",
        TensorProto.BOOL: "BOOL",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.DOUBLE: "DOUBLE",
        TensorProto.UINT32: "UINT32",
        TensorProto.UINT64: "UINT64",
    }
    return type_mapping.get(data_type, f"UNKNOWN({data_type})")

def get_tensor_shape_str(tensor_info):
    """Extract shape string from ValueInfoProto."""
    if not tensor_info.type.tensor_type.HasField("shape"):
        return "[]" # Scalar or unknown
    
    dims = []
    for d in tensor_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return f"[{', '.join(dims)}]"

def get_attribute_value(attr):
    """Extract value from AttributeProto."""
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8', errors='ignore')
    elif attr.type == onnx.AttributeProto.TENSOR:
        return f"<Tensor: {attr.t.name}>"
    elif attr.type == onnx.AttributeProto.GRAPH:
        return "<Graph>"
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8', errors='ignore') for s in attr.strings]
    else:
        return "<Complex/Unknown>"

def generate_dot_graph(graph, output_path):
    """Generate a Graphviz DOT file for visualization."""
    dot_lines = []
    dot_lines.append("digraph G {")
    dot_lines.append("    rankdir=TB;")
    dot_lines.append("    node [shape=box, style=filled, color=lightblue];")
    
    # Map to track connections
    tensor_producers = {}
    
    # Graph inputs produce tensors
    for inp in graph.input:
        clean_name = inp.name.replace(":", "_").replace(".", "_").replace("/", "_")
        # Use explicit node for Input
        dot_lines.append(f'    "{clean_name}" [shape=oval, style=filled, color=lightgreen, label="{inp.name}"];')
        tensor_producers[inp.name] = clean_name
        
    # Initializers produce tensors (constants)
    for init in graph.initializer:
        clean_name = init.name.replace(":", "_").replace(".", "_").replace("/", "_")
        # If not already in inputs (sometimes initializers are also listed in inputs)
        if init.name not in tensor_producers:
            dot_lines.append(f'    "{clean_name}" [shape=diamond, style=filled, color=lightgrey, label="{init.name}\\n(Const)"];')
            tensor_producers[init.name] = clean_name

    # Nodes produce tensors
    for i, node in enumerate(graph.node):
        node_id = f"node_{i}"
        op_type = node.op_type
        name = node.name if node.name else f"{op_type}_{i}"
        label = f"{op_type}\\n{name}"
        dot_lines.append(f'    "{node_id}" [label="{label}", color=lightblue];')
        
        # Edges from inputs to this node
        for inp_name in node.input:
            if not inp_name: continue
            if inp_name in tensor_producers:
                producer_id = tensor_producers[inp_name]
                dot_lines.append(f'    "{producer_id}" -> "{node_id}" [label="{inp_name}", fontsize=10];')
            else:
                # Input coming from unknown source (maybe missing in graph.input or optional)
                clean_inp = inp_name.replace(":", "_").replace(".", "_").replace("/", "_")
                # Check if we already created a dummy for this
                # Ideally we shouldn't duplicate dummies, but for now it's okay
                # dot_lines.append(f'    "{clean_inp}" [shape=point, label="?"];')
                # dot_lines.append(f'    "{clean_inp}" -> "{node_id}" [label="{inp_name}"];')
                # Actually, let's just draw the edge from a new node if it doesn't exist
                # But simpler to assume it exists if we processed topological order.
                # If not, just create a node for it.
                dot_lines.append(f'    "{clean_inp}" [shape=plaintext, label="{inp_name}"];')
                dot_lines.append(f'    "{clean_inp}" -> "{node_id}" [color=red];')
                tensor_producers[inp_name] = clean_inp

        # Register outputs of this node
        for out_name in node.output:
            if not out_name: continue
            tensor_producers[out_name] = node_id

    # Graph outputs
    for out in graph.output:
        clean_name = out.name.replace(":", "_").replace(".", "_").replace("/", "_")
        out_node_id = f"out_{clean_name}"
        dot_lines.append(f'    "{out_node_id}" [shape=oval, style=filled, color=orange, label="{out.name}"];')
        
        if out.name in tensor_producers:
            producer_id = tensor_producers[out.name]
            dot_lines.append(f'    "{producer_id}" -> "{out_node_id}" [label="{out.name}"];')

    dot_lines.append("}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))
    print(f"DOT graph saved to {output_path}")

def analyze_model(model_path, output_dir):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "model_report.md")
    dot_path = os.path.join(output_dir, "model_graph.dot")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# ONNX Model Report\n\n")
        f.write(f"- **Model Path:** `{model_path}`\n")
        f.write(f"- **IR Version:** {model.ir_version}\n")
        f.write(f"- **Opset Import:** {[f'{op.domain}:{op.version}' for op in model.opset_import]}\n")
        f.write(f"- **Producer:** {model.producer_name} ({model.producer_version})\n\n")
        
        # Inputs
        f.write("## Inputs\n")
        f.write("| Name | Shape | Type |\n")
        f.write("|------|-------|------|\n")
        for inp in graph.input:
            shape = get_tensor_shape_str(inp)
            dtype = get_data_type_name(inp.type.tensor_type.elem_type)
            f.write(f"| `{inp.name}` | `{shape}` | {dtype} |\n")
        f.write("\n")
        
        # Outputs
        f.write("## Outputs\n")
        f.write("| Name | Shape | Type |\n")
        f.write("|------|-------|------|\n")
        for out in graph.output:
            shape = get_tensor_shape_str(out)
            dtype = get_data_type_name(out.type.tensor_type.elem_type)
            f.write(f"| `{out.name}` | `{shape}` | {dtype} |\n")
        f.write("\n")
        
        # Initializers (Weights)
        f.write("## Initializers (Weights)\n")
        f.write(f"Total Initializers: {len(graph.initializer)}\n\n")
        if len(graph.initializer) > 0:
            f.write("| Name | Shape | Type | Stats |\n")
            f.write("|------|-------|------|-------|\n")
            for init in graph.initializer:
                shape = f"[{', '.join(str(d) for d in init.dims)}]"
                dtype = get_data_type_name(init.data_type)
                
                # Calculate simple stats if small enough
                stats = "-"
                try:
                    if init.data_type == TensorProto.FLOAT: # Only for float
                        arr = onnx.numpy_helper.to_array(init)
                        if arr.size > 0:
                            stats = f"min:{arr.min():.2f}, max:{arr.max():.2f}"
                except:
                    pass
                    
                f.write(f"| `{init.name}` | `{shape}` | {dtype} | {stats} |\n")
        f.write("\n")

        # Nodes
        f.write("## Computation Graph (Nodes)\n")
        f.write(f"Total Nodes: {len(graph.node)}\n\n")
        
        for i, node in enumerate(graph.node):
            f.write(f"### Node {i}: `{node.op_type}`\n")
            if node.name:
                f.write(f"- **Name:** {node.name}\n")
            
            f.write("- **Inputs:**\n")
            for inp in node.input:
                f.write(f"  - `{inp}`\n")
            
            f.write("- **Outputs:**\n")
            for out in node.output:
                f.write(f"  - `{out}`\n")
                
            if len(node.attribute) > 0:
                f.write("- **Attributes:**\n")
                for attr in node.attribute:
                    val = get_attribute_value(attr)
                    f.write(f"  - `{attr.name}`: {val}\n")
            f.write("\n")

    print(f"Report saved to {report_path}")
    
    # Generate DOT
    generate_dot_graph(graph, dot_path)

if __name__ == "__main__":
    # Default path from user script
    default_model = r"C:\Users\Administrator\Desktop\hdkx-aicompiler-1\resnet18.onnx"
    
    parser = argparse.ArgumentParser(description="Analyze ONNX Model")
    parser.add_argument("--model", type=str, default=default_model, help="Path to ONNX model")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    analyze_model(args.model, args.output)
