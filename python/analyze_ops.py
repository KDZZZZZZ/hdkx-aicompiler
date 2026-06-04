"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import onnx
import collections
import os
from onnx import helper, TensorProto

def get_attr_value(attr):
    """Helper to get attribute value based on type."""
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode('utf-8')
    elif attr.type == onnx.AttributeProto.TENSOR:
        return f"Tensor({attr.t.dims})"
    elif attr.type == onnx.AttributeProto.GRAPH:
        return "Graph"
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings]
    else:
        return "Unknown"

def analyze_onnx_operators(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    
    # Store operator info: { op_type: { "count": n, "inputs": set(num), "outputs": set(num), "attrs": { name: example_value } } }
    op_stats = collections.defaultdict(lambda: {
        "count": 0,
        "inputs": set(),
        "outputs": set(),
        "attrs": {}
    })

    for node in graph.node:
        stats = op_stats[node.op_type]
        stats["count"] += 1
        stats["inputs"].add(len(node.input))
        stats["outputs"].add(len(node.output))
        
        for attr in node.attribute:
            if attr.name not in stats["attrs"]:
                # Store type and example value
                stats["attrs"][attr.name] = get_attr_value(attr)

    print("\n" + "="*60)
    print(f"Analysis Result for {os.path.basename(model_path)}")
    print("="*60 + "\n")
    
    for op_type in sorted(op_stats.keys()):
        info = op_stats[op_type]
        print(f"Operator: {op_type}")
        print(f"  Count: {info['count']}")
        print(f"  Inputs: {sorted(list(info['inputs']))}")
        print(f"  Outputs: {sorted(list(info['outputs']))}")
        
        if info["attrs"]:
            print("  Attributes:")
            for attr_name, attr_val in info["attrs"].items():
                print(f"    - {attr_name}: {attr_val} (Example)")
        else:
            print("  Attributes: None")
        print("-" * 40)

if __name__ == "__main__":
    # Use a default model path or create a dummy one if needed
    model_path = r"C:\Users\Administrator\Desktop\model.onnx"
    
    # Fallback to creating the dummy model from the user's script if file doesn't exist
    if not os.path.exists(model_path):
        print("Model not found, creating simple example...")
        # Create dummy model logic from previous context
        import numpy as np
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 3, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 3])
        W = onnx.numpy_helper.from_array(np.random.rand(2, 2, 3, 3).astype(np.float32), name='W')
        B = onnx.numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='B')
        
        node1 = helper.make_node('Conv', ['X', 'W', 'B'], ['conv_out'], kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1])
        node2 = helper.make_node('Relu', ['conv_out'], ['relu_out'])
        node3 = helper.make_node('Add', ['relu_out', 'Y'], ['out'])
        
        graph = helper.make_graph([node1, node2, node3], 'test', [X, Y], [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1,2,3,3])], [W, B])
        model = helper.make_model(graph)
        model_path = "simple_example.onnx"
        onnx.save(model, model_path)

    analyze_onnx_operators(model_path)
