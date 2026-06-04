"""
职责简介：
- 提供 ONNX 解析、模型报告和 C++ Relay 构图辅助脚本。
"""

import os
import subprocess
import sys

def parse_and_visualize_onnx(model_path):
    """
    解析并可视化 ONNX 模型的全部信息。
    
    Args:
        model_path (str): ONNX 模型的路径。
    """
    
    # 确保安装了必要的库
    try:
        import onnx
        import graphviz
    except ImportError:
        print("安装 ONNX 和 Graphviz 库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "graphviz"])
        import onnx
        import graphviz

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在于路径 '{model_path}'")
        return

    print(f"--- 正在解析 ONNX 模型: {model_path} ---")

    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"错误: 无法加载 ONNX 模型。请检查模型文件是否有效。详细信息: {e}")
        return

    graph = model.graph

    # --- 1. 模型基本信息 ---
    print("\n--- 模型基本信息 ---")
    print(f"IR 版本: {model.ir_version}")
    print(f"生产者名称: {model.producer_name}")
    print(f"生产者版本: {model.producer_version}")
    print(f"模型版本: {model.model_version}")
    print(f"文档字符串: {model.doc_string}")
    
    print("\n--- 操作集导入 (Opset Imports) ---")
    for opset_id, opset_import in enumerate(model.opset_import):
        print(f"  Opset {opset_id}: Domain='{opset_import.domain}', Version={opset_import.version}")

    # --- 2. 模型输入 ---
    print("\n--- 模型输入 (Inputs) ---")
    for input_idx, model_input in enumerate(graph.input):
        name = model_input.name
        data_type = onnx.TensorProto.DataType.Name(model_input.type.tensor_type.elem_type)
        shape = [d.dim_value if d.dim_value != 0 else '?' for d in model_input.type.tensor_type.shape.dim]
        print(f"  输入 {input_idx}: 名称='{name}', 类型='{data_type}', 形状='{shape}'")

    # --- 3. 模型输出 ---
    print("\n--- 模型输出 (Outputs) ---")
    for output_idx, model_output in enumerate(graph.output):
        name = model_output.name
        data_type = onnx.TensorProto.DataType.Name(model_output.type.tensor_type.elem_type)
        shape = [d.dim_value if d.dim_value != 0 else '?' for d in model_output.type.tensor_type.shape.dim]
        print(f"  输出 {output_idx}: 名称='{name}', 类型='{data_type}', 形状='{shape}'")

    # --- 4. 初始化器 (权重、偏置等) ---
    print("\n--- 初始化器 (Initializers - Weights, Biases etc.) ---")
    if not graph.initializer:
        print("  无初始化器。")
    for initializer_idx, initializer in enumerate(graph.initializer):
        name = initializer.name
        data_type = onnx.TensorProto.DataType.Name(initializer.data_type)
        shape = initializer.dims
        print(f"  初始化器 {initializer_idx}: 名称='{name}', 类型='{data_type}', 形状='{shape}'")

    # --- 5. 模型节点 (操作符) ---
    print("\n--- 模型节点 (Nodes - Operators) ---")
    for node_idx, node in enumerate(graph.node):
        print(f"  节点 {node_idx}: 操作符='{node.op_type}', 名称='{node.name}'")
        print(f"    输入: {node.input}")
        print(f"    输出: {node.output}")
        if node.attribute:
            print("    属性:")
            for attr in node.attribute:
                attr_type_name = onnx.AttributeProto.AttributeType.Name(attr.type)
                attr_value = ""
                # 根据属性类型打印值
                if attr.type == onnx.AttributeProto.FLOAT:
                    attr_value = attr.f
                elif attr.type == onnx.AttributeProto.INT:
                    attr_value = attr.i
                elif attr.type == onnx.AttributeProto.STRING:
                    attr_value = attr.s.decode('utf-8') # Protobuf string is bytes
                elif attr.type == onnx.AttributeProto.TENSOR:
                    attr_value = f"Tensor (shape={attr.t.dims}, dtype={onnx.TensorProto.DataType.Name(attr.t.data_type)})"
                elif attr.type == onnx.AttributeProto.GRAPH:
                    attr_value = "Graph (subgraph)" # 嵌套图
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attr_value = list(attr.floats)
                elif attr.type == onnx.AttributeProto.INTS:
                    attr_value = list(attr.ints)
                elif attr.type == onnx.AttributeProto.STRINGS:
                    attr_value = [s.decode('utf-8') for s in attr.strings]
                else:
                    attr_value = f"<Unsupported Type: {attr_type_name}>"
                print(f"      - {attr.name} ({attr_type_name}): {attr_value}")

    # --- 6. 模型图可视化 ---
    print("\n--- 正在生成模型可视化图 ---")
    dot = graphviz.Digraph(comment=graph.name, graph_attr={'rankdir': 'TB', 'splines': 'spline'}, node_attr={'shape': 'box', 'style': 'filled', 'color': 'lightblue'})

    # 1. 添加输入节点
    for model_input in graph.input:
        is_initializer = False
        for init in graph.initializer:
            if init.name == model_input.name:
                is_initializer = True
                break
        if not is_initializer: # 只有非初始化的输入才是真正的图输入
            shape = [str(d.dim_value) if d.dim_value != 0 else '?' for d in model_input.type.tensor_type.shape.dim]
            dot.node(model_input.name, label=f"Input: {model_input.name}\nShape: {shape}", shape='oval', style='filled', color='lightgreen')

    # 2. 添加初始化器 (权重)
    for initializer in graph.initializer:
        shape = list(initializer.dims)
        dot.node(initializer.name, label=f"Initializer: {initializer.name}\nShape: {shape}", shape='box', style='filled', color='yellow')

    # 3. 添加操作符节点
    for node in graph.node:
        dot.node(node.name if node.name else node.op_type + "_" + str(hash(node)), label=f"{node.op_type}\n({node.name})", shape='box', style='filled', color='lightblue')
        
        # 添加输入边
        for input_name in node.input:
            dot.edge(input_name, node.name if node.name else node.op_type + "_" + str(hash(node)), label=input_name)

        # 添加输出边
        for output_name in node.output:
            dot.edge(node.name if node.name else node.op_type + "_" + str(hash(node)), output_name, label=output_name)

    # 4. 添加模型输出
    for model_output in graph.output:
        shape = [str(d.dim_value) if d.dim_value != 0 else '?' for d in model_output.type.tensor_type.shape.dim]
        dot.node(model_output.name + "_output_node", label=f"Output: {model_output.name}\nShape: {shape}", shape='oval', style='filled', color='lightgreen')
        # 连接图的实际输出到这个虚拟输出节点
        dot.edge(model_output.name, model_output.name + "_output_node", label=model_output.name)

    output_filename = os.path.splitext(os.path.basename(model_path))[0] + "_onnx_graph"
    dot.render(output_filename, view=False, format='png') # view=False prevents opening in default viewer
    print(f"模型可视化图已保存为 '{output_filename}.png' 和 '{output_filename}.dot'")
    print("--- 解析与可视化完成 ---")

# --- 调用函数 ---
# 替换 'your_model.onnx' 为你的 ONNX 模型文件路径
# 例如: parse_and_visualize_onnx("path/to/your/model.onnx")
# 或者下载一个示例模型，如 MobileNetV2:
# wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
# parse_and_visualize_onnx("mobilenetv2-7.onnx")

# 在这里调用解析函数
# 如果你没有ONNX模型，可以运行以下代码创建一个简单的示例模型
# import onnx
# from onnx import helper, TensorProto
# from onnx import numpy_helper
# import numpy as np

# # 定义图的输入
# X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 3, 3])
# Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 3, 3])

# # 定义权重 (初始化器)
# W = numpy_helper.from_array(np.random.rand(2, 2, 3, 3).astype(np.float32), name='W')
# B = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='B')

# # 定义节点
# node1 = helper.make_node(
#     'Conv',
#     inputs=['X', 'W', 'B'],
#     outputs=['conv_output'],
#     kernel_shape=[3, 3],
#     pads=[1, 1, 1, 1],
#     strides=[1, 1],
#     group=1,
#     dilations=[1, 1]
# )

# node2 = helper.make_node(
#     'Relu',
#     inputs=['conv_output'],
#     outputs=['relu_output']
# )

# node3 = helper.make_node(
#     'Add',
#     inputs=['relu_output', 'Y'],
#     outputs=['add_output']
# )

# # 定义图的输出
# Z = helper.make_tensor_value_info('add_output', TensorProto.FLOAT, [1, 2, 3, 3])

# # 创建图
# graph_def = helper.make_graph(
#     [node1, node2, node3],
#     'simple_graph',
#     [X, Y],
#     [Z],
#     [W, B] # 初始化器
# )

# # 创建模型
# model_def = helper.make_model(graph_def, producer_name='my_producer')
# model_def.opset_import[0].version = 13 # 设置Opset版本

# # 验证模型
# onnx.checker.check_model(model_def)

# # 保存模型
# simple_onnx_path = "simple_example.onnx"
# onnx.save(model_def, simple_onnx_path)

# # 调用解析和可视化函数
# parse_and_visualize_onnx(simple_onnx_path)
