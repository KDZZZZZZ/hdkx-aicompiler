import onnx
model_path = "C:/Users/Administrator/Desktop/hdkx-aicompiler/test/test-onnx/model.onnx"
model = onnx.load(model_path)
# 获取模型的输入信息
information = ""
input_info = model.graph.input
information += str("INPUT INFO:\n")
for input_tensor in input_info:
    information += str(input_tensor)

# 获取模型的输出信息
output_info = model.graph.output
information += str("OUTPUT INFO:\n")
for output_tensor in output_info:
    information += str(output_tensor)

# 获取模型的节点信息
nodes = model.graph.node
information += str("NODE INFO:\n")
for node in nodes:
    information += str(node)+"\n"

with open("model_info.txt", "w") as file:
    file.write(str(information))