import torch
import torchvision
import torch.onnx

def export_resnet18():
    # 1. 加载预训练的ResNet-18模型
    # 如果不需要预训练权重（只关心结构），可以将 weights 设置为 None
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # 切换到推理模式，这很重要（固定BatchNorm和Dropout）

    # 2. 创建一个Dummy Input（伪造输入）
    # ONNX 导出通常基于 Tracing（追踪），需要一个示例输入来执行一次前向传播
    # 格式: [Batch_Size, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. 导出为 ONNX
    onnx_path = "resnet18.onnx"
    torch.onnx.export(
        model,                      # 待导出的模型
        dummy_input,                # 示例输入
        onnx_path,                  # 输出路径
        export_params=True,         # 是否在模型文件中存储权重
        opset_version=11,           # Opset版本，11是目前兼容性最好的版本之一
        do_constant_folding=True,   # 是否执行常量折叠优化
        input_names=['input'],      # 输入节点的名称
        output_names=['output'],    # 输出节点的名称
        dynamic_axes={              # 定义动态轴（非常重要，否则Batch Size会被固定为1）
            'input': {0: 'batch_size'},  # 输入的第0维是动态的
            'output': {0: 'batch_size'}  # 输出的第0维也是动态的
        }
    )
    print(f"ResNet-18 已导出至: {onnx_path}")

if __name__ == "__main__":
    export_resnet18()