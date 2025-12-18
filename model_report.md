# ONNX Model Report

- **Model Path:** `C:\Users\Administrator\Desktop\hdkx-aicompiler-1\resnet18.onnx`
- **IR Version:** 6
- **Opset Import:** [':11']
- **Producer:** pytorch (2.5.1)

## Inputs
| Name | Shape | Type |
|------|-------|------|
| `input` | `[batch_size, 3, 224, 224]` | FLOAT |

## Outputs
| Name | Shape | Type |
|------|-------|------|
| `output` | `[batch_size, 1000]` | FLOAT |

## Initializers (Weights)
Total Initializers: 42

| Name | Shape | Type | Stats |
|------|-------|------|-------|
| `fc.weight` | `[1000, 512]` | FLOAT | min:-0.28, max:0.72 |
| `fc.bias` | `[1000]` | FLOAT | min:-0.05, max:0.06 |
| `onnx::Conv_193` | `[64, 3, 7, 7]` | FLOAT | min:-0.31, max:0.39 |
| `onnx::Conv_194` | `[64]` | FLOAT | min:-0.64, max:0.69 |
| `onnx::Conv_196` | `[64, 64, 3, 3]` | FLOAT | min:-0.37, max:0.33 |
| `onnx::Conv_197` | `[64]` | FLOAT | min:-0.82, max:1.10 |
| `onnx::Conv_199` | `[64, 64, 3, 3]` | FLOAT | min:-0.77, max:0.53 |
| `onnx::Conv_200` | `[64]` | FLOAT | min:-1.25, max:1.79 |
| `onnx::Conv_202` | `[64, 64, 3, 3]` | FLOAT | min:-0.27, max:0.28 |
| `onnx::Conv_203` | `[64]` | FLOAT | min:-1.02, max:1.21 |
| `onnx::Conv_205` | `[64, 64, 3, 3]` | FLOAT | min:-1.05, max:0.76 |
| `onnx::Conv_206` | `[64]` | FLOAT | min:-1.17, max:1.08 |
| `onnx::Conv_208` | `[128, 64, 3, 3]` | FLOAT | min:-0.14, max:0.21 |
| `onnx::Conv_209` | `[128]` | FLOAT | min:-0.44, max:0.74 |
| `onnx::Conv_211` | `[128, 128, 3, 3]` | FLOAT | min:-0.49, max:0.72 |
| `onnx::Conv_212` | `[128]` | FLOAT | min:-0.74, max:1.45 |
| `onnx::Conv_214` | `[128, 64, 1, 1]` | FLOAT | min:-0.57, max:0.69 |
| `onnx::Conv_215` | `[128]` | FLOAT | min:-1.11, max:0.97 |
| `onnx::Conv_217` | `[128, 128, 3, 3]` | FLOAT | min:-0.26, max:0.31 |
| `onnx::Conv_218` | `[128]` | FLOAT | min:-0.83, max:0.62 |
| `onnx::Conv_220` | `[128, 128, 3, 3]` | FLOAT | min:-0.88, max:0.61 |
| `onnx::Conv_221` | `[128]` | FLOAT | min:-1.14, max:1.16 |
| `onnx::Conv_223` | `[256, 128, 3, 3]` | FLOAT | min:-0.19, max:0.24 |
| `onnx::Conv_224` | `[256]` | FLOAT | min:-0.69, max:0.86 |
| `onnx::Conv_226` | `[256, 256, 3, 3]` | FLOAT | min:-0.38, max:0.56 |
| `onnx::Conv_227` | `[256]` | FLOAT | min:-0.37, max:0.61 |
| `onnx::Conv_229` | `[256, 128, 1, 1]` | FLOAT | min:-0.41, max:0.32 |
| `onnx::Conv_230` | `[256]` | FLOAT | min:-0.34, max:0.24 |
| `onnx::Conv_232` | `[256, 256, 3, 3]` | FLOAT | min:-0.22, max:0.27 |
| `onnx::Conv_233` | `[256]` | FLOAT | min:-0.81, max:0.74 |
| `onnx::Conv_235` | `[256, 256, 3, 3]` | FLOAT | min:-0.97, max:0.73 |
| `onnx::Conv_236` | `[256]` | FLOAT | min:-1.01, max:1.06 |
| `onnx::Conv_238` | `[512, 256, 3, 3]` | FLOAT | min:-0.15, max:0.30 |
| `onnx::Conv_239` | `[512]` | FLOAT | min:-0.48, max:0.56 |
| `onnx::Conv_241` | `[512, 512, 3, 3]` | FLOAT | min:-0.69, max:1.14 |
| `onnx::Conv_242` | `[512]` | FLOAT | min:-1.13, max:0.75 |
| `onnx::Conv_244` | `[512, 256, 1, 1]` | FLOAT | min:-0.83, max:1.00 |
| `onnx::Conv_245` | `[512]` | FLOAT | min:-0.76, max:0.26 |
| `onnx::Conv_247` | `[512, 512, 3, 3]` | FLOAT | min:-0.18, max:0.29 |
| `onnx::Conv_248` | `[512]` | FLOAT | min:-0.85, max:0.51 |
| `onnx::Conv_250` | `[512, 512, 3, 3]` | FLOAT | min:-2.33, max:3.65 |
| `onnx::Conv_251` | `[512]` | FLOAT | min:-0.70, max:2.23 |

## Computation Graph (Nodes)
Total Nodes: 49

### Node 0: `Conv`
- **Name:** /conv1/Conv
- **Inputs:**
  - `input`
  - `onnx::Conv_193`
  - `onnx::Conv_194`
- **Outputs:**
  - `/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [7, 7]
  - `pads`: [3, 3, 3, 3]
  - `strides`: [2, 2]

### Node 1: `Relu`
- **Name:** /relu/Relu
- **Inputs:**
  - `/conv1/Conv_output_0`
- **Outputs:**
  - `/relu/Relu_output_0`

### Node 2: `MaxPool`
- **Name:** /maxpool/MaxPool
- **Inputs:**
  - `/relu/Relu_output_0`
- **Outputs:**
  - `/maxpool/MaxPool_output_0`
- **Attributes:**
  - `ceil_mode`: 0
  - `dilations`: [1, 1]
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [2, 2]

### Node 3: `Conv`
- **Name:** /layer1/layer1.0/conv1/Conv
- **Inputs:**
  - `/maxpool/MaxPool_output_0`
  - `onnx::Conv_196`
  - `onnx::Conv_197`
- **Outputs:**
  - `/layer1/layer1.0/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 4: `Relu`
- **Name:** /layer1/layer1.0/relu/Relu
- **Inputs:**
  - `/layer1/layer1.0/conv1/Conv_output_0`
- **Outputs:**
  - `/layer1/layer1.0/relu/Relu_output_0`

### Node 5: `Conv`
- **Name:** /layer1/layer1.0/conv2/Conv
- **Inputs:**
  - `/layer1/layer1.0/relu/Relu_output_0`
  - `onnx::Conv_199`
  - `onnx::Conv_200`
- **Outputs:**
  - `/layer1/layer1.0/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 6: `Add`
- **Name:** /layer1/layer1.0/Add
- **Inputs:**
  - `/layer1/layer1.0/conv2/Conv_output_0`
  - `/maxpool/MaxPool_output_0`
- **Outputs:**
  - `/layer1/layer1.0/Add_output_0`

### Node 7: `Relu`
- **Name:** /layer1/layer1.0/relu_1/Relu
- **Inputs:**
  - `/layer1/layer1.0/Add_output_0`
- **Outputs:**
  - `/layer1/layer1.0/relu_1/Relu_output_0`

### Node 8: `Conv`
- **Name:** /layer1/layer1.1/conv1/Conv
- **Inputs:**
  - `/layer1/layer1.0/relu_1/Relu_output_0`
  - `onnx::Conv_202`
  - `onnx::Conv_203`
- **Outputs:**
  - `/layer1/layer1.1/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 9: `Relu`
- **Name:** /layer1/layer1.1/relu/Relu
- **Inputs:**
  - `/layer1/layer1.1/conv1/Conv_output_0`
- **Outputs:**
  - `/layer1/layer1.1/relu/Relu_output_0`

### Node 10: `Conv`
- **Name:** /layer1/layer1.1/conv2/Conv
- **Inputs:**
  - `/layer1/layer1.1/relu/Relu_output_0`
  - `onnx::Conv_205`
  - `onnx::Conv_206`
- **Outputs:**
  - `/layer1/layer1.1/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 11: `Add`
- **Name:** /layer1/layer1.1/Add
- **Inputs:**
  - `/layer1/layer1.1/conv2/Conv_output_0`
  - `/layer1/layer1.0/relu_1/Relu_output_0`
- **Outputs:**
  - `/layer1/layer1.1/Add_output_0`

### Node 12: `Relu`
- **Name:** /layer1/layer1.1/relu_1/Relu
- **Inputs:**
  - `/layer1/layer1.1/Add_output_0`
- **Outputs:**
  - `/layer1/layer1.1/relu_1/Relu_output_0`

### Node 13: `Conv`
- **Name:** /layer2/layer2.0/conv1/Conv
- **Inputs:**
  - `/layer1/layer1.1/relu_1/Relu_output_0`
  - `onnx::Conv_208`
  - `onnx::Conv_209`
- **Outputs:**
  - `/layer2/layer2.0/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [2, 2]

### Node 14: `Relu`
- **Name:** /layer2/layer2.0/relu/Relu
- **Inputs:**
  - `/layer2/layer2.0/conv1/Conv_output_0`
- **Outputs:**
  - `/layer2/layer2.0/relu/Relu_output_0`

### Node 15: `Conv`
- **Name:** /layer2/layer2.0/conv2/Conv
- **Inputs:**
  - `/layer2/layer2.0/relu/Relu_output_0`
  - `onnx::Conv_211`
  - `onnx::Conv_212`
- **Outputs:**
  - `/layer2/layer2.0/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 16: `Conv`
- **Name:** /layer2/layer2.0/downsample/downsample.0/Conv
- **Inputs:**
  - `/layer1/layer1.1/relu_1/Relu_output_0`
  - `onnx::Conv_214`
  - `onnx::Conv_215`
- **Outputs:**
  - `/layer2/layer2.0/downsample/downsample.0/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [1, 1]
  - `pads`: [0, 0, 0, 0]
  - `strides`: [2, 2]

### Node 17: `Add`
- **Name:** /layer2/layer2.0/Add
- **Inputs:**
  - `/layer2/layer2.0/conv2/Conv_output_0`
  - `/layer2/layer2.0/downsample/downsample.0/Conv_output_0`
- **Outputs:**
  - `/layer2/layer2.0/Add_output_0`

### Node 18: `Relu`
- **Name:** /layer2/layer2.0/relu_1/Relu
- **Inputs:**
  - `/layer2/layer2.0/Add_output_0`
- **Outputs:**
  - `/layer2/layer2.0/relu_1/Relu_output_0`

### Node 19: `Conv`
- **Name:** /layer2/layer2.1/conv1/Conv
- **Inputs:**
  - `/layer2/layer2.0/relu_1/Relu_output_0`
  - `onnx::Conv_217`
  - `onnx::Conv_218`
- **Outputs:**
  - `/layer2/layer2.1/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 20: `Relu`
- **Name:** /layer2/layer2.1/relu/Relu
- **Inputs:**
  - `/layer2/layer2.1/conv1/Conv_output_0`
- **Outputs:**
  - `/layer2/layer2.1/relu/Relu_output_0`

### Node 21: `Conv`
- **Name:** /layer2/layer2.1/conv2/Conv
- **Inputs:**
  - `/layer2/layer2.1/relu/Relu_output_0`
  - `onnx::Conv_220`
  - `onnx::Conv_221`
- **Outputs:**
  - `/layer2/layer2.1/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 22: `Add`
- **Name:** /layer2/layer2.1/Add
- **Inputs:**
  - `/layer2/layer2.1/conv2/Conv_output_0`
  - `/layer2/layer2.0/relu_1/Relu_output_0`
- **Outputs:**
  - `/layer2/layer2.1/Add_output_0`

### Node 23: `Relu`
- **Name:** /layer2/layer2.1/relu_1/Relu
- **Inputs:**
  - `/layer2/layer2.1/Add_output_0`
- **Outputs:**
  - `/layer2/layer2.1/relu_1/Relu_output_0`

### Node 24: `Conv`
- **Name:** /layer3/layer3.0/conv1/Conv
- **Inputs:**
  - `/layer2/layer2.1/relu_1/Relu_output_0`
  - `onnx::Conv_223`
  - `onnx::Conv_224`
- **Outputs:**
  - `/layer3/layer3.0/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [2, 2]

### Node 25: `Relu`
- **Name:** /layer3/layer3.0/relu/Relu
- **Inputs:**
  - `/layer3/layer3.0/conv1/Conv_output_0`
- **Outputs:**
  - `/layer3/layer3.0/relu/Relu_output_0`

### Node 26: `Conv`
- **Name:** /layer3/layer3.0/conv2/Conv
- **Inputs:**
  - `/layer3/layer3.0/relu/Relu_output_0`
  - `onnx::Conv_226`
  - `onnx::Conv_227`
- **Outputs:**
  - `/layer3/layer3.0/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 27: `Conv`
- **Name:** /layer3/layer3.0/downsample/downsample.0/Conv
- **Inputs:**
  - `/layer2/layer2.1/relu_1/Relu_output_0`
  - `onnx::Conv_229`
  - `onnx::Conv_230`
- **Outputs:**
  - `/layer3/layer3.0/downsample/downsample.0/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [1, 1]
  - `pads`: [0, 0, 0, 0]
  - `strides`: [2, 2]

### Node 28: `Add`
- **Name:** /layer3/layer3.0/Add
- **Inputs:**
  - `/layer3/layer3.0/conv2/Conv_output_0`
  - `/layer3/layer3.0/downsample/downsample.0/Conv_output_0`
- **Outputs:**
  - `/layer3/layer3.0/Add_output_0`

### Node 29: `Relu`
- **Name:** /layer3/layer3.0/relu_1/Relu
- **Inputs:**
  - `/layer3/layer3.0/Add_output_0`
- **Outputs:**
  - `/layer3/layer3.0/relu_1/Relu_output_0`

### Node 30: `Conv`
- **Name:** /layer3/layer3.1/conv1/Conv
- **Inputs:**
  - `/layer3/layer3.0/relu_1/Relu_output_0`
  - `onnx::Conv_232`
  - `onnx::Conv_233`
- **Outputs:**
  - `/layer3/layer3.1/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 31: `Relu`
- **Name:** /layer3/layer3.1/relu/Relu
- **Inputs:**
  - `/layer3/layer3.1/conv1/Conv_output_0`
- **Outputs:**
  - `/layer3/layer3.1/relu/Relu_output_0`

### Node 32: `Conv`
- **Name:** /layer3/layer3.1/conv2/Conv
- **Inputs:**
  - `/layer3/layer3.1/relu/Relu_output_0`
  - `onnx::Conv_235`
  - `onnx::Conv_236`
- **Outputs:**
  - `/layer3/layer3.1/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 33: `Add`
- **Name:** /layer3/layer3.1/Add
- **Inputs:**
  - `/layer3/layer3.1/conv2/Conv_output_0`
  - `/layer3/layer3.0/relu_1/Relu_output_0`
- **Outputs:**
  - `/layer3/layer3.1/Add_output_0`

### Node 34: `Relu`
- **Name:** /layer3/layer3.1/relu_1/Relu
- **Inputs:**
  - `/layer3/layer3.1/Add_output_0`
- **Outputs:**
  - `/layer3/layer3.1/relu_1/Relu_output_0`

### Node 35: `Conv`
- **Name:** /layer4/layer4.0/conv1/Conv
- **Inputs:**
  - `/layer3/layer3.1/relu_1/Relu_output_0`
  - `onnx::Conv_238`
  - `onnx::Conv_239`
- **Outputs:**
  - `/layer4/layer4.0/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [2, 2]

### Node 36: `Relu`
- **Name:** /layer4/layer4.0/relu/Relu
- **Inputs:**
  - `/layer4/layer4.0/conv1/Conv_output_0`
- **Outputs:**
  - `/layer4/layer4.0/relu/Relu_output_0`

### Node 37: `Conv`
- **Name:** /layer4/layer4.0/conv2/Conv
- **Inputs:**
  - `/layer4/layer4.0/relu/Relu_output_0`
  - `onnx::Conv_241`
  - `onnx::Conv_242`
- **Outputs:**
  - `/layer4/layer4.0/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 38: `Conv`
- **Name:** /layer4/layer4.0/downsample/downsample.0/Conv
- **Inputs:**
  - `/layer3/layer3.1/relu_1/Relu_output_0`
  - `onnx::Conv_244`
  - `onnx::Conv_245`
- **Outputs:**
  - `/layer4/layer4.0/downsample/downsample.0/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [1, 1]
  - `pads`: [0, 0, 0, 0]
  - `strides`: [2, 2]

### Node 39: `Add`
- **Name:** /layer4/layer4.0/Add
- **Inputs:**
  - `/layer4/layer4.0/conv2/Conv_output_0`
  - `/layer4/layer4.0/downsample/downsample.0/Conv_output_0`
- **Outputs:**
  - `/layer4/layer4.0/Add_output_0`

### Node 40: `Relu`
- **Name:** /layer4/layer4.0/relu_1/Relu
- **Inputs:**
  - `/layer4/layer4.0/Add_output_0`
- **Outputs:**
  - `/layer4/layer4.0/relu_1/Relu_output_0`

### Node 41: `Conv`
- **Name:** /layer4/layer4.1/conv1/Conv
- **Inputs:**
  - `/layer4/layer4.0/relu_1/Relu_output_0`
  - `onnx::Conv_247`
  - `onnx::Conv_248`
- **Outputs:**
  - `/layer4/layer4.1/conv1/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 42: `Relu`
- **Name:** /layer4/layer4.1/relu/Relu
- **Inputs:**
  - `/layer4/layer4.1/conv1/Conv_output_0`
- **Outputs:**
  - `/layer4/layer4.1/relu/Relu_output_0`

### Node 43: `Conv`
- **Name:** /layer4/layer4.1/conv2/Conv
- **Inputs:**
  - `/layer4/layer4.1/relu/Relu_output_0`
  - `onnx::Conv_250`
  - `onnx::Conv_251`
- **Outputs:**
  - `/layer4/layer4.1/conv2/Conv_output_0`
- **Attributes:**
  - `dilations`: [1, 1]
  - `group`: 1
  - `kernel_shape`: [3, 3]
  - `pads`: [1, 1, 1, 1]
  - `strides`: [1, 1]

### Node 44: `Add`
- **Name:** /layer4/layer4.1/Add
- **Inputs:**
  - `/layer4/layer4.1/conv2/Conv_output_0`
  - `/layer4/layer4.0/relu_1/Relu_output_0`
- **Outputs:**
  - `/layer4/layer4.1/Add_output_0`

### Node 45: `Relu`
- **Name:** /layer4/layer4.1/relu_1/Relu
- **Inputs:**
  - `/layer4/layer4.1/Add_output_0`
- **Outputs:**
  - `/layer4/layer4.1/relu_1/Relu_output_0`

### Node 46: `GlobalAveragePool`
- **Name:** /avgpool/GlobalAveragePool
- **Inputs:**
  - `/layer4/layer4.1/relu_1/Relu_output_0`
- **Outputs:**
  - `/avgpool/GlobalAveragePool_output_0`

### Node 47: `Flatten`
- **Name:** /Flatten
- **Inputs:**
  - `/avgpool/GlobalAveragePool_output_0`
- **Outputs:**
  - `/Flatten_output_0`
- **Attributes:**
  - `axis`: 1

### Node 48: `Gemm`
- **Name:** /fc/Gemm
- **Inputs:**
  - `/Flatten_output_0`
  - `fc.weight`
  - `fc.bias`
- **Outputs:**
  - `output`
- **Attributes:**
  - `alpha`: 1.0
  - `beta`: 1.0
  - `transB`: 1

