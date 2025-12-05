# ONNX Model Report

- **Model Path:** `C:\Users\Administrator\Desktop\model.onnx`
- **IR Version:** 7
- **Opset Import:** [':14']
- **Producer:** pytorch (2.4.1)

## Inputs
| Name | Shape | Type |
|------|-------|------|
| `input_ids` | `[batch_size, sequence_length]` | INT64 |
| `attention_mask` | `[batch_size, sequence_length]` | INT64 |

## Outputs
| Name | Shape | Type |
|------|-------|------|
| `start_logits` | `[batch_size, sequence_length]` | FLOAT |
| `end_logits` | `[batch_size, sequence_length]` | FLOAT |

## Initializers (Weights)
Total Initializers: 102

| Name | Shape | Type | Stats |
|------|-------|------|-------|
| `distilbert.embeddings.word_embeddings.weight` | `[30522, 768]` | FLOAT | min:-1.10, max:0.59 |
| `distilbert.embeddings.position_embeddings.weight` | `[512, 768]` | FLOAT | min:-0.95, max:0.73 |
| `distilbert.embeddings.LayerNorm.weight` | `[768]` | FLOAT | min:0.08, max:0.93 |
| `distilbert.embeddings.LayerNorm.bias` | `[768]` | FLOAT | min:-0.48, max:0.59 |
| `distilbert.transformer.layer.0.attention.q_lin.bias` | `[768]` | FLOAT | min:-1.09, max:1.10 |
| `distilbert.transformer.layer.0.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.01, max:0.01 |
| `distilbert.transformer.layer.0.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.38, max:0.43 |
| `distilbert.transformer.layer.0.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.56, max:0.14 |
| `distilbert.transformer.layer.0.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.50, max:2.96 |
| `distilbert.transformer.layer.0.sa_layer_norm.bias` | `[768]` | FLOAT | min:-4.30, max:0.85 |
| `distilbert.transformer.layer.0.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.33, max:0.43 |
| `distilbert.transformer.layer.0.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.53, max:0.33 |
| `distilbert.transformer.layer.0.output_layer_norm.weight` | `[768]` | FLOAT | min:0.10, max:0.85 |
| `distilbert.transformer.layer.0.output_layer_norm.bias` | `[768]` | FLOAT | min:-1.40, max:0.42 |
| `distilbert.transformer.layer.1.attention.q_lin.bias` | `[768]` | FLOAT | min:-0.89, max:0.56 |
| `distilbert.transformer.layer.1.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.02, max:0.03 |
| `distilbert.transformer.layer.1.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.31, max:0.70 |
| `distilbert.transformer.layer.1.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.43, max:0.39 |
| `distilbert.transformer.layer.1.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.43, max:2.04 |
| `distilbert.transformer.layer.1.sa_layer_norm.bias` | `[768]` | FLOAT | min:-2.09, max:0.71 |
| `distilbert.transformer.layer.1.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.36, max:0.64 |
| `distilbert.transformer.layer.1.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.31, max:0.31 |
| `distilbert.transformer.layer.1.output_layer_norm.weight` | `[768]` | FLOAT | min:0.35, max:0.93 |
| `distilbert.transformer.layer.1.output_layer_norm.bias` | `[768]` | FLOAT | min:-0.32, max:0.24 |
| `distilbert.transformer.layer.2.attention.q_lin.bias` | `[768]` | FLOAT | min:-0.76, max:0.66 |
| `distilbert.transformer.layer.2.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.02, max:0.02 |
| `distilbert.transformer.layer.2.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.20, max:0.20 |
| `distilbert.transformer.layer.2.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.21, max:0.46 |
| `distilbert.transformer.layer.2.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.40, max:3.88 |
| `distilbert.transformer.layer.2.sa_layer_norm.bias` | `[768]` | FLOAT | min:-1.33, max:0.78 |
| `distilbert.transformer.layer.2.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.40, max:0.54 |
| `distilbert.transformer.layer.2.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.32, max:0.22 |
| `distilbert.transformer.layer.2.output_layer_norm.weight` | `[768]` | FLOAT | min:0.47, max:0.94 |
| `distilbert.transformer.layer.2.output_layer_norm.bias` | `[768]` | FLOAT | min:-0.26, max:0.25 |
| `distilbert.transformer.layer.3.attention.q_lin.bias` | `[768]` | FLOAT | min:-0.96, max:0.64 |
| `distilbert.transformer.layer.3.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.02, max:0.02 |
| `distilbert.transformer.layer.3.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.30, max:0.21 |
| `distilbert.transformer.layer.3.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.22, max:1.39 |
| `distilbert.transformer.layer.3.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.48, max:3.46 |
| `distilbert.transformer.layer.3.sa_layer_norm.bias` | `[768]` | FLOAT | min:-2.17, max:0.48 |
| `distilbert.transformer.layer.3.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.46, max:0.62 |
| `distilbert.transformer.layer.3.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.75, max:0.23 |
| `distilbert.transformer.layer.3.output_layer_norm.weight` | `[768]` | FLOAT | min:0.31, max:1.15 |
| `distilbert.transformer.layer.3.output_layer_norm.bias` | `[768]` | FLOAT | min:-0.23, max:0.96 |
| `distilbert.transformer.layer.4.attention.q_lin.bias` | `[768]` | FLOAT | min:-0.79, max:0.82 |
| `distilbert.transformer.layer.4.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.03, max:0.03 |
| `distilbert.transformer.layer.4.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.24, max:0.28 |
| `distilbert.transformer.layer.4.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.19, max:0.64 |
| `distilbert.transformer.layer.4.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.54, max:2.47 |
| `distilbert.transformer.layer.4.sa_layer_norm.bias` | `[768]` | FLOAT | min:-1.46, max:1.06 |
| `distilbert.transformer.layer.4.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.36, max:0.61 |
| `distilbert.transformer.layer.4.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.67, max:0.34 |
| `distilbert.transformer.layer.4.output_layer_norm.weight` | `[768]` | FLOAT | min:0.09, max:1.42 |
| `distilbert.transformer.layer.4.output_layer_norm.bias` | `[768]` | FLOAT | min:-0.40, max:0.38 |
| `distilbert.transformer.layer.5.attention.q_lin.bias` | `[768]` | FLOAT | min:-0.86, max:0.86 |
| `distilbert.transformer.layer.5.attention.k_lin.bias` | `[768]` | FLOAT | min:-0.02, max:0.03 |
| `distilbert.transformer.layer.5.attention.v_lin.bias` | `[768]` | FLOAT | min:-0.13, max:0.08 |
| `distilbert.transformer.layer.5.attention.out_lin.bias` | `[768]` | FLOAT | min:-0.16, max:0.20 |
| `distilbert.transformer.layer.5.sa_layer_norm.weight` | `[768]` | FLOAT | min:0.60, max:1.52 |
| `distilbert.transformer.layer.5.sa_layer_norm.bias` | `[768]` | FLOAT | min:-2.13, max:0.39 |
| `distilbert.transformer.layer.5.ffn.lin1.bias` | `[3072]` | FLOAT | min:-0.45, max:0.59 |
| `distilbert.transformer.layer.5.ffn.lin2.bias` | `[768]` | FLOAT | min:-0.58, max:0.15 |
| `distilbert.transformer.layer.5.output_layer_norm.weight` | `[768]` | FLOAT | min:0.21, max:0.77 |
| `distilbert.transformer.layer.5.output_layer_norm.bias` | `[768]` | FLOAT | min:-0.17, max:0.38 |
| `qa_outputs.bias` | `[2]` | FLOAT | min:0.00, max:0.00 |
| `onnx::MatMul_861` | `[768, 768]` | FLOAT | min:-0.70, max:0.50 |
| `onnx::MatMul_871` | `[768, 768]` | FLOAT | min:-0.94, max:0.77 |
| `onnx::MatMul_872` | `[768, 768]` | FLOAT | min:-0.19, max:0.20 |
| `onnx::MatMul_875` | `[768, 768]` | FLOAT | min:-0.61, max:0.74 |
| `onnx::MatMul_876` | `[768, 3072]` | FLOAT | min:-0.35, max:0.33 |
| `onnx::MatMul_877` | `[3072, 768]` | FLOAT | min:-2.08, max:0.92 |
| `onnx::MatMul_878` | `[768, 768]` | FLOAT | min:-0.53, max:0.57 |
| `onnx::MatMul_888` | `[768, 768]` | FLOAT | min:-0.59, max:0.52 |
| `onnx::MatMul_889` | `[768, 768]` | FLOAT | min:-0.22, max:0.26 |
| `onnx::MatMul_892` | `[768, 768]` | FLOAT | min:-0.46, max:0.45 |
| `onnx::MatMul_893` | `[768, 3072]` | FLOAT | min:-0.69, max:0.61 |
| `onnx::MatMul_894` | `[3072, 768]` | FLOAT | min:-4.05, max:1.18 |
| `onnx::MatMul_895` | `[768, 768]` | FLOAT | min:-0.40, max:0.40 |
| `onnx::MatMul_905` | `[768, 768]` | FLOAT | min:-0.34, max:0.37 |
| `onnx::MatMul_906` | `[768, 768]` | FLOAT | min:-0.24, max:0.24 |
| `onnx::MatMul_909` | `[768, 768]` | FLOAT | min:-0.36, max:0.31 |
| `onnx::MatMul_910` | `[768, 3072]` | FLOAT | min:-0.79, max:1.21 |
| `onnx::MatMul_911` | `[3072, 768]` | FLOAT | min:-10.08, max:2.19 |
| `onnx::MatMul_912` | `[768, 768]` | FLOAT | min:-0.27, max:0.30 |
| `onnx::MatMul_922` | `[768, 768]` | FLOAT | min:-0.42, max:0.48 |
| `onnx::MatMul_923` | `[768, 768]` | FLOAT | min:-0.32, max:0.27 |
| `onnx::MatMul_926` | `[768, 768]` | FLOAT | min:-0.28, max:0.29 |
| `onnx::MatMul_927` | `[768, 3072]` | FLOAT | min:-0.38, max:0.49 |
| `onnx::MatMul_928` | `[3072, 768]` | FLOAT | min:-3.15, max:1.00 |
| `onnx::MatMul_929` | `[768, 768]` | FLOAT | min:-0.26, max:0.25 |
| `onnx::MatMul_939` | `[768, 768]` | FLOAT | min:-0.59, max:0.66 |
| `onnx::MatMul_940` | `[768, 768]` | FLOAT | min:-0.26, max:0.24 |
| `onnx::MatMul_943` | `[768, 768]` | FLOAT | min:-0.38, max:0.44 |
| `onnx::MatMul_944` | `[768, 3072]` | FLOAT | min:-0.88, max:0.86 |
| `onnx::MatMul_945` | `[3072, 768]` | FLOAT | min:-4.28, max:2.44 |
| `onnx::MatMul_946` | `[768, 768]` | FLOAT | min:-0.65, max:0.63 |
| `onnx::MatMul_956` | `[768, 768]` | FLOAT | min:-0.45, max:0.29 |
| `onnx::MatMul_957` | `[768, 768]` | FLOAT | min:-0.32, max:0.26 |
| `onnx::MatMul_960` | `[768, 768]` | FLOAT | min:-0.52, max:0.43 |
| `onnx::MatMul_961` | `[768, 3072]` | FLOAT | min:-0.42, max:0.49 |
| `onnx::MatMul_962` | `[3072, 768]` | FLOAT | min:-3.19, max:1.09 |
| `onnx::MatMul_963` | `[768, 2]` | FLOAT | min:-0.08, max:0.08 |

## Computation Graph (Nodes)
Total Nodes: 639

### Node 0: `Shape`
- **Name:** /distilbert/Shape
- **Inputs:**
  - `input_ids`
- **Outputs:**
  - `/distilbert/Shape_output_0`

### Node 1: `Constant`
- **Name:** /distilbert/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 2: `Gather`
- **Name:** /distilbert/Gather
- **Inputs:**
  - `/distilbert/Shape_output_0`
  - `/distilbert/Constant_output_0`
- **Outputs:**
  - `/distilbert/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 3: `Gather`
- **Name:** /distilbert/embeddings/word_embeddings/Gather
- **Inputs:**
  - `distilbert.embeddings.word_embeddings.weight`
  - `input_ids`
- **Outputs:**
  - `/distilbert/embeddings/word_embeddings/Gather_output_0`

### Node 4: `Shape`
- **Name:** /distilbert/embeddings/Shape
- **Inputs:**
  - `/distilbert/embeddings/word_embeddings/Gather_output_0`
- **Outputs:**
  - `/distilbert/embeddings/Shape_output_0`

### Node 5: `Constant`
- **Name:** /distilbert/embeddings/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 6: `Gather`
- **Name:** /distilbert/embeddings/Gather
- **Inputs:**
  - `/distilbert/embeddings/Shape_output_0`
  - `/distilbert/embeddings/Constant_output_0`
- **Outputs:**
  - `/distilbert/embeddings/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 7: `Constant`
- **Name:** Constant_187
- **Inputs:**
- **Outputs:**
  - `onnx::Slice_113`
- **Attributes:**
  - `value`: <Tensor: >

### Node 8: `Constant`
- **Name:** /distilbert/embeddings/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 9: `Constant`
- **Name:** /distilbert/embeddings/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 10: `Constant`
- **Name:** /distilbert/embeddings/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 11: `Unsqueeze`
- **Name:** /distilbert/embeddings/Unsqueeze
- **Inputs:**
  - `/distilbert/embeddings/Gather_output_0`
  - `/distilbert/embeddings/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/embeddings/Unsqueeze_output_0`

### Node 12: `Constant`
- **Name:** /distilbert/embeddings/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 13: `Slice`
- **Name:** /distilbert/embeddings/Slice
- **Inputs:**
  - `onnx::Slice_113`
  - `/distilbert/embeddings/Constant_2_output_0`
  - `/distilbert/embeddings/Unsqueeze_output_0`
  - `/distilbert/embeddings/Constant_1_output_0`
  - `/distilbert/embeddings/Constant_4_output_0`
- **Outputs:**
  - `/distilbert/embeddings/Slice_output_0`

### Node 14: `Gather`
- **Name:** /distilbert/embeddings/position_embeddings/Gather
- **Inputs:**
  - `distilbert.embeddings.position_embeddings.weight`
  - `/distilbert/embeddings/Slice_output_0`
- **Outputs:**
  - `/distilbert/embeddings/position_embeddings/Gather_output_0`

### Node 15: `Add`
- **Name:** /distilbert/embeddings/Add
- **Inputs:**
  - `/distilbert/embeddings/word_embeddings/Gather_output_0`
  - `/distilbert/embeddings/position_embeddings/Gather_output_0`
- **Outputs:**
  - `/distilbert/embeddings/Add_output_0`

### Node 16: `ReduceMean`
- **Name:** /distilbert/embeddings/LayerNorm/ReduceMean
- **Inputs:**
  - `/distilbert/embeddings/Add_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 17: `Sub`
- **Name:** /distilbert/embeddings/LayerNorm/Sub
- **Inputs:**
  - `/distilbert/embeddings/Add_output_0`
  - `/distilbert/embeddings/LayerNorm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Sub_output_0`

### Node 18: `Constant`
- **Name:** /distilbert/embeddings/LayerNorm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 19: `Pow`
- **Name:** /distilbert/embeddings/LayerNorm/Pow
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Sub_output_0`
  - `/distilbert/embeddings/LayerNorm/Constant_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Pow_output_0`

### Node 20: `ReduceMean`
- **Name:** /distilbert/embeddings/LayerNorm/ReduceMean_1
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Pow_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 21: `Constant`
- **Name:** /distilbert/embeddings/LayerNorm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 22: `Add`
- **Name:** /distilbert/embeddings/LayerNorm/Add
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/ReduceMean_1_output_0`
  - `/distilbert/embeddings/LayerNorm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Add_output_0`

### Node 23: `Sqrt`
- **Name:** /distilbert/embeddings/LayerNorm/Sqrt
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Add_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Sqrt_output_0`

### Node 24: `Div`
- **Name:** /distilbert/embeddings/LayerNorm/Div
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Sub_output_0`
  - `/distilbert/embeddings/LayerNorm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Div_output_0`

### Node 25: `Mul`
- **Name:** /distilbert/embeddings/LayerNorm/Mul
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Div_output_0`
  - `distilbert.embeddings.LayerNorm.weight`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Mul_output_0`

### Node 26: `Add`
- **Name:** /distilbert/embeddings/LayerNorm/Add_1
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Mul_output_0`
  - `distilbert.embeddings.LayerNorm.bias`
- **Outputs:**
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`

### Node 27: `Shape`
- **Name:** /distilbert/Shape_1
- **Inputs:**
  - `attention_mask`
- **Outputs:**
  - `/distilbert/Shape_1_output_0`

### Node 28: `Constant`
- **Name:** /distilbert/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 29: `Gather`
- **Name:** /distilbert/Gather_1
- **Inputs:**
  - `/distilbert/Shape_1_output_0`
  - `/distilbert/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/Gather_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 30: `Shape`
- **Name:** /distilbert/Shape_2
- **Inputs:**
  - `attention_mask`
- **Outputs:**
  - `/distilbert/Shape_2_output_0`

### Node 31: `Constant`
- **Name:** /distilbert/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 32: `Gather`
- **Name:** /distilbert/Gather_2
- **Inputs:**
  - `/distilbert/Shape_2_output_0`
  - `/distilbert/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/Gather_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 33: `Constant`
- **Name:** /distilbert/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 34: `Unsqueeze`
- **Name:** /distilbert/Unsqueeze
- **Inputs:**
  - `attention_mask`
  - `/distilbert/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/Unsqueeze_output_0`

### Node 35: `Constant`
- **Name:** /distilbert/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 36: `Unsqueeze`
- **Name:** /distilbert/Unsqueeze_1
- **Inputs:**
  - `/distilbert/Unsqueeze_output_0`
  - `/distilbert/Constant_4_output_0`
- **Outputs:**
  - `/distilbert/Unsqueeze_1_output_0`

### Node 37: `Constant`
- **Name:** /distilbert/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 38: `Unsqueeze`
- **Name:** /distilbert/Unsqueeze_2
- **Inputs:**
  - `/distilbert/Gather_1_output_0`
  - `/distilbert/Constant_5_output_0`
- **Outputs:**
  - `/distilbert/Unsqueeze_2_output_0`

### Node 39: `Constant`
- **Name:** /distilbert/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 40: `Constant`
- **Name:** /distilbert/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 41: `Unsqueeze`
- **Name:** /distilbert/Unsqueeze_3
- **Inputs:**
  - `/distilbert/Gather_output_0`
  - `/distilbert/Constant_7_output_0`
- **Outputs:**
  - `/distilbert/Unsqueeze_3_output_0`

### Node 42: `Constant`
- **Name:** /distilbert/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 43: `Unsqueeze`
- **Name:** /distilbert/Unsqueeze_4
- **Inputs:**
  - `/distilbert/Gather_2_output_0`
  - `/distilbert/Constant_8_output_0`
- **Outputs:**
  - `/distilbert/Unsqueeze_4_output_0`

### Node 44: `Concat`
- **Name:** /distilbert/Concat
- **Inputs:**
  - `/distilbert/Unsqueeze_2_output_0`
  - `/distilbert/Constant_6_output_0`
  - `/distilbert/Unsqueeze_3_output_0`
  - `/distilbert/Unsqueeze_4_output_0`
- **Outputs:**
  - `/distilbert/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 45: `Constant`
- **Name:** /distilbert/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 46: `Reshape`
- **Name:** /distilbert/Reshape
- **Inputs:**
  - `/distilbert/Concat_output_0`
  - `/distilbert/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 47: `Shape`
- **Name:** /distilbert/Shape_3
- **Inputs:**
  - `/distilbert/Reshape_output_0`
- **Outputs:**
  - `/distilbert/Shape_3_output_0`

### Node 48: `ConstantOfShape`
- **Name:** /distilbert/ConstantOfShape
- **Inputs:**
  - `/distilbert/Shape_3_output_0`
- **Outputs:**
  - `/distilbert/ConstantOfShape_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 49: `Constant`
- **Name:** /distilbert/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 50: `Mul`
- **Name:** /distilbert/Mul
- **Inputs:**
  - `/distilbert/ConstantOfShape_output_0`
  - `/distilbert/Constant_10_output_0`
- **Outputs:**
  - `/distilbert/Mul_output_0`

### Node 51: `Equal`
- **Name:** /distilbert/Equal
- **Inputs:**
  - `/distilbert/Reshape_output_0`
  - `/distilbert/Mul_output_0`
- **Outputs:**
  - `/distilbert/Equal_output_0`

### Node 52: `Where`
- **Name:** /distilbert/Where
- **Inputs:**
  - `/distilbert/Equal_output_0`
  - `/distilbert/ConstantOfShape_output_0`
  - `/distilbert/Reshape_output_0`
- **Outputs:**
  - `/distilbert/Where_output_0`

### Node 53: `Expand`
- **Name:** /distilbert/Expand
- **Inputs:**
  - `/distilbert/Unsqueeze_1_output_0`
  - `/distilbert/Where_output_0`
- **Outputs:**
  - `/distilbert/Expand_output_0`

### Node 54: `Cast`
- **Name:** /distilbert/Cast
- **Inputs:**
  - `/distilbert/Expand_output_0`
- **Outputs:**
  - `/distilbert/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 55: `Constant`
- **Name:** /distilbert/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 56: `Sub`
- **Name:** /distilbert/Sub
- **Inputs:**
  - `/distilbert/Constant_11_output_0`
  - `/distilbert/Cast_output_0`
- **Outputs:**
  - `/distilbert/Sub_output_0`

### Node 57: `Cast`
- **Name:** /distilbert/Cast_1
- **Inputs:**
  - `/distilbert/Sub_output_0`
- **Outputs:**
  - `/distilbert/Cast_1_output_0`
- **Attributes:**
  - `to`: 9

### Node 58: `Cast`
- **Name:** /distilbert/Cast_2
- **Inputs:**
  - `/distilbert/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/Cast_2_output_0`
- **Attributes:**
  - `to`: 9

### Node 59: `Constant`
- **Name:** /distilbert/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 60: `Where`
- **Name:** /distilbert/Where_1
- **Inputs:**
  - `/distilbert/Cast_2_output_0`
  - `/distilbert/Constant_12_output_0`
  - `/distilbert/Sub_output_0`
- **Outputs:**
  - `/distilbert/Where_1_output_0`

### Node 61: `Shape`
- **Name:** /distilbert/transformer/layer.0/attention/Shape
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Shape_output_0`

### Node 62: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 63: `Gather`
- **Name:** /distilbert/transformer/layer.0/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Shape_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 64: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`
  - `onnx::MatMul_861`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/q_lin/MatMul_output_0`

### Node 65: `Add`
- **Name:** /distilbert/transformer/layer.0/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.attention.q_lin.bias`
  - `/distilbert/transformer/layer.0/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/q_lin/Add_output_0`

### Node 66: `Constant`
- **Name:** Constant_246
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_180`
- **Attributes:**
  - `value`: <Tensor: >

### Node 67: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.0/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Gather_output_0`
  - `onnx::Unsqueeze_180`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_output_0`

### Node 68: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 69: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 70: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 71: `Concat`
- **Name:** /distilbert/transformer/layer.0/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 72: `Constant`
- **Name:** Constant_252
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_189`
- **Attributes:**
  - `value`: <Tensor: >

### Node 73: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.0/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Gather_output_0`
  - `onnx::Unsqueeze_189`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_1_output_0`

### Node 74: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 75: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 76: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 77: `Concat`
- **Name:** /distilbert/transformer/layer.0/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 78: `Constant`
- **Name:** Constant_258
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_198`
- **Attributes:**
  - `value`: <Tensor: >

### Node 79: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.0/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Gather_output_0`
  - `onnx::Unsqueeze_198`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_2_output_0`

### Node 80: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 81: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 82: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 83: `Concat`
- **Name:** /distilbert/transformer/layer.0/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 84: `Reshape`
- **Name:** /distilbert/transformer/layer.0/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.0/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 85: `Transpose`
- **Name:** /distilbert/transformer/layer.0/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 86: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`
  - `onnx::MatMul_871`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/k_lin/MatMul_output_0`

### Node 87: `Add`
- **Name:** /distilbert/transformer/layer.0/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.attention.k_lin.bias`
  - `/distilbert/transformer/layer.0/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/k_lin/Add_output_0`

### Node 88: `Reshape`
- **Name:** /distilbert/transformer/layer.0/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.0/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 89: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`
  - `onnx::MatMul_872`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/v_lin/MatMul_output_0`

### Node 90: `Add`
- **Name:** /distilbert/transformer/layer.0/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.attention.v_lin.bias`
  - `/distilbert/transformer/layer.0/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/v_lin/Add_output_0`

### Node 91: `Reshape`
- **Name:** /distilbert/transformer/layer.0/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.0/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 92: `Transpose`
- **Name:** /distilbert/transformer/layer.0/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 93: `Shape`
- **Name:** /distilbert/transformer/layer.0/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Shape_1_output_0`

### Node 94: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 95: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 96: `Slice`
- **Name:** /distilbert/transformer/layer.0/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Slice_output_0`

### Node 97: `Cast`
- **Name:** /distilbert/transformer/layer.0/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 98: `Sqrt`
- **Name:** /distilbert/transformer/layer.0/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Sqrt_output_0`

### Node 99: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 100: `Div`
- **Name:** /distilbert/transformer/layer.0/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.0/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Div_output_0`

### Node 101: `Cast`
- **Name:** /distilbert/transformer/layer.0/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 102: `Transpose`
- **Name:** /distilbert/transformer/layer.0/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 103: `Sqrt`
- **Name:** /distilbert/transformer/layer.0/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Sqrt_1_output_0`

### Node 104: `Mul`
- **Name:** /distilbert/transformer/layer.0/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.0/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Mul_output_0`

### Node 105: `Sqrt`
- **Name:** /distilbert/transformer/layer.0/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Sqrt_2_output_0`

### Node 106: `Mul`
- **Name:** /distilbert/transformer/layer.0/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.0/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Mul_1_output_0`

### Node 107: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Mul_output_0`
  - `/distilbert/transformer/layer.0/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/MatMul_output_0`

### Node 108: `Add`
- **Name:** /distilbert/transformer/layer.0/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Add_output_0`

### Node 109: `Softmax`
- **Name:** /distilbert/transformer/layer.0/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 110: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.0/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/MatMul_1_output_0`

### Node 111: `Transpose`
- **Name:** /distilbert/transformer/layer.0/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 112: `Constant`
- **Name:** Constant_292
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_238`
- **Attributes:**
  - `value`: <Tensor: >

### Node 113: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.0/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Gather_output_0`
  - `onnx::Unsqueeze_238`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_3_output_0`

### Node 114: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 115: `Constant`
- **Name:** /distilbert/transformer/layer.0/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 116: `Concat`
- **Name:** /distilbert/transformer/layer.0/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.0/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 117: `Reshape`
- **Name:** /distilbert/transformer/layer.0/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.0/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 118: `MatMul`
- **Name:** /distilbert/transformer/layer.0/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/Reshape_3_output_0`
  - `onnx::MatMul_875`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/out_lin/MatMul_output_0`

### Node 119: `Add`
- **Name:** /distilbert/transformer/layer.0/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.attention.out_lin.bias`
  - `/distilbert/transformer/layer.0/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/attention/out_lin/Add_output_0`

### Node 120: `Add`
- **Name:** /distilbert/transformer/layer.0/Add
- **Inputs:**
  - `/distilbert/transformer/layer.0/attention/out_lin/Add_output_0`
  - `/distilbert/embeddings/LayerNorm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/Add_output_0`

### Node 121: `ReduceMean`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.0/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 122: `Sub`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.0/Add_output_0`
  - `/distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Sub_output_0`

### Node 123: `Constant`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 124: `Pow`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.0/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Pow_output_0`

### Node 125: `ReduceMean`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 126: `Constant`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 127: `Add`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.0/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Add_output_0`

### Node 128: `Sqrt`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Sqrt_output_0`

### Node 129: `Div`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.0/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Div_output_0`

### Node 130: `Mul`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.0.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Mul_output_0`

### Node 131: `Add`
- **Name:** /distilbert/transformer/layer.0/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.0.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Add_1_output_0`

### Node 132: `MatMul`
- **Name:** /distilbert/transformer/layer.0/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_876`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/lin1/MatMul_output_0`

### Node 133: `Add`
- **Name:** /distilbert/transformer/layer.0/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.ffn.lin1.bias`
  - `/distilbert/transformer/layer.0/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/lin1/Add_output_0`

### Node 134: `Constant`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 135: `Div`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Div_output_0`

### Node 136: `Erf`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Erf_output_0`

### Node 137: `Constant`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 138: `Add`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Add_output_0`

### Node 139: `Mul`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.0/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Mul_output_0`

### Node 140: `Constant`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 141: `Mul`
- **Name:** /distilbert/transformer/layer.0/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.0/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Mul_1_output_0`

### Node 142: `MatMul`
- **Name:** /distilbert/transformer/layer.0/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_877`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/lin2/MatMul_output_0`

### Node 143: `Add`
- **Name:** /distilbert/transformer/layer.0/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.0.ffn.lin2.bias`
  - `/distilbert/transformer/layer.0/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/ffn/lin2/Add_output_0`

### Node 144: `Add`
- **Name:** /distilbert/transformer/layer.0/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.0/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/Add_1_output_0`

### Node 145: `ReduceMean`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.0/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 146: `Sub`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.0/Add_1_output_0`
  - `/distilbert/transformer/layer.0/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Sub_output_0`

### Node 147: `Constant`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 148: `Pow`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.0/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Pow_output_0`

### Node 149: `ReduceMean`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 150: `Constant`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 151: `Add`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.0/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_output_0`

### Node 152: `Sqrt`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Sqrt_output_0`

### Node 153: `Div`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.0/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Div_output_0`

### Node 154: `Mul`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.0.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Mul_output_0`

### Node 155: `Add`
- **Name:** /distilbert/transformer/layer.0/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.0.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`

### Node 156: `Shape`
- **Name:** /distilbert/transformer/layer.1/attention/Shape
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Shape_output_0`

### Node 157: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 158: `Gather`
- **Name:** /distilbert/transformer/layer.1/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Shape_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 159: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_878`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/q_lin/MatMul_output_0`

### Node 160: `Add`
- **Name:** /distilbert/transformer/layer.1/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.attention.q_lin.bias`
  - `/distilbert/transformer/layer.1/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/q_lin/Add_output_0`

### Node 161: `Constant`
- **Name:** Constant_341
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_293`
- **Attributes:**
  - `value`: <Tensor: >

### Node 162: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.1/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Gather_output_0`
  - `onnx::Unsqueeze_293`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_output_0`

### Node 163: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 164: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 165: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 166: `Concat`
- **Name:** /distilbert/transformer/layer.1/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 167: `Constant`
- **Name:** Constant_347
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_302`
- **Attributes:**
  - `value`: <Tensor: >

### Node 168: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.1/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Gather_output_0`
  - `onnx::Unsqueeze_302`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_1_output_0`

### Node 169: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 170: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 171: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 172: `Concat`
- **Name:** /distilbert/transformer/layer.1/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 173: `Constant`
- **Name:** Constant_353
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_311`
- **Attributes:**
  - `value`: <Tensor: >

### Node 174: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.1/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Gather_output_0`
  - `onnx::Unsqueeze_311`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_2_output_0`

### Node 175: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 176: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 177: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 178: `Concat`
- **Name:** /distilbert/transformer/layer.1/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 179: `Reshape`
- **Name:** /distilbert/transformer/layer.1/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.1/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 180: `Transpose`
- **Name:** /distilbert/transformer/layer.1/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 181: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_888`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/k_lin/MatMul_output_0`

### Node 182: `Add`
- **Name:** /distilbert/transformer/layer.1/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.attention.k_lin.bias`
  - `/distilbert/transformer/layer.1/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/k_lin/Add_output_0`

### Node 183: `Reshape`
- **Name:** /distilbert/transformer/layer.1/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.1/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 184: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_889`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/v_lin/MatMul_output_0`

### Node 185: `Add`
- **Name:** /distilbert/transformer/layer.1/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.attention.v_lin.bias`
  - `/distilbert/transformer/layer.1/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/v_lin/Add_output_0`

### Node 186: `Reshape`
- **Name:** /distilbert/transformer/layer.1/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.1/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 187: `Transpose`
- **Name:** /distilbert/transformer/layer.1/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 188: `Shape`
- **Name:** /distilbert/transformer/layer.1/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Shape_1_output_0`

### Node 189: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 190: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 191: `Slice`
- **Name:** /distilbert/transformer/layer.1/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Slice_output_0`

### Node 192: `Cast`
- **Name:** /distilbert/transformer/layer.1/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 193: `Sqrt`
- **Name:** /distilbert/transformer/layer.1/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Sqrt_output_0`

### Node 194: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 195: `Div`
- **Name:** /distilbert/transformer/layer.1/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.1/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Div_output_0`

### Node 196: `Cast`
- **Name:** /distilbert/transformer/layer.1/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 197: `Transpose`
- **Name:** /distilbert/transformer/layer.1/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 198: `Sqrt`
- **Name:** /distilbert/transformer/layer.1/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Sqrt_1_output_0`

### Node 199: `Mul`
- **Name:** /distilbert/transformer/layer.1/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.1/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Mul_output_0`

### Node 200: `Sqrt`
- **Name:** /distilbert/transformer/layer.1/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Sqrt_2_output_0`

### Node 201: `Mul`
- **Name:** /distilbert/transformer/layer.1/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.1/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Mul_1_output_0`

### Node 202: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Mul_output_0`
  - `/distilbert/transformer/layer.1/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/MatMul_output_0`

### Node 203: `Add`
- **Name:** /distilbert/transformer/layer.1/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Add_output_0`

### Node 204: `Softmax`
- **Name:** /distilbert/transformer/layer.1/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 205: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.1/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/MatMul_1_output_0`

### Node 206: `Transpose`
- **Name:** /distilbert/transformer/layer.1/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 207: `Constant`
- **Name:** Constant_387
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_350`
- **Attributes:**
  - `value`: <Tensor: >

### Node 208: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.1/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Gather_output_0`
  - `onnx::Unsqueeze_350`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_3_output_0`

### Node 209: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 210: `Constant`
- **Name:** /distilbert/transformer/layer.1/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 211: `Concat`
- **Name:** /distilbert/transformer/layer.1/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.1/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 212: `Reshape`
- **Name:** /distilbert/transformer/layer.1/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.1/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 213: `MatMul`
- **Name:** /distilbert/transformer/layer.1/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/Reshape_3_output_0`
  - `onnx::MatMul_892`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/out_lin/MatMul_output_0`

### Node 214: `Add`
- **Name:** /distilbert/transformer/layer.1/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.attention.out_lin.bias`
  - `/distilbert/transformer/layer.1/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/attention/out_lin/Add_output_0`

### Node 215: `Add`
- **Name:** /distilbert/transformer/layer.1/Add
- **Inputs:**
  - `/distilbert/transformer/layer.1/attention/out_lin/Add_output_0`
  - `/distilbert/transformer/layer.0/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/Add_output_0`

### Node 216: `ReduceMean`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.1/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 217: `Sub`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.1/Add_output_0`
  - `/distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Sub_output_0`

### Node 218: `Constant`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 219: `Pow`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.1/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Pow_output_0`

### Node 220: `ReduceMean`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 221: `Constant`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 222: `Add`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.1/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Add_output_0`

### Node 223: `Sqrt`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Sqrt_output_0`

### Node 224: `Div`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.1/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Div_output_0`

### Node 225: `Mul`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.1.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Mul_output_0`

### Node 226: `Add`
- **Name:** /distilbert/transformer/layer.1/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.1.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Add_1_output_0`

### Node 227: `MatMul`
- **Name:** /distilbert/transformer/layer.1/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_893`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/lin1/MatMul_output_0`

### Node 228: `Add`
- **Name:** /distilbert/transformer/layer.1/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.ffn.lin1.bias`
  - `/distilbert/transformer/layer.1/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/lin1/Add_output_0`

### Node 229: `Constant`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 230: `Div`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Div_output_0`

### Node 231: `Erf`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Erf_output_0`

### Node 232: `Constant`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 233: `Add`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Add_output_0`

### Node 234: `Mul`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.1/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Mul_output_0`

### Node 235: `Constant`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 236: `Mul`
- **Name:** /distilbert/transformer/layer.1/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.1/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Mul_1_output_0`

### Node 237: `MatMul`
- **Name:** /distilbert/transformer/layer.1/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_894`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/lin2/MatMul_output_0`

### Node 238: `Add`
- **Name:** /distilbert/transformer/layer.1/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.1.ffn.lin2.bias`
  - `/distilbert/transformer/layer.1/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/ffn/lin2/Add_output_0`

### Node 239: `Add`
- **Name:** /distilbert/transformer/layer.1/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.1/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/Add_1_output_0`

### Node 240: `ReduceMean`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.1/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 241: `Sub`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.1/Add_1_output_0`
  - `/distilbert/transformer/layer.1/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Sub_output_0`

### Node 242: `Constant`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 243: `Pow`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.1/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Pow_output_0`

### Node 244: `ReduceMean`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 245: `Constant`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 246: `Add`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.1/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_output_0`

### Node 247: `Sqrt`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Sqrt_output_0`

### Node 248: `Div`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.1/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Div_output_0`

### Node 249: `Mul`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.1.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Mul_output_0`

### Node 250: `Add`
- **Name:** /distilbert/transformer/layer.1/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.1.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`

### Node 251: `Shape`
- **Name:** /distilbert/transformer/layer.2/attention/Shape
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Shape_output_0`

### Node 252: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 253: `Gather`
- **Name:** /distilbert/transformer/layer.2/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Shape_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 254: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_895`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/q_lin/MatMul_output_0`

### Node 255: `Add`
- **Name:** /distilbert/transformer/layer.2/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.attention.q_lin.bias`
  - `/distilbert/transformer/layer.2/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/q_lin/Add_output_0`

### Node 256: `Constant`
- **Name:** Constant_436
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_405`
- **Attributes:**
  - `value`: <Tensor: >

### Node 257: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.2/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Gather_output_0`
  - `onnx::Unsqueeze_405`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_output_0`

### Node 258: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 259: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 260: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 261: `Concat`
- **Name:** /distilbert/transformer/layer.2/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 262: `Constant`
- **Name:** Constant_442
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_414`
- **Attributes:**
  - `value`: <Tensor: >

### Node 263: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.2/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Gather_output_0`
  - `onnx::Unsqueeze_414`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_1_output_0`

### Node 264: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 265: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 266: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 267: `Concat`
- **Name:** /distilbert/transformer/layer.2/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 268: `Constant`
- **Name:** Constant_448
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_423`
- **Attributes:**
  - `value`: <Tensor: >

### Node 269: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.2/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Gather_output_0`
  - `onnx::Unsqueeze_423`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_2_output_0`

### Node 270: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 271: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 272: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 273: `Concat`
- **Name:** /distilbert/transformer/layer.2/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 274: `Reshape`
- **Name:** /distilbert/transformer/layer.2/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.2/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 275: `Transpose`
- **Name:** /distilbert/transformer/layer.2/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 276: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_905`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/k_lin/MatMul_output_0`

### Node 277: `Add`
- **Name:** /distilbert/transformer/layer.2/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.attention.k_lin.bias`
  - `/distilbert/transformer/layer.2/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/k_lin/Add_output_0`

### Node 278: `Reshape`
- **Name:** /distilbert/transformer/layer.2/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.2/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 279: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_906`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/v_lin/MatMul_output_0`

### Node 280: `Add`
- **Name:** /distilbert/transformer/layer.2/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.attention.v_lin.bias`
  - `/distilbert/transformer/layer.2/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/v_lin/Add_output_0`

### Node 281: `Reshape`
- **Name:** /distilbert/transformer/layer.2/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.2/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 282: `Transpose`
- **Name:** /distilbert/transformer/layer.2/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 283: `Shape`
- **Name:** /distilbert/transformer/layer.2/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Shape_1_output_0`

### Node 284: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 285: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 286: `Slice`
- **Name:** /distilbert/transformer/layer.2/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Slice_output_0`

### Node 287: `Cast`
- **Name:** /distilbert/transformer/layer.2/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 288: `Sqrt`
- **Name:** /distilbert/transformer/layer.2/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Sqrt_output_0`

### Node 289: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 290: `Div`
- **Name:** /distilbert/transformer/layer.2/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.2/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Div_output_0`

### Node 291: `Cast`
- **Name:** /distilbert/transformer/layer.2/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 292: `Transpose`
- **Name:** /distilbert/transformer/layer.2/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 293: `Sqrt`
- **Name:** /distilbert/transformer/layer.2/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Sqrt_1_output_0`

### Node 294: `Mul`
- **Name:** /distilbert/transformer/layer.2/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.2/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Mul_output_0`

### Node 295: `Sqrt`
- **Name:** /distilbert/transformer/layer.2/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Sqrt_2_output_0`

### Node 296: `Mul`
- **Name:** /distilbert/transformer/layer.2/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.2/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Mul_1_output_0`

### Node 297: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Mul_output_0`
  - `/distilbert/transformer/layer.2/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/MatMul_output_0`

### Node 298: `Add`
- **Name:** /distilbert/transformer/layer.2/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Add_output_0`

### Node 299: `Softmax`
- **Name:** /distilbert/transformer/layer.2/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 300: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.2/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/MatMul_1_output_0`

### Node 301: `Transpose`
- **Name:** /distilbert/transformer/layer.2/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 302: `Constant`
- **Name:** Constant_482
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_462`
- **Attributes:**
  - `value`: <Tensor: >

### Node 303: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.2/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Gather_output_0`
  - `onnx::Unsqueeze_462`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_3_output_0`

### Node 304: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 305: `Constant`
- **Name:** /distilbert/transformer/layer.2/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 306: `Concat`
- **Name:** /distilbert/transformer/layer.2/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.2/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 307: `Reshape`
- **Name:** /distilbert/transformer/layer.2/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.2/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 308: `MatMul`
- **Name:** /distilbert/transformer/layer.2/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/Reshape_3_output_0`
  - `onnx::MatMul_909`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/out_lin/MatMul_output_0`

### Node 309: `Add`
- **Name:** /distilbert/transformer/layer.2/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.attention.out_lin.bias`
  - `/distilbert/transformer/layer.2/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/attention/out_lin/Add_output_0`

### Node 310: `Add`
- **Name:** /distilbert/transformer/layer.2/Add
- **Inputs:**
  - `/distilbert/transformer/layer.2/attention/out_lin/Add_output_0`
  - `/distilbert/transformer/layer.1/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/Add_output_0`

### Node 311: `ReduceMean`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.2/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 312: `Sub`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.2/Add_output_0`
  - `/distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Sub_output_0`

### Node 313: `Constant`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 314: `Pow`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.2/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Pow_output_0`

### Node 315: `ReduceMean`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 316: `Constant`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 317: `Add`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.2/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Add_output_0`

### Node 318: `Sqrt`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Sqrt_output_0`

### Node 319: `Div`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.2/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Div_output_0`

### Node 320: `Mul`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.2.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Mul_output_0`

### Node 321: `Add`
- **Name:** /distilbert/transformer/layer.2/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.2.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Add_1_output_0`

### Node 322: `MatMul`
- **Name:** /distilbert/transformer/layer.2/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_910`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/lin1/MatMul_output_0`

### Node 323: `Add`
- **Name:** /distilbert/transformer/layer.2/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.ffn.lin1.bias`
  - `/distilbert/transformer/layer.2/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/lin1/Add_output_0`

### Node 324: `Constant`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 325: `Div`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Div_output_0`

### Node 326: `Erf`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Erf_output_0`

### Node 327: `Constant`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 328: `Add`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Add_output_0`

### Node 329: `Mul`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.2/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Mul_output_0`

### Node 330: `Constant`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 331: `Mul`
- **Name:** /distilbert/transformer/layer.2/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.2/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Mul_1_output_0`

### Node 332: `MatMul`
- **Name:** /distilbert/transformer/layer.2/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_911`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/lin2/MatMul_output_0`

### Node 333: `Add`
- **Name:** /distilbert/transformer/layer.2/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.2.ffn.lin2.bias`
  - `/distilbert/transformer/layer.2/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/ffn/lin2/Add_output_0`

### Node 334: `Add`
- **Name:** /distilbert/transformer/layer.2/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.2/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/Add_1_output_0`

### Node 335: `ReduceMean`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.2/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 336: `Sub`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.2/Add_1_output_0`
  - `/distilbert/transformer/layer.2/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Sub_output_0`

### Node 337: `Constant`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 338: `Pow`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.2/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Pow_output_0`

### Node 339: `ReduceMean`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 340: `Constant`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 341: `Add`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.2/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_output_0`

### Node 342: `Sqrt`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Sqrt_output_0`

### Node 343: `Div`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.2/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Div_output_0`

### Node 344: `Mul`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.2.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Mul_output_0`

### Node 345: `Add`
- **Name:** /distilbert/transformer/layer.2/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.2.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`

### Node 346: `Shape`
- **Name:** /distilbert/transformer/layer.3/attention/Shape
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Shape_output_0`

### Node 347: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 348: `Gather`
- **Name:** /distilbert/transformer/layer.3/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Shape_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 349: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_912`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/q_lin/MatMul_output_0`

### Node 350: `Add`
- **Name:** /distilbert/transformer/layer.3/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.attention.q_lin.bias`
  - `/distilbert/transformer/layer.3/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/q_lin/Add_output_0`

### Node 351: `Constant`
- **Name:** Constant_531
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_517`
- **Attributes:**
  - `value`: <Tensor: >

### Node 352: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.3/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Gather_output_0`
  - `onnx::Unsqueeze_517`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_output_0`

### Node 353: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 354: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 355: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 356: `Concat`
- **Name:** /distilbert/transformer/layer.3/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 357: `Constant`
- **Name:** Constant_537
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_526`
- **Attributes:**
  - `value`: <Tensor: >

### Node 358: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.3/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Gather_output_0`
  - `onnx::Unsqueeze_526`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_1_output_0`

### Node 359: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 360: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 361: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 362: `Concat`
- **Name:** /distilbert/transformer/layer.3/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 363: `Constant`
- **Name:** Constant_543
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_535`
- **Attributes:**
  - `value`: <Tensor: >

### Node 364: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.3/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Gather_output_0`
  - `onnx::Unsqueeze_535`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_2_output_0`

### Node 365: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 366: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 367: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 368: `Concat`
- **Name:** /distilbert/transformer/layer.3/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 369: `Reshape`
- **Name:** /distilbert/transformer/layer.3/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.3/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 370: `Transpose`
- **Name:** /distilbert/transformer/layer.3/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 371: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_922`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/k_lin/MatMul_output_0`

### Node 372: `Add`
- **Name:** /distilbert/transformer/layer.3/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.attention.k_lin.bias`
  - `/distilbert/transformer/layer.3/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/k_lin/Add_output_0`

### Node 373: `Reshape`
- **Name:** /distilbert/transformer/layer.3/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.3/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 374: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_923`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/v_lin/MatMul_output_0`

### Node 375: `Add`
- **Name:** /distilbert/transformer/layer.3/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.attention.v_lin.bias`
  - `/distilbert/transformer/layer.3/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/v_lin/Add_output_0`

### Node 376: `Reshape`
- **Name:** /distilbert/transformer/layer.3/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.3/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 377: `Transpose`
- **Name:** /distilbert/transformer/layer.3/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 378: `Shape`
- **Name:** /distilbert/transformer/layer.3/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Shape_1_output_0`

### Node 379: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 380: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 381: `Slice`
- **Name:** /distilbert/transformer/layer.3/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Slice_output_0`

### Node 382: `Cast`
- **Name:** /distilbert/transformer/layer.3/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 383: `Sqrt`
- **Name:** /distilbert/transformer/layer.3/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Sqrt_output_0`

### Node 384: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 385: `Div`
- **Name:** /distilbert/transformer/layer.3/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.3/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Div_output_0`

### Node 386: `Cast`
- **Name:** /distilbert/transformer/layer.3/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 387: `Transpose`
- **Name:** /distilbert/transformer/layer.3/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 388: `Sqrt`
- **Name:** /distilbert/transformer/layer.3/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Sqrt_1_output_0`

### Node 389: `Mul`
- **Name:** /distilbert/transformer/layer.3/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.3/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Mul_output_0`

### Node 390: `Sqrt`
- **Name:** /distilbert/transformer/layer.3/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Sqrt_2_output_0`

### Node 391: `Mul`
- **Name:** /distilbert/transformer/layer.3/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.3/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Mul_1_output_0`

### Node 392: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Mul_output_0`
  - `/distilbert/transformer/layer.3/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/MatMul_output_0`

### Node 393: `Add`
- **Name:** /distilbert/transformer/layer.3/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Add_output_0`

### Node 394: `Softmax`
- **Name:** /distilbert/transformer/layer.3/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 395: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.3/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/MatMul_1_output_0`

### Node 396: `Transpose`
- **Name:** /distilbert/transformer/layer.3/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 397: `Constant`
- **Name:** Constant_577
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_574`
- **Attributes:**
  - `value`: <Tensor: >

### Node 398: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.3/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Gather_output_0`
  - `onnx::Unsqueeze_574`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_3_output_0`

### Node 399: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 400: `Constant`
- **Name:** /distilbert/transformer/layer.3/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 401: `Concat`
- **Name:** /distilbert/transformer/layer.3/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.3/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 402: `Reshape`
- **Name:** /distilbert/transformer/layer.3/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.3/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 403: `MatMul`
- **Name:** /distilbert/transformer/layer.3/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/Reshape_3_output_0`
  - `onnx::MatMul_926`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/out_lin/MatMul_output_0`

### Node 404: `Add`
- **Name:** /distilbert/transformer/layer.3/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.attention.out_lin.bias`
  - `/distilbert/transformer/layer.3/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/attention/out_lin/Add_output_0`

### Node 405: `Add`
- **Name:** /distilbert/transformer/layer.3/Add
- **Inputs:**
  - `/distilbert/transformer/layer.3/attention/out_lin/Add_output_0`
  - `/distilbert/transformer/layer.2/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/Add_output_0`

### Node 406: `ReduceMean`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.3/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 407: `Sub`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.3/Add_output_0`
  - `/distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Sub_output_0`

### Node 408: `Constant`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 409: `Pow`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.3/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Pow_output_0`

### Node 410: `ReduceMean`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 411: `Constant`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 412: `Add`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.3/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Add_output_0`

### Node 413: `Sqrt`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Sqrt_output_0`

### Node 414: `Div`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.3/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Div_output_0`

### Node 415: `Mul`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.3.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Mul_output_0`

### Node 416: `Add`
- **Name:** /distilbert/transformer/layer.3/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.3.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Add_1_output_0`

### Node 417: `MatMul`
- **Name:** /distilbert/transformer/layer.3/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_927`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/lin1/MatMul_output_0`

### Node 418: `Add`
- **Name:** /distilbert/transformer/layer.3/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.ffn.lin1.bias`
  - `/distilbert/transformer/layer.3/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/lin1/Add_output_0`

### Node 419: `Constant`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 420: `Div`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Div_output_0`

### Node 421: `Erf`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Erf_output_0`

### Node 422: `Constant`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 423: `Add`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Add_output_0`

### Node 424: `Mul`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.3/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Mul_output_0`

### Node 425: `Constant`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 426: `Mul`
- **Name:** /distilbert/transformer/layer.3/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.3/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Mul_1_output_0`

### Node 427: `MatMul`
- **Name:** /distilbert/transformer/layer.3/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_928`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/lin2/MatMul_output_0`

### Node 428: `Add`
- **Name:** /distilbert/transformer/layer.3/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.3.ffn.lin2.bias`
  - `/distilbert/transformer/layer.3/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/ffn/lin2/Add_output_0`

### Node 429: `Add`
- **Name:** /distilbert/transformer/layer.3/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.3/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/Add_1_output_0`

### Node 430: `ReduceMean`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.3/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 431: `Sub`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.3/Add_1_output_0`
  - `/distilbert/transformer/layer.3/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Sub_output_0`

### Node 432: `Constant`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 433: `Pow`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.3/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Pow_output_0`

### Node 434: `ReduceMean`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 435: `Constant`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 436: `Add`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.3/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_output_0`

### Node 437: `Sqrt`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Sqrt_output_0`

### Node 438: `Div`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.3/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Div_output_0`

### Node 439: `Mul`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.3.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Mul_output_0`

### Node 440: `Add`
- **Name:** /distilbert/transformer/layer.3/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.3.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`

### Node 441: `Shape`
- **Name:** /distilbert/transformer/layer.4/attention/Shape
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Shape_output_0`

### Node 442: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 443: `Gather`
- **Name:** /distilbert/transformer/layer.4/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Shape_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 444: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_929`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/q_lin/MatMul_output_0`

### Node 445: `Add`
- **Name:** /distilbert/transformer/layer.4/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.attention.q_lin.bias`
  - `/distilbert/transformer/layer.4/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/q_lin/Add_output_0`

### Node 446: `Constant`
- **Name:** Constant_626
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_629`
- **Attributes:**
  - `value`: <Tensor: >

### Node 447: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.4/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Gather_output_0`
  - `onnx::Unsqueeze_629`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_output_0`

### Node 448: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 449: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 450: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 451: `Concat`
- **Name:** /distilbert/transformer/layer.4/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 452: `Constant`
- **Name:** Constant_632
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_638`
- **Attributes:**
  - `value`: <Tensor: >

### Node 453: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.4/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Gather_output_0`
  - `onnx::Unsqueeze_638`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_1_output_0`

### Node 454: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 455: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 456: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 457: `Concat`
- **Name:** /distilbert/transformer/layer.4/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 458: `Constant`
- **Name:** Constant_638
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_647`
- **Attributes:**
  - `value`: <Tensor: >

### Node 459: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.4/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Gather_output_0`
  - `onnx::Unsqueeze_647`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_2_output_0`

### Node 460: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 461: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 462: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 463: `Concat`
- **Name:** /distilbert/transformer/layer.4/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 464: `Reshape`
- **Name:** /distilbert/transformer/layer.4/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.4/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 465: `Transpose`
- **Name:** /distilbert/transformer/layer.4/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 466: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_939`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/k_lin/MatMul_output_0`

### Node 467: `Add`
- **Name:** /distilbert/transformer/layer.4/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.attention.k_lin.bias`
  - `/distilbert/transformer/layer.4/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/k_lin/Add_output_0`

### Node 468: `Reshape`
- **Name:** /distilbert/transformer/layer.4/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.4/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 469: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_940`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/v_lin/MatMul_output_0`

### Node 470: `Add`
- **Name:** /distilbert/transformer/layer.4/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.attention.v_lin.bias`
  - `/distilbert/transformer/layer.4/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/v_lin/Add_output_0`

### Node 471: `Reshape`
- **Name:** /distilbert/transformer/layer.4/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.4/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 472: `Transpose`
- **Name:** /distilbert/transformer/layer.4/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 473: `Shape`
- **Name:** /distilbert/transformer/layer.4/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Shape_1_output_0`

### Node 474: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 475: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 476: `Slice`
- **Name:** /distilbert/transformer/layer.4/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Slice_output_0`

### Node 477: `Cast`
- **Name:** /distilbert/transformer/layer.4/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 478: `Sqrt`
- **Name:** /distilbert/transformer/layer.4/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Sqrt_output_0`

### Node 479: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 480: `Div`
- **Name:** /distilbert/transformer/layer.4/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.4/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Div_output_0`

### Node 481: `Cast`
- **Name:** /distilbert/transformer/layer.4/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 482: `Transpose`
- **Name:** /distilbert/transformer/layer.4/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 483: `Sqrt`
- **Name:** /distilbert/transformer/layer.4/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Sqrt_1_output_0`

### Node 484: `Mul`
- **Name:** /distilbert/transformer/layer.4/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.4/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Mul_output_0`

### Node 485: `Sqrt`
- **Name:** /distilbert/transformer/layer.4/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Sqrt_2_output_0`

### Node 486: `Mul`
- **Name:** /distilbert/transformer/layer.4/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.4/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Mul_1_output_0`

### Node 487: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Mul_output_0`
  - `/distilbert/transformer/layer.4/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/MatMul_output_0`

### Node 488: `Add`
- **Name:** /distilbert/transformer/layer.4/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Add_output_0`

### Node 489: `Softmax`
- **Name:** /distilbert/transformer/layer.4/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 490: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.4/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/MatMul_1_output_0`

### Node 491: `Transpose`
- **Name:** /distilbert/transformer/layer.4/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 492: `Constant`
- **Name:** Constant_672
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_686`
- **Attributes:**
  - `value`: <Tensor: >

### Node 493: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.4/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Gather_output_0`
  - `onnx::Unsqueeze_686`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_3_output_0`

### Node 494: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 495: `Constant`
- **Name:** /distilbert/transformer/layer.4/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 496: `Concat`
- **Name:** /distilbert/transformer/layer.4/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.4/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 497: `Reshape`
- **Name:** /distilbert/transformer/layer.4/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.4/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 498: `MatMul`
- **Name:** /distilbert/transformer/layer.4/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/Reshape_3_output_0`
  - `onnx::MatMul_943`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/out_lin/MatMul_output_0`

### Node 499: `Add`
- **Name:** /distilbert/transformer/layer.4/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.attention.out_lin.bias`
  - `/distilbert/transformer/layer.4/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/attention/out_lin/Add_output_0`

### Node 500: `Add`
- **Name:** /distilbert/transformer/layer.4/Add
- **Inputs:**
  - `/distilbert/transformer/layer.4/attention/out_lin/Add_output_0`
  - `/distilbert/transformer/layer.3/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/Add_output_0`

### Node 501: `ReduceMean`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.4/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 502: `Sub`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.4/Add_output_0`
  - `/distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Sub_output_0`

### Node 503: `Constant`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 504: `Pow`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.4/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Pow_output_0`

### Node 505: `ReduceMean`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 506: `Constant`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 507: `Add`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.4/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Add_output_0`

### Node 508: `Sqrt`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Sqrt_output_0`

### Node 509: `Div`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.4/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Div_output_0`

### Node 510: `Mul`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.4.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Mul_output_0`

### Node 511: `Add`
- **Name:** /distilbert/transformer/layer.4/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.4.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Add_1_output_0`

### Node 512: `MatMul`
- **Name:** /distilbert/transformer/layer.4/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_944`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/lin1/MatMul_output_0`

### Node 513: `Add`
- **Name:** /distilbert/transformer/layer.4/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.ffn.lin1.bias`
  - `/distilbert/transformer/layer.4/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/lin1/Add_output_0`

### Node 514: `Constant`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 515: `Div`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Div_output_0`

### Node 516: `Erf`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Erf_output_0`

### Node 517: `Constant`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 518: `Add`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Add_output_0`

### Node 519: `Mul`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.4/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Mul_output_0`

### Node 520: `Constant`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 521: `Mul`
- **Name:** /distilbert/transformer/layer.4/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.4/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Mul_1_output_0`

### Node 522: `MatMul`
- **Name:** /distilbert/transformer/layer.4/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_945`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/lin2/MatMul_output_0`

### Node 523: `Add`
- **Name:** /distilbert/transformer/layer.4/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.4.ffn.lin2.bias`
  - `/distilbert/transformer/layer.4/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/ffn/lin2/Add_output_0`

### Node 524: `Add`
- **Name:** /distilbert/transformer/layer.4/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.4/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/Add_1_output_0`

### Node 525: `ReduceMean`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.4/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 526: `Sub`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.4/Add_1_output_0`
  - `/distilbert/transformer/layer.4/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Sub_output_0`

### Node 527: `Constant`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 528: `Pow`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.4/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Pow_output_0`

### Node 529: `ReduceMean`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 530: `Constant`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 531: `Add`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.4/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_output_0`

### Node 532: `Sqrt`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Sqrt_output_0`

### Node 533: `Div`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.4/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Div_output_0`

### Node 534: `Mul`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.4.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Mul_output_0`

### Node 535: `Add`
- **Name:** /distilbert/transformer/layer.4/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.4.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`

### Node 536: `Shape`
- **Name:** /distilbert/transformer/layer.5/attention/Shape
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Shape_output_0`

### Node 537: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 538: `Gather`
- **Name:** /distilbert/transformer/layer.5/attention/Gather
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Shape_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Gather_output_0`
- **Attributes:**
  - `axis`: 0

### Node 539: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/q_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_946`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/q_lin/MatMul_output_0`

### Node 540: `Add`
- **Name:** /distilbert/transformer/layer.5/attention/q_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.attention.q_lin.bias`
  - `/distilbert/transformer/layer.5/attention/q_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/q_lin/Add_output_0`

### Node 541: `Constant`
- **Name:** Constant_721
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_741`
- **Attributes:**
  - `value`: <Tensor: >

### Node 542: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.5/attention/Unsqueeze
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Gather_output_0`
  - `onnx::Unsqueeze_741`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_output_0`

### Node 543: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 544: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 545: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_3
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_3_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 546: `Concat`
- **Name:** /distilbert/transformer/layer.5/attention/Concat
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_1_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_2_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Concat_output_0`
- **Attributes:**
  - `axis`: 0

### Node 547: `Constant`
- **Name:** Constant_727
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_750`
- **Attributes:**
  - `value`: <Tensor: >

### Node 548: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.5/attention/Unsqueeze_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Gather_output_0`
  - `onnx::Unsqueeze_750`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_1_output_0`

### Node 549: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_4
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_4_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 550: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_5
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_5_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 551: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_6
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_6_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 552: `Concat`
- **Name:** /distilbert/transformer/layer.5/attention/Concat_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_1_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_4_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_5_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_6_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Concat_1_output_0`
- **Attributes:**
  - `axis`: 0

### Node 553: `Constant`
- **Name:** Constant_733
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_759`
- **Attributes:**
  - `value`: <Tensor: >

### Node 554: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.5/attention/Unsqueeze_2
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Gather_output_0`
  - `onnx::Unsqueeze_759`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_2_output_0`

### Node 555: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_7
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_7_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 556: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_8
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_8_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 557: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_9
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_9_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 558: `Concat`
- **Name:** /distilbert/transformer/layer.5/attention/Concat_2
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_2_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_7_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_8_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_9_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Concat_2_output_0`
- **Attributes:**
  - `axis`: 0

### Node 559: `Reshape`
- **Name:** /distilbert/transformer/layer.5/attention/Reshape
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/q_lin/Add_output_0`
  - `/distilbert/transformer/layer.5/attention/Concat_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 560: `Transpose`
- **Name:** /distilbert/transformer/layer.5/attention/Transpose
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 561: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/k_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_956`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/k_lin/MatMul_output_0`

### Node 562: `Add`
- **Name:** /distilbert/transformer/layer.5/attention/k_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.attention.k_lin.bias`
  - `/distilbert/transformer/layer.5/attention/k_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/k_lin/Add_output_0`

### Node 563: `Reshape`
- **Name:** /distilbert/transformer/layer.5/attention/Reshape_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/k_lin/Add_output_0`
  - `/distilbert/transformer/layer.5/attention/Concat_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_1_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 564: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/v_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_957`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/v_lin/MatMul_output_0`

### Node 565: `Add`
- **Name:** /distilbert/transformer/layer.5/attention/v_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.attention.v_lin.bias`
  - `/distilbert/transformer/layer.5/attention/v_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/v_lin/Add_output_0`

### Node 566: `Reshape`
- **Name:** /distilbert/transformer/layer.5/attention/Reshape_2
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/v_lin/Add_output_0`
  - `/distilbert/transformer/layer.5/attention/Concat_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_2_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 567: `Transpose`
- **Name:** /distilbert/transformer/layer.5/attention/Transpose_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_1_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 568: `Shape`
- **Name:** /distilbert/transformer/layer.5/attention/Shape_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Shape_1_output_0`

### Node 569: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_10
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_10_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 570: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_11
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_11_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 571: `Slice`
- **Name:** /distilbert/transformer/layer.5/attention/Slice
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Shape_1_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_10_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_11_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Slice_output_0`

### Node 572: `Cast`
- **Name:** /distilbert/transformer/layer.5/attention/Cast
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Slice_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Cast_output_0`
- **Attributes:**
  - `to`: 1

### Node 573: `Sqrt`
- **Name:** /distilbert/transformer/layer.5/attention/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Cast_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Sqrt_output_0`

### Node 574: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_12
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_12_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 575: `Div`
- **Name:** /distilbert/transformer/layer.5/attention/Div
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_12_output_0`
  - `/distilbert/transformer/layer.5/attention/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Div_output_0`

### Node 576: `Cast`
- **Name:** /distilbert/transformer/layer.5/attention/Cast_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Cast_1_output_0`
- **Attributes:**
  - `to`: 1

### Node 577: `Transpose`
- **Name:** /distilbert/transformer/layer.5/attention/Transpose_2
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_2_output_0`
- **Attributes:**
  - `perm`: [0, 2, 3, 1]

### Node 578: `Sqrt`
- **Name:** /distilbert/transformer/layer.5/attention/Sqrt_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Sqrt_1_output_0`

### Node 579: `Mul`
- **Name:** /distilbert/transformer/layer.5/attention/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_output_0`
  - `/distilbert/transformer/layer.5/attention/Sqrt_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Mul_output_0`

### Node 580: `Sqrt`
- **Name:** /distilbert/transformer/layer.5/attention/Sqrt_2
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Cast_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Sqrt_2_output_0`

### Node 581: `Mul`
- **Name:** /distilbert/transformer/layer.5/attention/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_2_output_0`
  - `/distilbert/transformer/layer.5/attention/Sqrt_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Mul_1_output_0`

### Node 582: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Mul_output_0`
  - `/distilbert/transformer/layer.5/attention/Mul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/MatMul_output_0`

### Node 583: `Add`
- **Name:** /distilbert/transformer/layer.5/attention/Add
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/MatMul_output_0`
  - `/distilbert/Where_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Add_output_0`

### Node 584: `Softmax`
- **Name:** /distilbert/transformer/layer.5/attention/Softmax
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Softmax_output_0`
- **Attributes:**
  - `axis`: -1

### Node 585: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/MatMul_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Softmax_output_0`
  - `/distilbert/transformer/layer.5/attention/Transpose_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/MatMul_1_output_0`

### Node 586: `Transpose`
- **Name:** /distilbert/transformer/layer.5/attention/Transpose_3
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/MatMul_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_3_output_0`
- **Attributes:**
  - `perm`: [0, 2, 1, 3]

### Node 587: `Constant`
- **Name:** Constant_767
- **Inputs:**
- **Outputs:**
  - `onnx::Unsqueeze_798`
- **Attributes:**
  - `value`: <Tensor: >

### Node 588: `Unsqueeze`
- **Name:** /distilbert/transformer/layer.5/attention/Unsqueeze_3
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Gather_output_0`
  - `onnx::Unsqueeze_798`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_3_output_0`

### Node 589: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_13
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_13_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 590: `Constant`
- **Name:** /distilbert/transformer/layer.5/attention/Constant_14
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Constant_14_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 591: `Concat`
- **Name:** /distilbert/transformer/layer.5/attention/Concat_3
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Unsqueeze_3_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_13_output_0`
  - `/distilbert/transformer/layer.5/attention/Constant_14_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Concat_3_output_0`
- **Attributes:**
  - `axis`: 0

### Node 592: `Reshape`
- **Name:** /distilbert/transformer/layer.5/attention/Reshape_3
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Transpose_3_output_0`
  - `/distilbert/transformer/layer.5/attention/Concat_3_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_3_output_0`
- **Attributes:**
  - `allowzero`: 0

### Node 593: `MatMul`
- **Name:** /distilbert/transformer/layer.5/attention/out_lin/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/Reshape_3_output_0`
  - `onnx::MatMul_960`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/out_lin/MatMul_output_0`

### Node 594: `Add`
- **Name:** /distilbert/transformer/layer.5/attention/out_lin/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.attention.out_lin.bias`
  - `/distilbert/transformer/layer.5/attention/out_lin/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/attention/out_lin/Add_output_0`

### Node 595: `Add`
- **Name:** /distilbert/transformer/layer.5/Add
- **Inputs:**
  - `/distilbert/transformer/layer.5/attention/out_lin/Add_output_0`
  - `/distilbert/transformer/layer.4/output_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/Add_output_0`

### Node 596: `ReduceMean`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.5/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 597: `Sub`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.5/Add_output_0`
  - `/distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Sub_output_0`

### Node 598: `Constant`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 599: `Pow`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.5/sa_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Pow_output_0`

### Node 600: `ReduceMean`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 601: `Constant`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 602: `Add`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.5/sa_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Add_output_0`

### Node 603: `Sqrt`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Sqrt_output_0`

### Node 604: `Div`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.5/sa_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Div_output_0`

### Node 605: `Mul`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.5.sa_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Mul_output_0`

### Node 606: `Add`
- **Name:** /distilbert/transformer/layer.5/sa_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.5.sa_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Add_1_output_0`

### Node 607: `MatMul`
- **Name:** /distilbert/transformer/layer.5/ffn/lin1/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.5/sa_layer_norm/Add_1_output_0`
  - `onnx::MatMul_961`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/lin1/MatMul_output_0`

### Node 608: `Add`
- **Name:** /distilbert/transformer/layer.5/ffn/lin1/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.ffn.lin1.bias`
  - `/distilbert/transformer/layer.5/ffn/lin1/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/lin1/Add_output_0`

### Node 609: `Constant`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 610: `Div`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Div
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Div_output_0`

### Node 611: `Erf`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Erf
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Div_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Erf_output_0`

### Node 612: `Constant`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 613: `Add`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Add
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Erf_output_0`
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Add_output_0`

### Node 614: `Mul`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/lin1/Add_output_0`
  - `/distilbert/transformer/layer.5/ffn/activation/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Mul_output_0`

### Node 615: `Constant`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Constant_2
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 616: `Mul`
- **Name:** /distilbert/transformer/layer.5/ffn/activation/Mul_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Mul_output_0`
  - `/distilbert/transformer/layer.5/ffn/activation/Constant_2_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Mul_1_output_0`

### Node 617: `MatMul`
- **Name:** /distilbert/transformer/layer.5/ffn/lin2/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/activation/Mul_1_output_0`
  - `onnx::MatMul_962`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/lin2/MatMul_output_0`

### Node 618: `Add`
- **Name:** /distilbert/transformer/layer.5/ffn/lin2/Add
- **Inputs:**
  - `distilbert.transformer.layer.5.ffn.lin2.bias`
  - `/distilbert/transformer/layer.5/ffn/lin2/MatMul_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/ffn/lin2/Add_output_0`

### Node 619: `Add`
- **Name:** /distilbert/transformer/layer.5/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/ffn/lin2/Add_output_0`
  - `/distilbert/transformer/layer.5/sa_layer_norm/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/Add_1_output_0`

### Node 620: `ReduceMean`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/ReduceMean
- **Inputs:**
  - `/distilbert/transformer/layer.5/Add_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/ReduceMean_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 621: `Sub`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Sub
- **Inputs:**
  - `/distilbert/transformer/layer.5/Add_1_output_0`
  - `/distilbert/transformer/layer.5/output_layer_norm/ReduceMean_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Sub_output_0`

### Node 622: `Constant`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Constant
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 623: `Pow`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Pow
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.5/output_layer_norm/Constant_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Pow_output_0`

### Node 624: `ReduceMean`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/ReduceMean_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Pow_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/ReduceMean_1_output_0`
- **Attributes:**
  - `axes`: [-1]

### Node 625: `Constant`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Constant_1
- **Inputs:**
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 626: `Add`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Add
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/ReduceMean_1_output_0`
  - `/distilbert/transformer/layer.5/output_layer_norm/Constant_1_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Add_output_0`

### Node 627: `Sqrt`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Sqrt
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Add_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Sqrt_output_0`

### Node 628: `Div`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Div
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Sub_output_0`
  - `/distilbert/transformer/layer.5/output_layer_norm/Sqrt_output_0`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Div_output_0`

### Node 629: `Mul`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Mul
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Div_output_0`
  - `distilbert.transformer.layer.5.output_layer_norm.weight`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Mul_output_0`

### Node 630: `Add`
- **Name:** /distilbert/transformer/layer.5/output_layer_norm/Add_1
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Mul_output_0`
  - `distilbert.transformer.layer.5.output_layer_norm.bias`
- **Outputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Add_1_output_0`

### Node 631: `MatMul`
- **Name:** /qa_outputs/MatMul
- **Inputs:**
  - `/distilbert/transformer/layer.5/output_layer_norm/Add_1_output_0`
  - `onnx::MatMul_963`
- **Outputs:**
  - `/qa_outputs/MatMul_output_0`

### Node 632: `Add`
- **Name:** /qa_outputs/Add
- **Inputs:**
  - `qa_outputs.bias`
  - `/qa_outputs/MatMul_output_0`
- **Outputs:**
  - `/qa_outputs/Add_output_0`

### Node 633: `Constant`
- **Name:** /Constant
- **Inputs:**
- **Outputs:**
  - `/Constant_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 634: `Split`
- **Name:** /Split
- **Inputs:**
  - `/qa_outputs/Add_output_0`
  - `/Constant_output_0`
- **Outputs:**
  - `/Split_output_0`
  - `/Split_output_1`
- **Attributes:**
  - `axis`: -1

### Node 635: `Constant`
- **Name:** /Constant_1
- **Inputs:**
- **Outputs:**
  - `/Constant_1_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 636: `Squeeze`
- **Name:** /Squeeze
- **Inputs:**
  - `/Split_output_0`
  - `/Constant_1_output_0`
- **Outputs:**
  - `start_logits`

### Node 637: `Constant`
- **Name:** /Constant_2
- **Inputs:**
- **Outputs:**
  - `/Constant_2_output_0`
- **Attributes:**
  - `value`: <Tensor: >

### Node 638: `Squeeze`
- **Name:** /Squeeze_1
- **Inputs:**
  - `/Split_output_1`
  - `/Constant_2_output_0`
- **Outputs:**
  - `end_logits`

