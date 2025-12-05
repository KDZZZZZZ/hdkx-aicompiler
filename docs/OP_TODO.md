============================================================
Analysis Result for model.onnx
============================================================

Operator: Add
  Count: 88
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Cast
  Count: 15
  Inputs: [1]
  Outputs: [1]
  Attributes:
    - to: 1 (Example)
----------------------------------------
Operator: Concat
  Count: 25
  Inputs: [3, 4]
  Outputs: [1]
  Attributes:
    - axis: 0 (Example)
----------------------------------------
Operator: Constant
  Count: 180
  Inputs: [0]
  Outputs: [1]
  Attributes:
    - value: Tensor([]) (Example)
----------------------------------------
Operator: ConstantOfShape
  Count: 1
  Inputs: [1]
  Outputs: [1]
  Attributes:
    - value: Tensor([1]) (Example)
----------------------------------------
Operator: Div
  Count: 25
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Equal
  Count: 1
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Erf
  Count: 6
  Inputs: [1]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Expand
  Count: 1
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Gather
  Count: 12
  Inputs: [2]
  Outputs: [1]
  Attributes:
    - axis: 0 (Example)
----------------------------------------
Operator: MatMul
  Count: 49
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Mul
  Count: 38
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Pow
  Count: 13
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: ReduceMean
  Count: 26
  Inputs: [1]
  Outputs: [1]
  Attributes:
    - axes: [-1] (Example)
----------------------------------------
Operator: Reshape
  Count: 25
  Inputs: [2]
  Outputs: [1]
  Attributes:
    - allowzero: 0 (Example)
----------------------------------------
Operator: Shape
  Count: 17
  Inputs: [1]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Slice
  Count: 7
  Inputs: [3, 5]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Softmax
  Count: 6
  Inputs: [1]
  Outputs: [1]
  Attributes:
    - axis: -1 (Example)
----------------------------------------
Operator: Split
  Count: 1
  Inputs: [2]
  Outputs: [2]
  Attributes:
    - axis: -1 (Example)
----------------------------------------
Operator: Sqrt
  Count: 31
  Inputs: [1]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Squeeze
  Count: 2
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Sub
  Count: 14
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Transpose
  Count: 24
  Inputs: [1]
  Outputs: [1]
  Attributes:
    - perm: [0, 2, 1, 3] (Example)
----------------------------------------
Operator: Unsqueeze
  Count: 30
  Inputs: [2]
  Outputs: [1]
  Attributes: None
----------------------------------------
Operator: Where
  Count: 2
  Inputs: [3]
  Outputs: [1]
  Attributes: None
----------------------------------------