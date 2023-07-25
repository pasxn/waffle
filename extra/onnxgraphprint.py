#For Node Details including bias & Weights
import onnx
import numpy as np


# Load the ONNX model
model_path = 'models/resnet50/resnet50.onnx'

model = onnx.load(model_path)

# Print node details
print("Node Details:")
for i, node in enumerate(model.graph.node):
  print(f"========================================== Node {i}:==========================================")
  print("Name:", node.name)
  print()
  print("Input Nodes:", node.input)
  print()
  print("Output Nodes:", node.output)
  print()
  print("Op Type:", node.op_type)
  print()
  print("Attribute:")
  for attr in node.attribute:
    print(f"- Name: {attr.name}")
    print(f"  Type: {attr.type}")
    #attributes can be numerics (kernal size,shape,etc.)
    if attr.type == onnx.AttributeProto.FLOATS:
      print(f"  Values1: {attr.floats}")
    elif attr.type == onnx.AttributeProto.INTS:
      print(f"  Values2: {attr.ints}") 
    #attributes can be stored in texts (Names)
    elif attr.type == onnx.AttributeProto.STRING:
      print(f"  Values3: {attr.s}")
    #can be tensors??  
    elif attr.type == onnx.AttributeProto.TENSOR:
      print(f" Tensor Value: ", attr.t.float_data)
    elif attr.name == "axis":
      print(f"  Value: {attr.i}")
    elif attr.name == "group":
      print(f"  Value: {attr.i}")     
    print()
    print()
  
  for input_name in node.input:
    # Check for the input = to a weight tensor
      for initializer in model.graph.initializer:
        if input_name == initializer.name:
          weight_tensor = initializer
          print("Weight tensor name:", weight_tensor.name)
          print()
          print("Shape:", weight_tensor.dims)
          print()
          weight_array = np.frombuffer(weight_tensor.raw_data, dtype=np.float32).reshape(weight_tensor.dims)
          print("Weight Values:", weight_array)
          print()

print()
print()
