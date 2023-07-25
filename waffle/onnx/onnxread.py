from waffle.onnx.graph import Node
from typing import List
import onnx
import numpy as np

def read_onnx(model_path:str) -> List[Node]:
  model = onnx.load(model_path); nodes = []

  for i, node in enumerate(model.graph.node):
    attributes = []
    for attr in node.attribute:
      attribute = {}
      attribute['name'] = attr.name
      attribute['type'] = attr.type

      if attr.type == onnx.AttributeProto.FLOATS:
        attribute['values'] = list(attr.floats)
      elif attr.type == onnx.AttributeProto.INTS:
        attribute['values'] = list(attr.ints)
      elif attr.type == onnx.AttributeProto.STRING:
        attribute['values'] = attr.s.decode('utf-8')
      elif attr.type == onnx.AttributeProto.TENSOR:
        tensor_data = np.frombuffer(attr.t.raw_data, dtype=np.float32)
        attribute['values'] = tensor_data.reshape(attr.t.dims)

      attributes.append(attribute)

    weight_tensors = []
    for input_name in node.input:
      for initializer in model.graph.initializer:
        if input_name == initializer.name:
          weight_tensor = {}
          weight_tensor['name'] = initializer.name
          weight_tensor['shape'] = initializer.dims

          weight_array = np.frombuffer(initializer.raw_data, dtype=np.float32)
          weight_tensor['values'] = weight_array.reshape(initializer.dims)

          weight_tensors.append(weight_tensor)

    node_weight = None; node_bias = None
    assert len(weight_tensors) <= 2, 'there are more tan 2 weight tensors!'
    for i in weight_tensors:
      node_weight = i if i['name'] == 'weight' else None
      node_bias   = i if i['name'] == 'bias' else None

    nodes.append(Node(node.name, node.input, node.output, node.op_type, attributes, node_weight, node_bias))

    return nodes
  