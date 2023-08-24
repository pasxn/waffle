from waffle.onnx.node import Node
from waffle import tensor
from typing import List
import onnx


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
        attribute['values'] = tensor.frombuffer(attr.t.raw_data).remove_nan().data.reshape(attr.t.dims)

      attributes.append(attribute)

    params = []
    for input_name in node.input:
      for initializer in model.graph.initializer:
        if input_name == initializer.name:
          parameter = {}
          parameter['name'] = initializer.name
          parameter['shape'] = initializer.dims

          weight_array = tensor.frombuffer(initializer.raw_data)
          parameter['values'] = tensor(weight_array.data.reshape(initializer.dims))

          params.append(parameter)

    nodes.append(Node(node.name, node.input, node.output, node.op_type, attributes, params))

  return nodes
  