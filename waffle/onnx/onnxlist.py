from waffle.onnx.node import Node
from waffle import tensor
from typing import List


class onnxList:
  def __init__(self, linearized_list:List[Node]):
    self.linearized_list = linearized_list

  def hard_traverse(self):
    for node in self.linearized_list:
      for input in node.input:
        if input == 'input.1':
          node.traverse_input = -1
        for i in range(len(self.linearized_list)):
          if input in self.linearized_list[i].output:
            node.traverse_input = i

    for i in range(len(self.linearized_list)):
      print(i, ':', self.linearized_list[i].name ,">>>", self.linearized_list[i].traverse_input)
