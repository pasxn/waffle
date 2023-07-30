from waffle.onnx.node import Node
from waffle import tensor
from typing import List


class onnxList:
  def __init__(self, linearized_list:List[Node]):
    self.linearized_list = linearized_list

  def hard_traverse(self):
    for node in self.linearized_list:
      inputs = []
      for input in node.input:
        if input == 'input.1':
          inputs.append(-1)
        
        for i in range(len(self.linearized_list)):
          if input in self.linearized_list[i].output:
            inputs.append(i)
        
        node.set_traverse_input(inputs)
      
      node.search_layer()

    # remove
    for i in range(len(self.linearized_list)):
      print(i, ':', self.linearized_list[i].name ,">>>", self.linearized_list[i].traverse_input)

  def run(self, input:tensor) -> tensor:
    i = 0; current_node = self.linearized_list[i]

    while self.linearized_list[-1].output_computed is not None:
      if len(current_node.traverse_input) == 1:
        if current_node.traverse_input[0] == -1:
          current_node.compute_node(input)
        else:
          current_input_node = self.linearized_list[current_node.traverse_input[0]]

          if current_input_node.output_computed is not None:
            current_node.compute_node(current_input_node.output_computed)
      elif len(current_node.traverse_input) == 2:
        current_input_node_0 = self.linearized_list[current_node.traverse_input[0]]
        current_input_node_1 = self.linearized_list[current_node.traverse_input[1]]

        if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None:
          current_node.compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed)
      elif len(current_node.traverse_input) == 3:
        current_input_node_0 = self.linearized_list[current_node.traverse_input[0]]
        current_input_node_1 = self.linearized_list[current_node.traverse_input[1]]
        current_input_node_2 = self.linearized_list[current_node.traverse_input[2]]

        if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None and current_input_node_2.output_computed is not None:
          current_node.compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed, z=current_input_node_2.output_computed)

      i = i + 1 if i<len(self.linearized_list) else 0
    
    return self.linearized_list[-1].output_computed
