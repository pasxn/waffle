from waffle.onnx.node import Node
from waffle import tensor
from typing import List
import os

WFLDBG = os.environ.get('WFLDBG')

class onnxList:
  def __init__(self, linearized_list:List[Node]):
    self.linearized_list = linearized_list

  def hard_traverse(self):
    for node_i in range(len(self.linearized_list)):
      inputs = []
      if node_i == 0:
        inputs.append(-1)
      for input in self.linearized_list[node_i].input:
        for i in range(len(self.linearized_list)):
          if input in self.linearized_list[i].output:
            inputs.append(i)
        
      self.linearized_list[node_i].set_traverse_input(inputs)
      self.linearized_list[node_i].search_layer()

    if WFLDBG:
      print('\n------------------ graph nodes list ------------------')
      for i in range(len(self.linearized_list)):
        print(f"node {str(i).rjust(4)}: {self.linearized_list[i].name.rjust(15)},   inputs -> {self.linearized_list[i].traverse_input}")

  def run(self, input:tensor) -> tensor:
    i = 0
    if WFLDBG: print('\n------------------ computation ------------------')
    while self.linearized_list[-1].output_computed is None:      
      current_node_traverse_input_len = len(self.linearized_list[i].traverse_input) if self.linearized_list[i].traverse_input is not None else 1
      if current_node_traverse_input_len == 1:
        if self.linearized_list[i].traverse_input[0] == -1:
          self.linearized_list[i].compute_node(input)
        else:
          current_input_node = self.linearized_list[self.linearized_list[i].traverse_input[0]]

          if current_input_node.output_computed is not None:
            self.linearized_list[i].compute_node(current_input_node.output_computed)
      elif current_node_traverse_input_len == 2:
        current_input_node_0 = self.linearized_list[self.linearized_list[i].traverse_input[0]]
        current_input_node_1 = self.linearized_list[self.linearized_list[i].traverse_input[1]]

        if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None:
          self.linearized_list[i].compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed)
      elif current_node_traverse_input_len == 3:
        current_input_node_0 = self.linearized_list[self.linearized_list[i].traverse_input[0]]
        current_input_node_1 = self.linearized_list[self.linearized_list[i].traverse_input[1]]
        current_input_node_2 = self.linearized_list[self.linearized_list[i].traverse_input[2]]

        if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None and current_input_node_2.output_computed is not None:
          self.linearized_list[i].compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed, z=current_input_node_2.output_computed)

      if WFLDBG:
        computed_output = self.linearized_list[i].output_computed
        print(f'at index{str(i).rjust(4)}   node: {self.linearized_list[i].name.rjust(15)}   output shape: {computed_output.shape if computed_output is not None else "None"}')
      
      i = i + 1 if i<len(self.linearized_list)-1 else 0
    
    return self.linearized_list[-1].output_computed
