from waffle import tensor
from typing import List, Dict, Union, Any

class Node:
  def __init__(self, name:str, input:str, output:str, op_type:str, attributes:List[Dict[Any,Any]], params:tensor):
    self.name = name
    self.input = input
    self.output = output
    self.op_type = op_type

    self.attributes = attributes

    self.params = params

    self.traverse_input = None
    self.callable = None
    self.output_computed = None

  def set_traverse_input(self, traverse_input:Union[List[int], int]):
    self.traverse_input = traverse_input

  def search_layer(self):
    pass
