from waffle import tensor
from typing import List, Dict, Any

class Node:
  def __init__(self, name:str, input:str, output:str, op_type:str, attributes:List[Dict[Any,Any]], weight:tensor, bias:tensor):
    self.name = name
    self.input = input
    self.output = output
    self.op_type = op_type

    self.attributes = attributes

    self.weight = weight
    self.bias = bias

    self.callable = None
