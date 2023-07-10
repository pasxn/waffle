class Node:
  def __init__(self, name, input, output, op_type, attributes, weight, bias):
    self.name = name
    self.input = input
    self.output = output
    self.op_type = op_type

    self.attributes = attributes

    self.weight = weight
    self.bias = bias

    self.callable = None

class Graph:
  def __init__(self, graph):
    self.liniearized_graph = graph

  def search_layer_node(self):
    pass

  def construct_graph(self):
    pass

  def traverse_graph(self):
    pass