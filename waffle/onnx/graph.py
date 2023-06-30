

class Node:
  def __init__(self):
    self.node_name = None
    self.callable = None
    self.weights = None
    self.biases = None

  def load_callable(self):
    callable.set_weights(self.weights)
    callable.set_weights(self.biases)

class Graph:
  def __init__(self, graph):
    self.liniearized_graph = graph