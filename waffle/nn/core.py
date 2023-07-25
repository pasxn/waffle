from waffle import tensor
from waffle import ops
from waffle.onnx.onnxread import read_onnx
from waffle.onnx.graph import Graph

class Module:
  def __init__(self, name):
    self.model_name = name
    self.graph_obj = None

  def load(self, path):
    linearized_model = read_onnx(path)
    self.graph_obj = Graph(linearized_model)

  def compile(self):
    pass
    # create the in memory graph

  def run(self, x:tensor) -> tensor:
    return self.forward(x)

if __name__ == '__main__':
  model_path = './models/resnet18/resnet18.onnx'
  graph_nodes = read_onnx(model_path)

  print(graph_nodes)
  print(f"\nnodes length: {len(graph_nodes)}\n")
