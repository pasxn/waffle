from waffle import tensor
from waffle import ops
from waffle.onnx import read_onnx
from waffle.onnx import graph

class Module:
  def __init__(self, name):
    self.model_name = name
    self.graph_obj = None

  def load(self, path):
    linearized_model = read_onnx(path)
    self.graph_obj = graph(linearized_model)

  def compile(self):
    ops.compile()
    # create the in memory graph

  def run(self, x:tensor) -> tensor:
    return self.forward(x)
  