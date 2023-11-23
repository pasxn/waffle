from waffle.nn.layers import Linear, Batchnorm, MaxPool2D, Conv2D
from waffle.nn.layers import ReLU, LeakyReLU, Softmax, LogSoftmax, Sigmoid, Tanh
from waffle.nn.layers import Flatten, Add, Fake

from waffle import tensor
from waffle.onnx.onnxread import read_onnx
from waffle.onnx.onnxlist import onnxList


class Module:
  def __init__(self, name:str, path:str):
    self.model_name = name; self.onnx_obj = None
    self.linearized_model = read_onnx(path)
    self.onnx_obj = onnxList(self.linearized_model)

  def compile(self):
    self.onnx_obj.hard_traverse()

  def run(self, x:tensor) -> tensor:
    return self.onnx_obj.run(x)
