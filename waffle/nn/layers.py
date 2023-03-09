from waffle import tensor
from waffle import ops


# ***** nn ops ****
class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = tensor.uniform(out_features, in_features)
    self.bias = tensor.zeros(out_features, 1) if bias else None

  def set_weight(self, weight):
    self.weight = weight

  def set_bias(self, bias):
    self.bias = bias

  def __call__(self, x):
    x = ops.gemm(self.weight, x)
    return x.add(self.bias) if self.bias is not None else x
    

class Batchnorm2D:
  def __init__(self):
    pass

class Conv2D:
  def __init__(self):
    pass

class MaxPool2D:
  def __init__(self):
    pass


# ***** nonLeniarities ****
class Relu:
  def __call__(self, x):
    return ops.relu(x)