from waffle import tensor
from waffle import ops


# ***** nn ops ****
class Linear:
  def __init__(self, in_features: tensor, out_features:tensor, bias=True):
    self.weight = tensor.glorot_uniform(out_features, in_features)
    self.bias = tensor.zeros(out_features, 1) if bias else None

  def set_weight(self, weight:tensor):
    self.weight = weight

  def set_bias(self, bias:tensor):
    self.bias = bias

  def __call__(self, x:tensor) -> tensor:
    x = self.weight@x
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
  def __call__(self, x:tensor) -> tensor:
    return ops.relu(x)
  