from waffle import tensor
from waffle import ops


# ***** nn ops ****
class Linear:
  def __init__(self, in_features:tensor, out_features:tensor, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = tensor.glorot_uniform(self.out_features, self.in_features)
    self.bias = tensor.zeros(out_features, 1) if bias else None

  def set_weight(self, weight:tensor):
    self.weight = weight

  def set_bias(self, bias:tensor):
    self.bias = bias

  def __call__(self, x:tensor) -> tensor:
    assert x.shape == (self.in_features, 1), f'The inputa shape is should be ({self.in_features}, {1})'
    x = self.weight@x
    return x.add(self.bias) if self.bias is not None else x
    

class Batchnorm:
  def __init__(self, input_mean:tensor, input_var:tensor, epsilon:tensor, scale:tensor, B:tensor):
    self.input_mean = input_mean
    self.input_var = input_var
    self.epsilon = epsilon
    self.scale = scale
    self.B = B

  def __call__(self, x):
    return (x - self.input_mean)/((self.input_var + self.epsilon)**0.5) * self.scale + self.B


class Conv2D:
  def __init__(self):
    pass

class MaxPool2D:
  def __init__(self):
    pass


# ***** nonLeniarities ****
class ReLU:
  def __call__(self, x:tensor) -> tensor:
    return ops.relu(x)
  