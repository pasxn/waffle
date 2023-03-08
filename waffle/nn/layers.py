from waffle import tensor


# ***** nn ops ****
class Linear:
  def __init__(self, in_features, out_features, bias=True):
    #self.weight = tensor.glorot_uniform(out_features, in_features)
    self.bias = tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    print("cat")

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
  def __init__(self):
    pass