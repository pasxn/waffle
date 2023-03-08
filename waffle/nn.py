'''
onnx and compilation code will go here 
extend class nn to keep track of kernels
Layets will directly come from engine as functions
This class will keep track of them using the layer constructs or the onnx graph 

This class should have

nn.load_onnx()
nn.compile()
nn.run()


'''
from waffle import tensor

class Module:
  def __init__(self):
    pass

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