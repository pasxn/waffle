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
  def __init__(self, *args, **kwargs):
    print(args)

  def run(self):
    pass