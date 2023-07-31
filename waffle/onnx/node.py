from waffle import tensor
from waffle import nn
from typing import List, Dict, Union, Any

class Node:
  def __init__(self, name:str, input:str, output:str, op_type:str, attributes:List[Dict[Any,Any]], params:tensor):
    self.name    = name; self.op_type = op_type
    self.input   = input; self.output  = output
    self.attributes = attributes; self.params = params
    self.traverse_input = None; self.output_computed = None
    self.callable = None

  def set_traverse_input(self, traverse_input:Union[List[int], int]):
    self.traverse_input = traverse_input

  def search_layer(self):   
    name_lowercase = self.name.lower()
    if 'gemm' in name_lowercase:
      in_features  = self.params[0]['shape'][1]; out_features = self.params[0]['shape'][0]
      self.callable = nn.Linear(in_features, out_features, weight=self.params[0]['values'], bias=self.params[1]['values'])
    
    elif 'batchnorm' in name_lowercase: pass #NOTE: Impl later
    elif 'conv' in name_lowercase:
      _kernel_size = self.attributes[2]['values']; _padding = self.attributes[3]['values']
      kernel_size = _kernel_size[0] if _kernel_size[0] == _kernel_size[1] else tuple(_kernel_size)
      padding = _padding[0] if _padding[0] == _padding[2] else (_padding[0], _padding[2])
      stride = self.attributes[4]['values'][0]; num_kernels = self.params[0]['shape'][0]; num_channels = self.params[0]['shape'][1]
      self.callable = nn.Conv2D(kernel_size, num_kernels, num_channels, padding, stride, weight=self.params[0]['values'], bias=self.params[1]['values'])
      
    elif 'maxpool' in name_lowercase:
      _kernel_size = self.attributes[1]['values']; stride = self.attributes[3]['values'][0]
      kernel_size = _kernel_size[0] if _kernel_size[0] == _kernel_size[1] else tuple(_kernel_size)
      self.callable = nn.MaxPool2D(kernel_size, stride)
    
    elif 'leakyrelu' in name_lowercase: self.callable = nn.LeakyReLU()
    elif 'relu' in name_lowercase: self.callable = nn.ReLU()
    elif 'logsoftmax' in name_lowercase: self.callable = nn.LogSoftmax()
    elif 'softmax' in name_lowercase: self.callable = nn.Softmax()
    elif 'sigmoid' in name_lowercase: self.callable = nn.Sigmoid()
    elif 'tanh' in name_lowercase: self.callable = nn.Tanh()
    elif 'reshape' in name_lowercase: self.callable = nn.Flatten()
    else: self.callable = nn.Fake()

  def compute_node(self, x:tensor, y:tensor=None, z:tensor=None):
    # y and z are not used yet
    self.output_computed = self.callable(x)
