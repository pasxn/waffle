from waffle import tensor
from waffle import ops
from typing import Tuple, Union


# ***** nn ops *****
class Linear:
  def __init__(self, in_features:int, out_features:int, weight:tensor=None, bias:tensor=None):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = tensor.glorot_uniform(self.out_features, self.in_features) if weight is None else weight
    self.bias = tensor.zeros(out_features, 1) if bias is None else bias.expand(1)

  def __call__(self, x:tensor) -> tensor:
    assert x.shape == (self.in_features, 1), f'The input shape is should be ({self.in_features}, {1})'
    x = self.weight@x
    return x.add(self.bias)
    

class Batchnorm:
  def __init__(self, input_mean:tensor, input_var:tensor, epsilon:tensor, scale:tensor, B:tensor):
    self.input_mean = input_mean
    self.input_var = input_var
    self.epsilon = epsilon
    self.scale = scale
    self.B = B

  def __call__(self, x) -> tensor:
    return (x - self.input_mean)/((self.input_var + self.epsilon)**0.5) * self.scale + self.B


class Conv2D:
  def __init__(self, filter_size:Union[Tuple[int, ...], int], num_kernels:int, num_channels:int, padding:Union[Tuple[int, ...], int], stride:int, weight:tensor=None, bias:tensor=None):
    self.filter_size = filter_size
    self.num_kernels = num_kernels
    self.num_channels = num_channels
    self.padding = padding
    self.stride = stride

    self.filtr = None
    # a filter has x kernels of y channels
    if isinstance(self.filter_size, int):
      self.filtr = tensor.glorot_uniform(self.num_kernels, self.num_channels, self.filter_size, self.filter_size) if weight is None else weight
    elif isinstance(self.filter_size, tuple):
      self.filtr = tensor.glorot_uniform(self.num_kernels, self.num_channels, self.filter_size[0], self.filter_size[1]) if weight is None else weight

    self.bias = tensor.glorot_uniform(self.num_kernels) if bias is None else bias

  def __call__(self, image:tensor) -> tensor:
    image = image.expand(axis=-1) if len(image.shape) < 3 else image
    image = image.permute((2, 0, 1))

    image_height_original = image.shape[1]; image_width_original = image.shape[2]

    # check padding + filter size combination
    if isinstance(self.padding, tuple) and isinstance(self.filter_size, tuple):
      new_height = image_height_original + self.padding[0][0] + self.padding[0][1]; div_height = new_height/self.filter_size[0]
      new_width  = image_width_original + self.padding[1][0] + self.padding[1][1]; div_width = new_width/self.filter_size[1]
    elif isinstance(self.padding, int) and isinstance(self.filter_size, tuple):
      new_height = image_height_original + self.padding*2; div_height = new_height/self.filter_size[0]
      new_width  = image_width_original + self.padding*2; div_width = new_width/self.filter_size[1]
    elif isinstance(self.padding, tuple) and isinstance(self.filter_size, int):
      new_height = image_height_original + self.padding[0][0] + self.padding[0][1]; div_height = new_height/self.filter_size
      new_width  = image_width_original + self.padding[1][0] + self.padding[1][1]; div_width = new_width/self.filter_size
    elif isinstance(self.padding, int) and isinstance(self.filter_size, int):
      new_height = image_height_original + self.padding*2; div_height = new_height/self.filter_size
      new_width  = image_width_original + self.padding*2; div_width = new_width/self.filter_size

    if div_height != int(div_height) or div_width != int(div_width):
      RuntimeError("convolution cannot be performed with given parameters")

    # add self.padding
    padded_image = []
    for i in range(self.num_channels):
      padded_image.append(image[i].pad2d(self.padding))

    image = tensor(padded_image)
    image_height = image.shape[1]; image_width = image.shape[2]

    # check stride
    oh = ((image_height-self.filter_size)/self.stride) + 1; ow = ((image_width-self.filter_size)/self.stride) + 1
    if ((oh != int(oh)) or (ow != int(ow))) or (ow < 0 or oh < 0):
      RuntimeError("convolution cannot be performed with given parameters")
    
    filter_height = self.filtr.shape[2]; filter_width  = self.filtr.shape[3]

    intermediate_x = []
    for h in range(self.num_channels):
      filter_out = []
      for i in range(0, image_height-filter_height+1, self.stride):
        for j in range(0, image_width-filter_width+1, self.stride):
          filter_out.append(image.buffer_index(h, i, i+filter_height, j, j+filter_width).flatten())

      filter_out = tensor(filter_out).transpose()
      intermediate_x.append(filter_out)

    reshaped_x_height = filter_out.shape[0]*self.num_channels
    reshaped_x_width  = filter_out.shape[1]

    reshaped_x = tensor(intermediate_x).reshape(reshaped_x_height, reshaped_x_width)
    reshaped_w = self.filtr.reshape(self.num_kernels, reshaped_x_height)

    output = reshaped_w@reshaped_x

    if isinstance(self.padding, tuple):
      output_height = int(((image_height_original - filter_height + self.padding[0][0]+self.padding[0][1])/self.stride) + 1)
      output_width  = int(((image_width_original - filter_width + self.padding[1][0]+self.padding[1][1])/self.stride) + 1)
    elif isinstance(self.padding, int):
      output_height = int(((image_height_original - filter_height + 2*self.padding)/self.stride) + 1)
      output_width  = int(((image_width_original - filter_width + 2*self.padding)/self.stride) + 1)

    output = output.reshape(self.num_kernels, output_height, output_width)
    
    biased_output = []; self.bias = self.bias.flatten()
    for i in range(output.len):
      biased_array = output[i] + self.bias[i]
      biased_output.append(biased_array)

    biased_output = tensor(biased_output)
    biased_output = biased_output.permute((1, 2, 0))

    return biased_output


class MaxPool2D:
  def __init__(self, filter_size:Union[Tuple[int, ...], int], stride:int):
    self.filter_size = filter_size
    self.stride = stride

  def __call__(self, image:tensor) -> tensor:
    image = image.expand(axis=-1) if len(image.shape) < 3 else image
    image = image.permute((2, 0, 1))

    num_channels = image.shape[0]; image_height = image.shape[1]; image_width = image.shape[2]

    oh = ((image_height-self.filter_size)/self.stride) + 1; ow = ((image_width-self.filter_size)/self.stride) + 1
    if ((oh != int(oh)) or (ow != int(ow))) or (ow < 0 or oh < 0):
      RuntimeError("max pooling cannot be performed with given parameters")

    if isinstance(self.filter_size, tuple):
      filter_height = self.filter_size[0]; filter_width  = self.filter_size[1]
    elif isinstance(self.filter_size, int):
      filter_height = self.filter_size; filter_width  = self.filter_size

    intermediate_x = []
    for h in range(num_channels):
      filter_out = []
      for i in range(0, image_height-filter_height+1, self.stride):
        for j in range(0, image_width-filter_width+1, self.stride):
          filter_out.append(image.buffer_index(h, i, i+filter_height, j, j+filter_width).data.flatten().max())
      intermediate_x.append(filter_out)

    intermediate_x = tensor(intermediate_x)

    output_height = int(((image_height - filter_height)/self.stride) + 1)
    output_width  = int(((image_width - filter_width)/self.stride) + 1)

    output = intermediate_x.reshape(num_channels, output_height, output_width)
    output = output.permute((1, 2, 0))

    return output


# ***** nonleniarities *****
class ReLU:
  def __call__(self, x:tensor) -> tensor:
    return ops.relu(x)
  
class LeakyReLU:
  def __call__(self, x:tensor, neg_slope:float=0.01) -> tensor:
    return ops.relu(x) - ops.relu(-neg_slope*x)

class Softmax:
  def __call__(self, x:tensor) -> tensor:
    exp_x = ops.exp(x - ops.max(x))
    return exp_x / ops.sum(exp_x, axis=0)

class LogSoftmax:
  def __call__(self, x:tensor) -> tensor:
    exp_x = ops.exp(x - ops.max(x))
    return ops.log(exp_x / ops.sum(exp_x, axis=0))
  
class Sigmoid:
  def __call__(self, x:tensor) -> tensor:
    return (1 / (1 + ops.exp(0-x)))
  
class Tanh:
  def __call__(self, x:tensor) -> tensor:
    return (ops.exp(x)-ops.exp(-x)) / (ops.exp(x)+ops.exp(-x))
  

# ***** extra *****
class Flatten:
  def __call__(self, x:tensor, y:tensor=None) -> tensor:
    target_shape = x.flatten().shape[0]
    x = x.permute((2, 0, 1)).expand(0)
    return x.reshape(-1, target_shape).transpose()
  
class Add:
  def __call__(self, x:tensor, y:tensor) -> tensor:
    return x + y
  
class Fake:
  def __call__(self, x:tensor) -> tensor:
    return x
  