from waffle import tensor
from waffle import ops
from typing import Tuple, List, Union, Callable


# ***** nn ops *****
class Linear:
  def __init__(self, in_features:int, out_features:int, bias=True):
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

  def __call__(self, x) -> tensor:
    return (x - self.input_mean)/((self.input_var + self.epsilon)**0.5) * self.scale + self.B


class Conv2D:
  def __init__(self, filter_size:Union[Tuple[int, ...], int], num_kernels:int, padding:Union[Tuple[int, ...], int], stride:int):
    self.filter_size = filter_size
    self.num_kernels = num_kernels
    self.padding = padding
    self.stride = stride

  def __call__(self, image:tensor) -> tensor:
    image = image.expand(axis=-1) if len(image.shape) < 3 else image
    image = image.permute((2, 0, 1))

    num_channels = image.shape[0]; image_height_original = image.shape[1]; image_width_original = image.shape[2]

    # check padding + filter size combination
    if isinstance(self.padding, tuple) and isinstance(filter_size, tuple):
      new_height = image_height_original + self.padding[0][0] + self.padding[0][1]; div_height = new_height/filter_size[0]
      new_width  = image_width_original + self.padding[1][0] + self.padding[1][1]; div_width = new_width/filter_size[1]
    elif isinstance(self.padding, int) and isinstance(filter_size, tuple):
      new_height = image_height_original + self.padding*2; div_height = new_height/filter_size[0]
      new_width  = image_width_original + self.padding*2; div_width = new_width/filter_size[1]
    elif isinstance(self.padding, tuple) and isinstance(filter_size, int):
      new_height = image_height_original + self.padding[0][0] + self.padding[0][1]; div_height = new_height/filter_size
      new_width  = image_width_original + self.padding[1][0] + self.padding[1][1]; div_width = new_width/filter_size
    elif isinstance(self.padding, int) and isinstance(filter_size, int):
      new_height = image_height_original + self.padding*2; div_height = new_height/filter_size
      new_width  = image_width_original + self.padding*2; div_width = new_width/filter_size

    if div_height != int(div_height) or div_width != int(div_width):
      RuntimeError("convolution cannot be performed with given parameters")

    # add self.padding
    padded_image = []
    for i in range(num_channels):
      if isinstance(self.padding, tuple):
        padded_image.append(np.pad(image[i], self.padding, mode='constant', constant_values=0))
      elif isinstance(self.padding, int):
        padded_image.append(np.pad(image[i], pad_width=self.padding, mode='constant', constant_values=0))

    image = np.array(padded_image)
    image_height = image.shape[1]; image_width = image.shape[2]

    # check stride
    oh = ((image_height-filter_size)/stride) + 1; ow = ((image_width-filter_size)/stride) + 1
    if ((oh != int(oh)) or (ow != int(ow))) or (ow < 0 or oh < 0):
      RuntimeError("convolution cannot be performed with given parameters")

    # a filter has x kernels of y channels
    if isinstance(filter_size, int):
      filtr = np.random.randn(num_kernels, num_channels, filter_size, filter_size).astype(np.float32)
    elif isinstance(filter_size, tuple):
      filtr = np.random.randn(num_kernels, num_channels, filter_size[0], filter_size[1]).astype(np.float32)
    
    filter_height = filtr.shape[2]; filter_width  = filtr.shape[3]

    intermediate_x = []
    for h in range(num_channels):
      filter_out = []
      for i in range(0, image_height-filter_height+1, stride):
        for j in range(0, image_width-filter_width+1, stride):
          filter_out.append(image[h][i:i+filter_height, j:j+filter_width].flatten())

      filter_out = np.array(filter_out).transpose()
      intermediate_x.append(filter_out)

    reshaped_x_height = filter_out.shape[0]*num_channels
    reshaped_x_width  = filter_out.shape[1]

    reshaped_x = np.array(intermediate_x).reshape(reshaped_x_height, reshaped_x_width)
    reshaped_w = filtr.reshape(num_kernels, reshaped_x_height)

    output = reshaped_w@reshaped_x

    if isinstance(self.padding, tuple):
      output_height = int(((image_height_original - filter_height + self.padding[0][0]+self.padding[0][1])/stride) + 1)
      output_width  = int(((image_width_original - filter_width + self.padding[1][0]+self.padding[1][1])/stride) + 1)
    elif isinstance(self.padding, int):
      output_height = int(((image_height_original - filter_height + 2*self.padding)/stride) + 1)
      output_width  = int(((image_width_original - filter_width + 2*self.padding)/stride) + 1)

    output = output.reshape(num_kernels, output_height, output_width)
    output = np.transpose(output, (1, 2, 0))

    return output


class MaxPool2D:
  def __init__(self):
    pass


# ***** nonleniarities *****
class ReLU:
  def __call__(self, x:tensor) -> tensor:
    return ops.relu(x)
  