import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def maxpool2d(image, filter_size, stride):
  image = np.expand_dims(image, axis=-1) if len(image.shape) < 3 else image
  image = np.transpose(image, (2, 0, 1))

  num_channels = image.shape[0]; image_height = image.shape[1]; image_width = image.shape[2]

  # check stride
  oh = ((image_height-filter_size)/stride) + 1; ow = ((image_width-filter_size)/stride) + 1
  if ((oh != int(oh)) or (ow != int(ow))) or (ow < 0 or oh < 0):
    RuntimeError("convolution cannot be performed with given parameters")

  if isinstance(filter_size, tuple):
    filter_height = filter_size[0]; filter_width  = filter_size[1]
  elif isinstance(filter_size, int):
    filter_height = filter_size; filter_width  = filter_size

  intermediate_x = []
  for h in range(num_channels):
    filter_out = []
    for i in range(0, image_height-filter_height+1, stride):
      for j in range(0, image_width-filter_width+1, stride):
        filter_out.append(np.max(image[h][i:i+filter_height, j:j+filter_width].flatten()))

    filter_out = np.array(filter_out)
    intermediate_x.append(filter_out)

  intermediate_x = np.array(intermediate_x)

  output_height = int(((image_height - filter_height)/stride) + 1)
  output_width  = int(((image_width - filter_width)/stride) + 1)

  output = intermediate_x.reshape(num_channels, output_height, output_width)
  output = np.transpose(output, (1, 2, 0))

  return output


def maxpool_torch(img, kernel_size, stride):
  if len(img.shape) > 3:
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)
  else:
    img = img.unsqueeze(-1)
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)  
  
  pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
  output_torch =  pool_layer(img)

  return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))


if __name__ == '__main__':

  KERNEL_SIZE = 4
  STRIDE      = 1

  # image
  img = Image.open('./gemm_conv/' + sys.argv[1])
  img = np.array(img).astype(np.float32)

  plt.imshow(img.astype('uint8')); plt.show()

  torch
  img_torch = torch.tensor(img).unsqueeze(0)

  start_time_torch = time.time()
  output_torch  = maxpool_torch(img_torch, KERNEL_SIZE, STRIDE)
  mean_output_torch = np.mean(output_torch , axis=2)
  end_time_torch = time.time()

  plt.imshow(mean_output_torch.astype('uint8')); plt.show()


  # waffle
  start_time_waffle = time.time()
  output_waffle = maxpool2d(img, KERNEL_SIZE, STRIDE)
  mean_output_waffle = np.mean(output_waffle , axis=2)
  end_time_waffle = time.time()

  plt.imshow(mean_output_waffle.astype('uint8')); plt.show()

  assert output_waffle.shape == output_torch.shape, 'Error in pool output shape'
  
  print(f"torch output shape    : {output_torch.shape}")
  print(f"waffle output shape   : {output_waffle.shape}")
  print(f"torch execution time  : {(end_time_torch - start_time_torch)*1000:.5f} ms")
  print(f"waffle execution time : {(end_time_waffle - start_time_waffle)*1000:.5f} ms")

output_torch = output_torch.flatten()
output_waffle = output_waffle.flatten()

for i in range(len(output_torch)):
  assert output_waffle[i] == output_torch[i], 'Error in pool output'
  