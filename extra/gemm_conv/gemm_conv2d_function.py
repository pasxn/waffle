import sys
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def conv2d(image, filter_size, num_kernels, padding, stride):
  image = np.expand_dims(image, axis=-1) if len(image.shape) < 3 else image

  image = np.transpose(image, (2, 0, 1))
  num_channels = image.shape[0]; image_height = image.shape[1]; image_width = image.shape[2]

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

  output_height = int(((image_height - filter_height + 2*padding)/stride) + 1)
  output_width  = int(((image_width - filter_width + 2*padding)/stride) + 1)

  output = output.reshape(num_kernels, output_height, output_width)
  output = np.transpose(output, (1, 2, 0))

  return output


def conv_torch(img, kernel_size, num_kernels, padding, stride):
  if len(img.shape) > 3:
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)
  else:
    img = img.unsqueeze(-1)
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)  
  
  conv_layer = nn.Conv2d(in_channels=channels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding)
  output_torch =  conv_layer(img)

  return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))


if __name__ == '__main__':
  # image
  img = Image.open('./' + sys.argv[1])
  img = np.array(img).astype(np.float32)

  plt.imshow(img.astype('uint8')); plt.show()

  # torch
  img_torch = torch.tensor(img).unsqueeze(0)

  start_time_torch = time.time()
  output_torch  = conv_torch(img_torch, 4, 2, 0, 1)
  mean_output_torch = np.mean(output_torch , axis=2)
  end_time_torch = time.time()

  plt.imshow(mean_output_torch.astype('uint8')); plt.show()


  # waffle
  start_time_waffle = time.time()
  output_waffle = conv2d(img, 4, 2, 0, 1)
  mean_output_waffle = np.mean(output_waffle , axis=2)
  end_time_waffle = time.time()

  plt.imshow(mean_output_waffle.astype('uint8')); plt.show()

  assert output_waffle.shape == output_torch.shape, 'Error in conv output shape'
  print(f"torch output shape    : {output_torch.shape}")
  print(f"waffle output shape   : {output_waffle.shape}")
  print(f"torch execution time  : {(end_time_torch - start_time_torch)*1000:.5f} ms")
  print(f"waffle execution time : {(end_time_waffle - start_time_waffle)*1000:.5f} ms")
  