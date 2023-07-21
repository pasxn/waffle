import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def conv2d(image, filter_size, num_kernels):
  image = np.transpose(image, (2, 0, 1))
  image_height = image.shape[1]; image_width = image.shape[2]
  
  if len(image.shape) <= 2:
    num_channels = 1; image = image.reshape(num_channels, image.shape[0], image.shape[1])
  else:
    num_channels = image.shape[0]

  if isinstance(filter_size, int): # a filter has x kernels of y channels
    filtr = np.random.randn(num_kernels, num_channels, filter_size, filter_size)
  elif isinstance(filter_size, tuple):
    filtr = np.random.randn(num_kernels, num_channels, filter_size[0], filter_size[1])
  
  filter_height = filtr.shape[2]; filter_width  = filtr.shape[3]

  intermediate_x = []
  for h in range(num_channels):
    filter_out = []
    for i in range(image_height-filter_height+1):
      for j in range(image_width-filter_width+1):
        filter_out.append(image[h][i:i+filter_height, j:j+filter_width].flatten())

    filter_out = np.array(filter_out).transpose()
    intermediate_x.append(filter_out)

  reshaped_x_height = filter_out.shape[0]*num_channels
  reshaped_x_width  = filter_out.shape[1]

  reshaped_x = np.array(intermediate_x).reshape(reshaped_x_height, reshaped_x_width)

  reshaped_w = filtr.reshape(num_kernels, reshaped_x_height)

  output = reshaped_w@reshaped_x

  output_height = int(((image_height - filter_height + 2*(0))/1) + 1)  # 0: number of padding, 1: stride
  output_width  = int(((image_width - filter_width + 2*(0))/1) + 1)    # 0: number of padding, 1: stride

  output = output.reshape(num_kernels, output_height, output_width)
  output = np.transpose(output, (1, 2, 0))

  return output


def conv_torch(img, channels, num_kernels, kernel_size):
  conv_layer = nn.Conv2d(in_channels=channels, out_channels=num_kernels, kernel_size=kernel_size, stride=1, padding=0)
  return conv_layer(img)


if __name__ == '__main__':
  np.set_printoptions(threshold=np.inf)

  img = Image.open('./cfar.jpg')
  img = np.array(img).astype(np.float32)

  # print(img.shape)

  # output = conv2d(img, 4, 2)

  # print(output.shape)

  # torch
  img_torch = torch.tensor(img).unsqueeze(0)
  # img_torch = torch.tensor(img).squeeze(0)

  print(img_torch.shape)

  output = conv_torch(img_torch, 3, 2, 4)

  # mean_output = np.mean(img_torch.numpy(), axis=2)

  # mean_output = Image.fromarray(mean_output)
  # mean_output.show()

  #img = Image.fromarray(img)
  #img.show()
  