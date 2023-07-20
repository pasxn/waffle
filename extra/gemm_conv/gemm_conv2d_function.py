import numpy as np
from PIL import Image


def conv2d(image, filter_size, num_kernels):
  image_height = image.shape[0]; image_width  = image.shape[1]
  
  if len(image.shape) <= 2:
    num_channels = 1; image = image.reshape(image.shape[0], image.shape[1], num_channels)
  else:
    num_channels = image.shape[3]

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

  return output


if __name__ == '__main__':
  img = Image.open('./mnist.jpg')
  img_arr = np.array(img).reshape(28, 28, 1)

  #output = conv2d(img_arr, 4, 2)

  print(img_arr.shape)

  img = Image.fromarray(img_arr)
  img.show()
  