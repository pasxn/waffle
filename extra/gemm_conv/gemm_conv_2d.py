import numpy as np


image = np.array([i for i in range(1, 49)]).astype(np.float32).reshape(3, 4, 4)

filtr = np.array([i for i in range(1, 25)]).astype(np.float32).reshape(2, 3, 2, 2)

num_channels = image.shape[0]
image_height = image.shape[1]
image_width  = image.shape[2]

# a filter has x kernels of y channels
num_kernels   = filtr.shape[0]
num_channels  = filtr.shape[1]
filter_height = filtr.shape[2]
filter_width  = filtr.shape[3]

reshaped_x = []
for h in range(num_channels):
  filter_out = []
  for i in range(image_height-filter_height+1):
    for j in range(image_width-filter_width+1):
      filter_out.append(image[h][i:i+filter_height, j:j+filter_width].flatten())

  filter_out = np.array(filter_out).transpose()
  reshaped_x.append(filter_out)

reshaped_x = np.array(reshaped_x).reshape(reshaped_x[0].shape[0]*num_channels, reshaped_x[0].shape[1])

reshaped_w = filtr.reshape(num_kernels, reshaped_x.shape[0])

conv_result = reshaped_w@reshaped_x

height = int(((image_height - filter_height + 2*(0))/1) + 1)  # 0: number of padding, 1: stride
width  = int(((image_width - filter_width + 2*(0))/1) + 1)    # 0: number of padding, 1: stride

conv_result = conv_result.reshape(num_kernels, height, width)

print(f"\nimage :\n{image}")
print(f"\nfilter :\n{filtr}")
print(f"\nmasked output :\n{reshaped_x}")
print(f"\nreshaped filters :\n{reshaped_w}")
print(f"\nconvolution result :\n{conv_result}\n")
