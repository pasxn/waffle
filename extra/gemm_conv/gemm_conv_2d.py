import numpy as np


image = np.array([i for i in range(1, 49)]).astype(np.float32).reshape(4, 4, 3)

filtr = np.array([i for i in range(1, 25)]).astype(np.float32).reshape(2, 2, 2, 3)

image_height = image.shape[0]
image_width  = image.shape[1]
num_channels = image.shape[2]

# a filter has x kernels of y channels
num_channels  = filtr.shape[0]
filter_height = filtr.shape[1]
filter_width  = filtr.shape[2]
num_kernels   = filtr.shape[3]

# intermediate_x = []
# for h in range(num_channels):
#   filter_out = []
#   for i in range(image_height-filter_height+1):
#     for j in range(image_width-filter_width+1):
#       filter_out.append(image[h][i:i+filter_height, j:j+filter_width].flatten())

#   filter_out = np.array(filter_out).transpose()
#   intermediate_x.append(filter_out)

# reshaped_x_height = filter_out.shape[0]*num_channels
# reshaped_x_width  = filter_out.shape[1]

# reshaped_x = np.array(intermediate_x).reshape(reshaped_x_height, reshaped_x_width)

# reshaped_w = filtr.reshape(num_kernels, reshaped_x_height)

# conv_result = reshaped_w@reshaped_x

# conv_result_height = int(((image_height - filter_height + 2*(0))/1) + 1)  # 0: number of padding, 1: stride
# conv_result_width  = int(((image_width - filter_width + 2*(0))/1) + 1)    # 0: number of padding, 1: stride

# conv_result = conv_result.reshape(num_kernels, conv_result_height, conv_result_width)

print(f"\n  image               \n{image}")
print(f"\n  filter              \n{filtr}")
# print(f"\n  masked output       \n{reshaped_x}")
# print(f"\n  reshaped filters    \n{reshaped_w}")
# print(f"\n  convolution result  \n{conv_result}\n")
