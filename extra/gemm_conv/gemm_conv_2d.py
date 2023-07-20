import numpy as np


image = np.array([i for i in range(1, 49)]).astype(np.float32).reshape(3, 4, 4)

filtr = np.array([i for i in range(1, 25)]).astype(np.float32).reshape(2, 3, 2, 2)

reshaped_x = []
for h in range(image.shape[0]):
  filter_out = []
  for i in range(image.shape[1]-filtr.shape[2]+1):
    for j in range(image.shape[2]-filtr.shape[3]+1):
      filter_out.append(image[h][i:i+filtr.shape[2], j:j+filtr.shape[3]].flatten().tolist())

  filter_out = np.array(filter_out).transpose()
  reshaped_x.append(filter_out)

reshaped_x = np.array(reshaped_x).reshape(reshaped_x[0].shape[0]*image.shape[0], reshaped_x[0].shape[1])

num_kernels = len(filtr)

reshaped_w = filtr.reshape(num_kernels, reshaped_x.shape[0])

print(reshaped_x.shape, reshaped_w.shape)

conv_result = reshaped_w@reshaped_x

height = int(((image.shape[1] - filtr.shape[2] + 2*(0))/1) + 1) # 0: number of padding, 1: stride
width = int(((image.shape[2] - filtr.shape[3] + 2*(0))/1) + 1) # 0: number of padding, 1: stride

conv_result = conv_result.reshape(num_kernels, height, width)

print(f"\nimage :\n{image}")
print(f"\nfilter :\n{filtr}")
print(f"\nmasked output :\n{reshaped_x}")
print(f"\nreshaped filters :\n{reshaped_w}")
print(f"\nconvolution result :\n{conv_result}\n")
