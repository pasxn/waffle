# layers benchmark
import time
import sys
import torch
import torch.nn as tnn
from waffle import tensor
import waffle.nn as wnn
import platform
import numpy as np
from matplotlib import pyplot as plt


def get_platform():
  try:
    with open('/proc/device-tree/model', 'r') as f:
      model = f.read().strip()
  except IOError:
      model = ''

  machine = platform.machine()
  if 'Raspberry Pi' in model:
    return '_pi4' if '4 Model B' in model else '_pi2'
  elif 'x86' in machine:
    return "_x86"

machine = get_platform()

if machine == '_x86': M = 1001; N = 10
if machine == '_pi4': M = 201;  N = 2
if machine == '_pi4': M = 201;  N = 1

torch_times = []; waffle_times = []; torch_times_sum = []; waffle_times_sum = []
input_times = [i for i in range(1, M, N)];

input_size = 2048; output_size = 1024; batch_size = 1;
    
torch_linear = tnn.Linear(input_size, output_size)
waffle_linear = wnn.Linear(input_size, output_size)

input_img = np.random.randn(batch_size, input_size).astype(np.float32)

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    torch_output = torch_linear((torch.from_numpy(input_img))).detach().numpy()
  torch_times.append(time.perf_counter() - st/(1000000*input))
  torch_times_sum.append(time.perf_counter() - st/(1000000))

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    waffle_output = waffle_linear(tensor(input_img).transpose()).data
  waffle_times.append(time.perf_counter() - st/(1000000*input))
  waffle_times_sum.append(time.perf_counter() - st/(1000000))

# average time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('average_time_taken_for_each_iteration_linear' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()

# time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times_sum, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times_sum, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('time_taken_for_each_iteration_linear' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()


torch_times = []; waffle_times = []; torch_times_sum = []; waffle_times_sum = []
def torch_conv(img, kernel_size, num_kernels, padding, stride):
  img = img.clone().detach().unsqueeze(0)
  if len(img.shape) > 3:
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)
  else:
    img = img.unsqueeze(-1)
    channels = img.shape[-1]
    img = img.permute(0, 3, 1, 2)  
      
  conv_layer = tnn.Conv2d(in_channels=channels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding)
  output_torch =  conv_layer(img)

  return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))
    
def waffle_conv(img, kernel_size, num_kernels, channels, padding, stride):
  conv_layer = wnn.Conv2D(kernel_size, num_kernels, channels, padding, stride)

  return conv_layer(img)
  
image_waffle_3d = tensor.glorot_uniform(263, 376, 3)
image_torch_3d  = torch.from_numpy(image_waffle_3d.data)

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    output_torch_3d  = torch_conv(image_torch_3d, 4, 2, 2, 4)
  torch_times.append(time.perf_counter() - st/(1000000*input))
  torch_times_sum.append(time.perf_counter() - st/(1000000))

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    output_waffle_3d = waffle_conv(image_waffle_3d, 4, 2, 3, 2, 4)
  waffle_times.append(time.perf_counter() - st/(1000000*input))
  waffle_times_sum.append(time.perf_counter() - st/(1000000))

# average time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('average_time_taken_for_each_iteration_conv' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()

# time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times_sum, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times_sum, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('time_taken_for_each_iteration_conv' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()


torch_times = []; waffle_times = []; torch_times_sum = []; waffle_times_sum = []
def torch_maxpool(img, kernel_size, stride):
  img = img.clone().detach().unsqueeze(0)
  if len(img.shape) > 3:
    img = img.permute(0, 3, 1, 2)
  else:
    img = img.unsqueeze(-1)
    img = img.permute(0, 3, 1, 2)  
  
  pool_layer = tnn.MaxPool2d(kernel_size=kernel_size, stride=stride)
  output_torch =  pool_layer(img)

  return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))
    
def waffle_maxpool(img, kernel_size, stride):
  pool_layer = wnn.MaxPool2D(kernel_size, stride)

  return pool_layer(img)
  
image_waffle_3d = tensor.glorot_uniform(263, 376, 3)
image_torch_3d  = torch.from_numpy(image_waffle_3d.data)

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    output_torch_3d  = torch_maxpool(image_torch_3d, 4, 1)
  torch_times.append(time.perf_counter() - st/(1000000*input))
  torch_times_sum.append(time.perf_counter() - st/(1000000))

for input in input_times:
  st = time.perf_counter()
  for i in range(input):
    output_waffle_3d = waffle_maxpool(image_waffle_3d, 4, 1)
  waffle_times.append(time.perf_counter() - st/(1000000*input))
  waffle_times_sum.append(time.perf_counter() - st/(1000000))

# average time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('average_time_taken_for_each_iteration_maxpool' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()

# time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times_sum, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times_sum, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('time_taken_for_each_iteration_maxpool' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()
