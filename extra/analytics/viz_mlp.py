# mlp iterative benchmark
import sys
import platform
import numpy as np
from matplotlib import pyplot as plt
from extra.benchmark.mlp_loop import run_loop_mlp


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

M = 101; N = 2

input_times = [i for i in range(1, M, N)];
torch_times_sum = []; waffle_times_sum = []; torch_times = []; waffle_times = []

for input in input_times:
  time_torch, time_waffle = run_loop_mlp(input, './models/mnist_mlp/mnist_mlp.onnx')
  
  torch_times.append(time_torch/(1000000*input)); waffle_times.append(time_waffle/(1000000*input))
  torch_times_sum.append(time_torch/(1000000)); waffle_times_sum.append(time_waffle/(1000000))

# average time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('average_time_taken_for_each_iteration_mlp' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()

# time taken for each iteration
plt.figure(figsize=(16, 6))
plt.plot(input_times, torch_times_sum, label="torch", color=np.random.rand(3,))
plt.plot(input_times, waffle_times_sum, label="waffle", color=np.random.rand(3,))
plt.xlabel('iterations')
plt.ylabel('time(ms)')
plt.legend()
plt.savefig('time_taken_for_each_iteration_mlp' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()
