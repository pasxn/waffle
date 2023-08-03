# matmul benchmark
import torch
from waffle import tensor
import time
import sys
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

N = [32, 64, 128, 256, 512, 1024, 2048, 4096]
stw = []; flopsw = [];  stt = []; flopst = []
for n in N:
  FLOPS = n*n*n*2
  
  a = tensor.randn(n, n)
  b = tensor.randn(n, n)

  c = torch.rand(n, n)
  d = torch.rand(n, n)

  def waffle_prog(a, b):
    st = time.perf_counter()
    _ = a@b
    return time.perf_counter() - st
  
  def torch_prog(c, d):
    st = time.perf_counter()
    _ = c@d
    return time.perf_counter() - st

  tmw = min([waffle_prog(a, b) for _ in range(5)])
  tmt = min([torch_prog(a, b) for _ in range(5)])

  stw.append(tmw*1e6); flopsw.append(FLOPS*1e-9/tmw)
  stt.append(tmt*1e6); flopst.append(FLOPS*1e-9/tmt)

# matmul time
plt.figure(figsize=(16, 6))
plt.plot(N, stw, marker='.', label="waffle", color=np.random.rand(3,))
plt.plot(N, stt, marker='.', label="torch", color=np.random.rand(3,))
plt.xlabel('tensor size')
plt.ylabel('time(us)')
plt.legend()
plt.savefig('matmul_time' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()

# matmul flops
plt.figure(figsize=(16, 6))
plt.plot(N, flopsw, marker='.', label="waffle", color=np.random.rand(3,))
plt.plot(N, flopst, marker='.', label="torch", color=np.random.rand(3,))
plt.xlabel('tensor size')
plt.ylabel('GFLOPS')
plt.legend()
plt.savefig('matmul_flops' + machine +'.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
if len(sys.argv) > 1: plt.show()
