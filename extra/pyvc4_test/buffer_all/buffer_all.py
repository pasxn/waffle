import numpy as np
import time

from add_asm import add

def cpu(a, b):
  return a + b

if __name__ == '__main__':
  # Input vectors
  a = np.random.random(100000).astype('float32')
  b = np.random.random(100000).astype('float32')

  start_time = time.perf_counter_ns()
  out_cpu = cpu(a, b)
  end_time = time.perf_counter_ns()
  execution_time_cpu = end_time - start_time

  start_time = time.perf_counter_ns()
  out_gpu = add(a, b)
  end_time = time.perf_counter_ns()
  execution_time_gpu = end_time - start_time

  print(f"CPU Time: {execution_time_cpu/1000} ms")
  print(f"GPU Time: {execution_time_gpu/1000} ms")
  print(f"Error: {np.abs(out_cpu-out_gpu).sum()/len(out_cpu)}")
