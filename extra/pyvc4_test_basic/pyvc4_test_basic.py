import numpy as np
import time

from add_asm import add

def cpu(a, b):
  return a + b

if __name__ == '__main__':
  # Input vectors
  a = np.random.random(1000000).astype('float32')
  b = np.random.random(1000000).astype('float32')

  start_time = time.perf_counter_ns()
  out_cpu = cpu(a, b)
  end_time = time.perf_counter_ns()
  execution_time_cpu = end_time - start_time

  start_time = time.perf_counter_ns()
  out_gpu = add(a, b)
  end_time = time.perf_counter_ns()
  execution_time_gpu = end_time - start_time

  print(' a '.center(80, '='))
  print(a)
  print(' b '.center(80, '='))
  print(b)
  print(' a+b '.center(80, '='))
  print(out_gpu)
  print(' error '.center(80, '='))
  print(np.abs(a+b-out))
