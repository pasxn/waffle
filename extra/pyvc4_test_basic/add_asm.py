import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def add_kernel(asm):
  # Load two vectors of length 16 from the host memory (address=uniforms[0]) to VPM
  setup_dma_load(nrows=2)
  start_dma_load(uniform)
  wait_dma_load()

  # Setup VPM read/write operations
  setup_vpm_read(nrows=2)
  setup_vpm_write()

  # Compute a + b
  mov(r0, vpm)
  mov(r1, vpm)
  fadd(vpm, r0, r1)

  # Store the result vector from VPM to the host memory (address=uniforms[1])
  setup_dma_store(nrows=1)
  start_dma_store(uniform)
  wait_dma_store()

  # Finish the thread
  exit()

def excec_add(a, b):
  with Driver() as drv:
    # Copy vectors to shared memory for DMA transfer
    inp = drv.copy(np.r_[a, b])
    out = drv.alloc(16, 'float32')

    # Run the program
    drv.execute(n_threads=1, program=drv.program(add_kernel), uniforms=[inp.address, out.address])
  
  return out

def add(a, b):
  pad = 16-(len(a)%16)
  a_mod = np.concatenate((a, np.zeros(pad)))
  b_mod = np.concatenate((b, np.zeros(pad)))

  result = np.array([])
  
  for i in range(16, len(a_mod), 16):
    result = np.concatenate((result, excec_add(a_mod[i-16:i], b_mod[i-16:i])))

  result = result[:-pad]

  return result
