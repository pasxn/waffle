import ctypes
from enum import Enum
import numpy as np


OPS = Enum("OPS", ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])

class kernels:
  def __init__(self, op:OPS):
    # self.libs = [ctypes.CDLL(f'{op.name.lower()}_{str(i)}') for i in range(1, 6)]
    self.libs = [i for i in range(5)]
    
    # for lib in libs:
    #   lib.compile.argtypes = None; lib.compile.restype = None

  def compile(self):
    # for lib in self.libs: lib.compile
    print("compiled")

class add_kernels(kernels):
  def __init__(self):
    super().__init__(OPS.ADD)
    
    for lib in self.libs:
      # lib.add.restype = None
      # lib.add.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
      pass

  def compile(self):
    return super().compile()
    
  def __call__(self, size:int, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    dim = len(x.shape) if len(x.shape) is len(y.shape) else 0
      
    # Note: configure the data type here later\
    # create z here and return
    z = 0
    print("executed")

    return self.libs[dim-1](size, x, y, z)
    
