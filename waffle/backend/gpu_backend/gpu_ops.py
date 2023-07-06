import ctypes
import os
from enum import Enum
import numpy as np


OPS = Enum("OPS", ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
current_dir = os.path.dirname(os.path.abspath(__file__))

class add_kernel:
  def __init__(self): pass
    # self.lib = ctypes.CDLL(current_dir + '/target/kernels/add.so')
    # self.lib.add.restype = None
    # self.lib.add.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
    
  def __call__(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    dim = len(x.shape) if len(x.shape) is len(y.shape) else 0
      
    # Note: handle the complexity here
    # create z here and return
    z = 0; size = 0
    print("executed")
    self.lib(size, x, y, z)

    return z
