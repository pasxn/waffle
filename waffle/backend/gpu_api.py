import ctypes
from enum import Enum

KERNEL_TYPES = Enum("KERNEL_TYPES", ["CPU", "HET"])

class kernel:
  def __init__(self, obj_path:str, kernel_type:'enum'):
    lib = ctypes.CDLL(obj_path)
    

class gpu:
  @staticmethod
  def compile():
    print(1)