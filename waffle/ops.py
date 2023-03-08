from waffle.backends import cpu, gpu
from enum import Enum


DEVICES = Enum("DEVICES", ["CPU", "GPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])


GLOBAL_DEVICE = DEVICES.HET

def set_global_device(device):
   global GLOBAL_DEVICE
   GLOBAL_DEVICE = device
   

# ***** basic functional ops ****
class fops:
  @staticmethod
  def neg(x):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.neg(x)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.neg(x)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def relu(x):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.relu(x)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.relu(x)
    else:raise RuntimeError("device is not configured correctly") 
  
  @staticmethod
  def exp(x):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.exp(x)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.exp(x)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def log(x):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.log(x)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.log(x)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def add(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.add(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.add(x, y)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def sub(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sub(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sub(x, y)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def mul(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.mul(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.mul(x, y)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def div(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.div(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.div(x, y)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def pow(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.pow(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.pow(x, y)
    else:raise RuntimeError("device is not configured correctly")

  @staticmethod
  def gemm(x, y):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.gemm(x, y)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.gemm(x, y)
    else:raise RuntimeError("device is not configured correctly")

  @staticmethod
  def sum(x, axis=None):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sum(x, axis)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sum(x, axis)
    else:raise RuntimeError("device is not configured correctly") 

  @staticmethod
  def max(x, axis=None):
    if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.max(x, axis)
    elif GLOBAL_DEVICE == DEVICES.GPU: pass
    elif GLOBAL_DEVICE == DEVICES.HET: return cpu.max(x, axis)
    else:raise RuntimeError("device is not configured correctly")
