from waffle.backend import cpu, gpu, shapetracker
from enum import Enum


DEVICES = Enum("DEVICES", ["CPU", "GPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])


GLOBAL_DEVICE = DEVICES.HET

def set_global_device(device):
   global GLOBAL_DEVICE
   GLOBAL_DEVICE = device


def neg(x, compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.NEG)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.neg(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.neg(x)
  else:raise RuntimeError("device is not configured correctly")

def relu(x, compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.RELU)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.relu(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.relu(x)
  else:raise RuntimeError("device is not configured correctly")
  
def exp(x, compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.EXP)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.exp(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.exp(x)
  else:raise RuntimeError("device is not configured correctly")

def log(x, compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.LOG)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.log(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.log(x)
  else:raise RuntimeError("device is not configured correctly")

def add(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.ADD)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.add(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.add(x, y)
  else:raise RuntimeError("device is not configured correctly")

def sub(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.SUB)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sub(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sub(x, y)
  else:raise RuntimeError("device is not configured correctly")

def mul(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.MUL)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.mul(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.mul(x, y)
  else:raise RuntimeError("device is not configured correctly")

def div(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.DIV)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.div(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.div(x, y)
  else:raise RuntimeError("device is not configured correctly")

def pow(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.POW)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.pow(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.pow(x, y)
  else:raise RuntimeError("device is not configured correctly")

def gemm(x, y, compile=False):
  if compile is True: shapetracker.parse(x.shape, y.shape, OPS.GEMM)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.gemm(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.gemm(x, y)
  else:raise RuntimeError("device is not configured correctly")

def sum(x, axis=None,compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.SUM)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sum(x, axis)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sum(x, axis)
  else:raise RuntimeError("device is not configured correctly")

def max(x, axis=None,compile=False):
  if compile is True: shapetracker.parse(x.shape, None, OPS.MAX)
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.max(x, axis)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.max(x, axis)
  else:raise RuntimeError("device is not configured correctly")
