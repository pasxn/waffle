from waffle.backend import cpu, gpu
from enum import Enum
from waffle import tensor


DEVICES = Enum("DEVICES", ["CPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])

device = DEVICES.HET

def neg(x:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.neg(x)
  elif device == DEVICES.HET: return cpu.neg(x)
  else:raise RuntimeError("device is not configured correctly")

def relu(x:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.relu(x)
  elif device == DEVICES.HET: return cpu.relu(x)
  else:raise RuntimeError("device is not configured correctly")
  
def exp(x:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.exp(x)
  elif device == DEVICES.HET: return cpu.exp(x)
  else:raise RuntimeError("device is not configured correctly")

def log(x:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.log(x)
  elif device == DEVICES.HET: return cpu.log(x)
  else:raise RuntimeError("device is not configured correctly")

def add(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.add(x, y)
  elif device == DEVICES.HET: return cpu.add(x, y)
  else:raise RuntimeError("device is not configured correctly")

def sub(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.sub(x, y)
  elif device == DEVICES.HET: return cpu.sub(x, y)
  else:raise RuntimeError("device is not configured correctly")

def mul(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.mul(x, y)
  if device == DEVICES.HET: return cpu.mul(x, y)
  else:raise RuntimeError("device is not configured correctly")

def div(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.div(x, y)
  elif device == DEVICES.HET: return cpu.div(x, y)
  else:raise RuntimeError("device is not configured correctly")

def pow(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.pow(x, y)
  elif device == DEVICES.HET: return cpu.pow(x, y)
  else:raise RuntimeError("device is not configured correctly")

def gemm(x:tensor, y:tensor) -> tensor:
  if   device == DEVICES.CPU: return cpu.gemm(x, y)
  elif device == DEVICES.HET: return cpu.gemm(x, y)
  else:raise RuntimeError("device is not configured correctly")

def sum(x:tensor, axis=None) -> tensor:
  if   device == DEVICES.CPU: return cpu.sum(x, axis)
  elif device == DEVICES.HET: return cpu.sum(x, axis)
  else:raise RuntimeError("device is not configured correctly")

def max(x:tensor, axis=None) -> tensor:
  if   device == DEVICES.CPU: return cpu.max(x, axis)
  elif device == DEVICES.HET: return cpu.max(x, axis)
  else:raise RuntimeError("device is not configured correctly")
