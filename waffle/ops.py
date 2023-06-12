from waffle.backend import cpu, gpu, shapetracker
from enum import Enum
from waffle import tensor


DEVICES = Enum("DEVICES", ["CPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])

compile = False
device = DEVICES.HET

def neg(x:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.NEG)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.neg(x)
  else:
    if   device == DEVICES.CPU: return cpu.neg(x)
    elif device == DEVICES.HET: return cpu.neg(x)
    else:raise RuntimeError("device is not configured correctly")

def relu(x:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.RELU)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.relu(x)
  else:
    if   device == DEVICES.CPU: return cpu.relu(x)
    elif device == DEVICES.HET: return cpu.relu(x)
    else:raise RuntimeError("device is not configured correctly")
  
def exp(x:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.EXP)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.exp(x)
  else:
    if   device == DEVICES.CPU: return cpu.exp(x)
    elif device == DEVICES.HET: return cpu.exp(x)
    else:raise RuntimeError("device is not configured correctly")

def log(x:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.LOG)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.log(x)
  else:
    if   device == DEVICES.CPU: return cpu.log(x)
    elif device == DEVICES.HET: return cpu.log(x)
    else:raise RuntimeError("device is not configured correctly")

def add(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.ADD)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.add(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.add(x, y)
    elif device == DEVICES.HET: return cpu.add(x, y)
    else:raise RuntimeError("device is not configured correctly")

def sub(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.SUB)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.sub(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.sub(x, y)
    elif device == DEVICES.HET: return cpu.sub(x, y)
    else:raise RuntimeError("device is not configured correctly")

def mul(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.MUL)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.mul(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.mul(x, y)
    elif device == DEVICES.HET: return cpu.mul(x, y)
    else:raise RuntimeError("device is not configured correctly")

def div(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.DIV)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.div(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.div(x, y)
    elif device == DEVICES.HET: return cpu.div(x, y)
    else:raise RuntimeError("device is not configured correctly")

def pow(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.POW)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.pow(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.pow(x, y)
    elif device == DEVICES.HET: return cpu.pow(x, y)
    else:raise RuntimeError("device is not configured correctly")

def gemm(x:tensor, y:tensor) -> tensor:
  shape = shapetracker.parse(x.shape, y.shape, OPS.GEMM)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.gemm(x, y)
  else:
    if   device == DEVICES.CPU: return cpu.gemm(x, y)
    elif device == DEVICES.HET: return cpu.gemm(x, y)
    else:raise RuntimeError("device is not configured correctly")

def sum(x:tensor, axis=None) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.SUM)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.sum(x, axis)
  else:
    if   device == DEVICES.CPU: return cpu.sum(x, axis)
    elif device == DEVICES.HET: return cpu.sum(x, axis)
    else:raise RuntimeError("device is not configured correctly")

def max(x:tensor, axis=None) -> tensor:
  shape = shapetracker.parse(x.shape, None, OPS.MAX)
  if compile is True:
    shapetracker.fill(shape)
    return cpu.max(x, axis)
  else:
    if   device == DEVICES.CPU: return cpu.max(x, axis)
    elif device == DEVICES.HET: return cpu.max(x, axis)
    else:raise RuntimeError("device is not configured correctly")
