from waffle.backend import cpu, gpu
from enum import Enum
from waffle import tensor


DEVICES = Enum("DEVICES", ["CPU", "HET"])
OPS = Enum("OPS", ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])

device = DEVICES.HET

def compile():
  if device is DEVICES.HET: gpu.compile()

# NOTE: Put a het handler function to route to CPU if multidimensional limit is also 1Mn, chop and compute if possible
def neg(x:tensor) -> tensor : return cpu.neg(x) if device is DEVICES.HET else cpu.neg(x)
def exp(x:tensor) -> tensor : return cpu.exp(x) if device is DEVICES.HET else cpu.exp(x)
def log(x:tensor) -> tensor : return cpu.log(x) if device is DEVICES.HET else cpu.log(x)
def relu(x:tensor) -> tensor: return cpu.relu(x) if device is DEVICES.HET else cpu.relu(x)
def add(x:tensor, y:tensor) -> tensor : return cpu.add(x, y) if device is DEVICES.HET else cpu.add(x, y)
def sub(x:tensor, y:tensor) -> tensor : return cpu.sub(x, y) if device is DEVICES.HET else cpu.sub(x, y)
def mul(x:tensor, y:tensor) -> tensor : return cpu.mul(x, y) if device is DEVICES.HET else cpu.mul(x, y)
def div(x:tensor, y:tensor) -> tensor : return cpu.div(x, y) if device is DEVICES.HET else cpu.div(x, y)
def pow(x:tensor, y:tensor) -> tensor : return cpu.pow(x, y) if device is DEVICES.HET else cpu.pow(x, y)
def gemm(x:tensor, y:tensor) -> tensor: return cpu.gemm(x, y) if device is DEVICES.HET else cpu.gemm(x, y)
def sum(x:tensor, axis=None) -> tensor: return cpu.sum(x, axis) if device is DEVICES.HET else cpu.sum(x, axis)
def max(x:tensor, axis=None) -> tensor: return cpu.max(x, axis) if device is DEVICES.HET else cpu.max(x, axis)
