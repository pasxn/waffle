from waffle.backend import cpu
from enum import Enum
from waffle import tensor

OPS = Enum("OPS", ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])

# add description

def neg  (x:tensor)            -> tensor : return cpu.neg  (x)
def exp  (x:tensor)            -> tensor : return cpu.exp  (x)
def log  (x:tensor)            -> tensor : return cpu.log  (x)
def relu (x:tensor)            -> tensor : return cpu.relu (x)
def add  (x:tensor, y:tensor)  -> tensor : return cpu.add  (x, y)
def sub  (x:tensor, y:tensor)  -> tensor : return cpu.sub  (x, y)
def mul  (x:tensor, y:tensor)  -> tensor : return cpu.mul  (x, y)
def div  (x:tensor, y:tensor)  -> tensor : return cpu.div  (x, y)
def pow  (x:tensor, y:tensor)  -> tensor : return cpu.pow  (x, y)
def gemm (x:tensor, y:tensor)  -> tensor : return cpu.gemm (x, y)
def sum  (x:tensor, axis=None) -> tensor : return cpu.sum  (x, axis)
def max  (x:tensor, axis=None) -> tensor : return cpu.max  (x, axis)
