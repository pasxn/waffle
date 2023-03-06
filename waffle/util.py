import math
from enum import Enum

DEVICES = Enum("DEVICES", ["CPU", "GPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "RECIPROCAL", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])

def prod(x): return math.prod(x)
