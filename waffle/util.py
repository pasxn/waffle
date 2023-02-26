import math
from enum import Enum

DEVICES = Enum("DEVICES", ["CPU", "GPU", "NEON", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "RECIPROCAL", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX"])

def prod(x): return math.prod(x)
