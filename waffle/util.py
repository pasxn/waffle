import math
from enum import Enum

DEVICES = Enum("DEVICES", ["CPU", "GPU", "NEON", "HET"])

def prod(x): return math.prod(x)
