"""
This layer of abstraction was designed to add heterogeneous hardware support to the framework.

An attemp was taken to accelerate the operations on the Raspberry Pi 4 GPU (VideoCore Vi),
which was the target hardware at the design phase of the project. Later, due to technical complexities
that component (GPU backend) was then transferred to https://github.com/pasxn/v3dBLAS.git.

The method to add new hardware is to implement the backend and the API similar to the interface available
at waffle/backend. The CPU backend is already available at waffle/backend/cpu_backend and the CPU backend
API is aialable at waffle/backend/cpu.py!

Then the hardware which the operations are supposed to be executed can be mapped in the dedicated function
for the each operation below in this file.
"""

from waffle.backend import cpu
from enum import Enum
from waffle import tensor

OPS = Enum("OPS", ["NEG", "EXP", "LOG", "RELU", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])

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
