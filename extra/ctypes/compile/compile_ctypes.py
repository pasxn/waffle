import ctypes
import os
import numpy as np

os.system("g++ -c compile.cpp -fPIC -o compile.o")
os.system("g++ -shared -o compilelib.so -fPIC compile.o")

add_lib = ctypes.CDLL('./compilelib.so')

add_lib.compile.argtypes = None
add_lib.compile.restype = None

add_lib.run.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))
add_lib.run.restype = None

z = (ctypes.c_int)()

add_lib.compile()
add_lib.run(ctypes.c_int(5), z)

print(z.value)
