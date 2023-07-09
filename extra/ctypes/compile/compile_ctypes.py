import ctypes
import os
import numpy as np

os.system("g++ -c compile.cpp -fPIC -o compile.o")
os.system("g++ -shared -o compilelib.so -fPIC compile.o")

add_lib = ctypes.CDLL('./compilelib.so')

add_lib.compile.argtypes = None
add_lib.compile.restype = None

add_lib.run.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
add_lib.run.restype = None

input_value = np.array(np.ones(10), dtype=np.float32)
output_value = np.empty_like(input_value).astype(np.float32)

add_lib.compile()

input_ptr = input_value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
add_lib.run(ctypes.c_int(len(input_value)), input_ptr, output_ptr)

for i in range(len(input_value)):
  assert input_value[i] + 10 == output_value[i], f"wrong at {i}!"
  