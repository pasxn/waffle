import ctypes
import os
import numpy as np

os.system("g++ -c add.cpp -fPIC -o add.o")
os.system("g++ -shared -o add.so -fPIC add.o")

# Load the shared library
add_lib = ctypes.CDLL('./add.so')

# Define the argument and return types of the C function
add_lib.add.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float))
add_lib.add.restype = None

def add(size, x, y):
  # Convert Python lists to C arrays
  x_arr = (ctypes.c_float * size)(*x)
  y_arr = (ctypes.c_float * size)(*y)
  z_arr = (ctypes.c_float * size)()

  # Call the C function
  add_lib.add(ctypes.c_int(size), x_arr, y_arr, z_arr)

  # Convert the C array back to a Python list
  z = np.asarray(list(z_arr), dtype=np.float32)

  return z

# Example usage
a = np.asarray([i for i in range(5)], dtype=np.float32)
b = np.asarray([i for i in range(5)], dtype=np.float32)

z = add(len(a), a, b)
print(z)
