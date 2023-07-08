import ctypes
import os

os.system("g++ -c lib.cpp -fPIC -o lib.o")
os.system("g++ -shared -o lib.so -fPIC lib.o")

os.system("g++ -c add.cpp -fPIC -o add.o")
os.system("g++ -shared -o add.so -fPIC add.o")

add_lib = ctypes.CDLL('./add.so')

add_lib.run.argtypes = (ctypes.c_float, ctypes.POINTER(ctypes.c_float))
add_lib.run.restype = None

def add(x):
  z = (ctypes.c_float * 1)()
  add_lib.run(ctypes.c_float(x), z)

  return z

z = add(1.2)
print(z)
