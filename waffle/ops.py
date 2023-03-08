from waffle.backends import cpu, gpu
from enum import Enum


DEVICES = Enum("DEVICES", ["CPU", "GPU", "HET"])
OPS = Enum("OPS", ["NEG", "RELU", "EXP", "LOG", "ADD", "SUB", "MUL", "DIV", "POW", "SUM", "MAX", "GEMM"])
LAYERS = Enum("LAYERS", ["LINEAR", "BATCHNORM2D", "CONV2D", "MAXPOOL2D"])


GLOBAL_DEVICE = DEVICES.HET

def set_global_device(device):
   global GLOBAL_DEVICE
   GLOBAL_DEVICE = device


# ***** nn ops ****
class Linear:
    def __init__(self, device=GLOBAL_DEVICE):
      self.device = device

    # if self.device this do this else do that

class Batchnorm2D:
    def __init__(self, device=GLOBAL_DEVICE):
      self.device = device

class Conv2D:
    def __init__(self, device=GLOBAL_DEVICE):
      self.device = device

class MaxPool2D:
    def __init__(self, device=GLOBAL_DEVICE):
      self.device = device


# ***** basic functional ops ****
def neg(x):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.neg(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.neg(x)
  else:raise RuntimeError("device is not configured correctly") 

def relu(x):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.relu(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.relu(x)
  else:raise RuntimeError("device is not configured correctly") 

def exp(x):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.exp(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.exp(x)
  else:raise RuntimeError("device is not configured correctly") 

def log(x):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.log(x)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.log(x)
  else:raise RuntimeError("device is not configured correctly") 

def add(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.add(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.add(x, y)
  else:raise RuntimeError("device is not configured correctly") 

def sub(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sub(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sub(x, y)
  else:raise RuntimeError("device is not configured correctly") 

def mul(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.mul(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.mul(x, y)
  else:raise RuntimeError("device is not configured correctly") 

def div(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.div(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.div(x, y)
  else:raise RuntimeError("device is not configured correctly") 

def pow(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.pow(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.pow(x, y)
  else:raise RuntimeError("device is not configured correctly")

def gemm(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.gemm(x, y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.gemm(x, y)
  else:raise RuntimeError("device is not configured correctly")

def sum(x, axis=None):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.sum(x, axis)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.sum(x, axis)
  else:raise RuntimeError("device is not configured correctly") 

def max(x, axis=None):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.max(x, axis)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.max(x, axis)
  else:raise RuntimeError("device is not configured correctly")

# all the ops will be implemented here and will be called from here, device selection is also done here
# an operator function will be here to select layers for nn when implementing onnx 
# also same kind of thing for ops if needed

'''
neg <- implement on all the backends
relu
exp
log
>>>> gt0 = compire equal greater than 0
add
sub
mul
div
pow
>>>> cmpeq = compare if equal
sum
max
--
gemm

only gemm is enough
use gemm for conv
study best way to gemm
study about gemm kernels
add conv through gemm doc to doc library


<- implement on cpu only, prolly in somewhere like the shapetracker, or maybe in tensor itself ->
reshape [done]
resize [done]
cat [done]
pad2d [done]
transpose [done]
flatten [done]
reval [done]
permute [done]
slice [done]
expand [done]
flip [done]

sum [done]
max [done]
mean 


Linear <- implement in this file, use the ops implemented in all the backends
Batchnorm2D
MaxPool2D
Conv2D


NonLeniarities <- in this, ops.Relu
sigmoid
relu
elu
swish
relu6
hardswish
tanh
gelu
leakyrelu
mish
softplus

There should be a flag to track the backend for all the backend functions
Later some operations are explicitly mapped to the particular backend
Some implementations will be redundent but the final optimized ops map will give the best results

Numpy and GPU defa, if time permits, Neon

When compiling the model shapetrecker will compile the kernels for all the sizes needed for the particular network and after that those binaries can be executed
So there is no need of writing kernels with dynamic shapes, I think that is how it's done

20/02
iplemented slicing and indexing is wrong, decided to not implement it until it's needed if ever, can use np slising until then. even tested in unit test

21/02 

implementing all the ops in this file as classes or functions
'''