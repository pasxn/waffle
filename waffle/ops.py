'''
ops file is where the mapping of backends and keeping track of backends will be done
can be done using object oriented design as well as proceedural design
'''
from waffle.backends import cpu, gpu, neon
from waffle.util import DEVICES

GLOBAL_DEVICE = DEVICES.HET

def set_global_device(device):
   global GLOBAL_DEVICE
   GLOBAL_DEVICE = device

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

# def neg(x:tensor) -> tensor:
#   pass

# def relu(x:tensor) -> tensor:
#   return tensor(cpu_ops.relu(x.data))

# def exp(x:tensor) -> tensor:
#   return tensor(cpu_ops.exp(x.data))

# def log(x:tensor) -> tensor:
#   return tensor(cpu_ops.log(x.data))

# def reciprocal(x:tensor) -> tensor:
#   return tensor(cpu_ops.reciprocal(x.data))

def add(x, y):
  if   GLOBAL_DEVICE == DEVICES.CPU: return cpu.add(x,y)
  elif GLOBAL_DEVICE == DEVICES.GPU: pass
  elif GLOBAL_DEVICE == DEVICES.HET: return cpu.add(x,y)
  else:raise RuntimeError("device is not configured correctly") 

# def sub(x:tensor, y:tensor)-> tensor:
#   return tensor(cpu_ops.sub(x.data-y.data))

# def mul(x:tensor, y:tensor)-> tensor:
#   return tensor(cpu_ops.mul(x.data*y.data))

# def div(x:tensor, y:tensor)-> tensor:
#   return tensor(cpu_ops.div(x.data/y.data))

# def pow(x:tensor, y:tensor)-> tensor:
#   return tensor(cpu_ops.pow(x.data**y.data))

# def sum(x:tensor, axis:int=None)-> tensor:
#   return tensor(cpu_ops.sum((x.data, axis)))

# def max(x:tensor, axis:int=None)-> tensor:
#   return tensor(cpu_ops.max((x.data, axis)))

# all the ops will be implemented here and will be called from here, device selection is also done here
# an operator function will be here to select layers for nn when implementing onnx 
# also same kind of thing for ops if needed

'''
neg <- implement on all the backends
relu
exp
log
>>>> gt0 = compire equal greater than 0
reciprocal
add
sub
mul
div
pow
>>>> cmpeq = compare if equal
sum
max
--
conv <- implement on all the backends
gemm



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

sum
max
mean


Linear <- implement in engine file, use the ops implemented in all the backends
Batchnorm2D
MaxPool2D
Conv2D


NonLeniarities <- in nn
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