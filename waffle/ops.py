

'''
ops file is where the mapping of backends and keeping track of backends will be done
can be done using object oriented design as well as proceedural design
'''
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

def test(): print(GLOBAL_DEVICE)

# all the ops will be implemented here and will be called from here, device selection is also done here

'''
neg <- implement on all the backends
relu
exp
sign
add
sub
mul
div
pow
cmpeq
sum
max

conv <- implement on all the backends

<- implement on cpu only, prolly in somewhere like the shapetracker, or maybe in tensor itself ->
reshape [done]
resize [done]
cat [done]
pad2d
transpose
flatten
permute
slice
expland
flip

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
'''