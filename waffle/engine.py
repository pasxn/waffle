# backends = [CPU, NEON, GPU]

'''
engine file is where the mapping of backends and keeping track of backends will be done
can be done using object oriented design as well as proceedural design
'''

# engine file ops class maybe 

'''
noop <- implement on all the backends
neg
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

reshape <- implement on cpu only,prolly in somewhere like the shapetracker 
permute
slice
expland
flip

Linear <- implement in engine file, use the ops implemented in all the backends
Batchnorm2D
MaxPool2D
Conv2D

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

NonLeniarities <- prolly will be redundent

There should be a flag to track the backend for all the backend functions
Later some operations are explicitly mapped to the particular backend
Some implementations will be redundent but the final optimized ops map will give the best results

Numpy and GPU defa, if time permits, Neon

When compiling the model shapetrecker will compile the kernels for all the sizes needed for the particular network and after that those binaries can be executed
So there is no need of writing kernels with dynamic shapes, I think that is how it's done
'''