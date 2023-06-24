# %%

# Tensor Computation Module

# importing our package and other helper packages for the demo
from waffle import tensor
import numpy as np

# %%

# creating tensors from

# an integer
a = tensor(1)

# a float
b = tensor(2.4)

# a python list
c = tensor([1, 2, 3, 4, 5])

# a numpy array
d = tensor(np.array([[1, 2], [1, 2], [1, 2]]))

print(a)
print(b)
print(c)
print(d)

# %%

# obtaining the peoperties of the tensors using our framework
print(f"shape: {d.shape}")
print(f"length: {d.len}")
print(f"data type: {d.dtype}")

# %%

# creating tensors of a given shape of given properties

# zeros
a = tensor.zeros(2, 2)

# ones
b = tensor.ones(3, 3, 3)

# random floating point numbers
c = tensor.randn(8)

# evenly spaced values within a given interval
d = tensor.arange(0, 5, 0.5)


print(f"a [zeros]: {a.data}\n")
print(f"b [ones]: {b.data}\n")
print(f"c [random floating point numbers]: {c.data}\n")
print(f"c [evenly spaced values within a given interval]: {d.data}\n")

# %%

# unifrom distribution of numbers within-1 and 1 in a given shape
e = tensor.uniform(4, 4)

# glorot unifrom distribution of numbers within-1 and 1 in a given shape
f = tensor.glorot_uniform(4, 4)

# idenitiy matrix of s given size
g = tensor.eye(5)

print(f"e [unifrom distribution of numbers within-1 and 1 in a given shape]: {e.data}\n")
print(f"f [glorot unifrom distribution of numbers within-1 and 1 in a given shape]: {f.data}\n")
print(f"g [idenitiy matrix of s given size]: {g.data}")

# %%

# tensor reshape

# creating a tensor of shape (4, 2, 2, 4)
a = tensor.ones(4, 2, 2, 4)
print(a.shape)

# reshaping it to be in shape (4, 4)
print(a.reshape(4, 16).shape)

# %%

# tensor concatenation

# creating two tesnors to be concatenated
a = tensor.zeros(4, 4)
b = tensor.ones(4, 4)

print(a.concat(b))

# %%

# padding 

a = tensor.ones(4, 4)
print(a, "\n\n")

print(a.pad2d(2))

# %%

# transpose
a = tensor.ones(64, 128)
print(f"shape: {a.shape}")
print(f"transposed shape: {a.transpose().shape}")

# %%

# flatten

a = tensor.randn(2, 4, 4, 2)
print("shape a: ", a.shape); print("shape a[flattened]: ",a.flatten().shape)

# slice

a = tensor.randn(64, 64)
print(a.slice((1, 4), (5, 6)))

# expand

a = tensor.randn(2, 2)
print(a.expand(1).shape)

# flip

a = tensor.eye(4)
print(a.data); print(a.flip(axis=1).data)

# %%

# operations that will be accelerated in the GPU (currently running on CPU)

a = tensor.ones(4, 4)
b = tensor.ones(4, 4)

print(f"add: {a + b}\n")
print(f"sub: {a - b}\n")
print(f"mul: {a * b}\n")


# %%

print(f"div: {a / b}\n")
print(f"pow: {a ** b}\n")
print(f"dot: {a @ b}\n")
print(f"sum: {a.sum()}\n")
print(f"max: {a.max()}\n")

# %%

# Base Operation Mapper

from waffle import ops

a = tensor.randn(2, 2)
b = tensor.randn(2, 2)


print("NEG", ops.neg(a))
print("RELU", ops.relu(a))
print("EXP", ops.exp(a))
print("LOG", ops.log(a))

# %%

print("ADD", ops.add(a, b))
print("SUB", ops.sub(a,b))
print("MUL", ops.mul(a, b))
print("DIV", ops.div(a, b))

# %%

print("POW", ops.pow(a,b))
print("SUM", ops.sum(a))
print("MAX", ops.max(a))
print("GEMM", ops.gemm(a, b))

# %%
