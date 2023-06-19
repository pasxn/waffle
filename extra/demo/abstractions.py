# %%

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
g = tensor.eye(10)

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


