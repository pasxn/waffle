from waffle.backends.cpu_backend import cpu_ops


def neg(x):
  from waffle import tensor
  return tensor(cpu_ops.neg(x.data))

def relu(x):
  from waffle import tensor
  return tensor(cpu_ops.relu(x.data))

def exp(x):
  from waffle import tensor
  return tensor(cpu_ops.exp(x.data))

def log(x):
  from waffle import tensor
  return tensor(cpu_ops.log(x.data))

def add(x, y):
  from waffle import tensor
  return tensor(cpu_ops.add(x.data, y.data))

def sub(x, y):
  from waffle import tensor
  return tensor(cpu_ops.sub(x.data, y.data))

def mul(x, y):
  from waffle import tensor
  return tensor(cpu_ops.mul(x.data, y.data))

def div(x, y):
  from waffle import tensor
  return tensor(cpu_ops.div(x.data, y.data))

def pow(x, y):
  from waffle import tensor
  return tensor(cpu_ops.pow(x.data, y.data))

def gemm(x, y):
  from waffle import tensor
  return tensor(cpu_ops.gemm(x.data, y.data))

def sum(x, axis=None):
  from waffle import tensor
  return tensor(cpu_ops.sum(x.data, axis))

def max(x, axis=None):
  from waffle import tensor
  return tensor(cpu_ops.max(x.data, axis))
