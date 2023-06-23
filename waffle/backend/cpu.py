from waffle.backend.cpu_backend import cpu_ops


def neg(x:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.neg(x.data))

def exp(x:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.exp(x.data))

def log(x:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.log(x.data))

def relu(x:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.relu(x.data))

def add(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.add(x.data, y.data))

def sub(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.sub(x.data, y.data))

def mul(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.mul(x.data, y.data))

def div(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.div(x.data, y.data))

def pow(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.pow(x.data, y.data))

def gemm(x:'tensor', y:'tensor') -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.gemm(x.data, y.data))

def sum(x:'tensor', axis=None) -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.sum(x.data, axis))

def max(x:'tensor', axis=None) -> 'tensor':
  from waffle import tensor
  return tensor(cpu_ops.max(x.data, axis))
