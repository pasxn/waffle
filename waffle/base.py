import numpy as np
from typing import Tuple

from waffle import ops

class tensor:
  def __init__(self, data):
    if isinstance(data, list):
      self.data = np.array(data, dtype=np.float32)
    elif isinstance(data, int) or isinstance(data, float) or isinstance(data, np.float32):
      self.data = np.array([data], dtype=np.float32)
    elif isinstance(data, np.ndarray):
      if data.shape == tuple(): data = data.reshape((1,))
      self.data = data if data.shape else data.reshape((1,))
    else:
      raise RuntimeError(f"can't create Tensor from {data}")
    
  def __repr__(self):
    return f"<tensor {self.data!r}>"


  # ***** data handlers ****
  @property
  def shape(self): return self.data.shape

  @property
  def len(self): return self.data.shape[0]

  @property
  def dtype(self): return np.float32

    
  # ***** creation helper functions *****
  @classmethod
  def zeros(cls, *shape, **kwargs):
    return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs):
    return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs):
    return cls(np.random.randn(*shape).astype(np.float32), **kwargs)
    
  @classmethod
  def arange(cls, stop, start=0, **kwargs):
    return cls(np.arange(start=start, stop=stop).astype(np.float32), **kwargs)

  @classmethod
  def uniform(cls, *shape, **kwargs):
    return cls((np.random.uniform(-1., 1., size=shape)/np.sqrt(prod(shape))).astype(np.float32), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs):
    return cls(np.eye(dim).astype(np.float32), **kwargs)
    

  # ***** CPU explicit helper functions *****
  def resize(self, *shape, **kwargs): # order='C' or order='F'
    self.data = self.data.reshape(shape, **kwargs)

  def reshape(self, *shape, **kwargs): # order='C' or order='F'
    return tensor(self.data.reshape(shape, **kwargs))
    
  def concat(self, y, axis=0, order=0):
    if not isinstance(y, tensor): raise RuntimeError("input must be a waffle tensor")
    if order == 0:
      return tensor(np.concatenate((self.data, y.data), axis=axis))
    elif order == 1:
      return tensor(np.concatenate((y.data, self.data), axis=axis))
    else:
      raise RuntimeError(f"order is goven as {order}. order must be 0 or 1")
        
  def pad2d(self, arg, mode='constant'):
    if isinstance(arg, tuple):
      return tensor(np.pad(self.data, arg, mode=mode))
    elif isinstance(arg, int):
      return tensor(np.pad(self.data, pad_width=arg, mode=mode))
    else:
      raise RuntimeError("argument must be an int ora tuple ((top, bottom), (left, right))")
        
  def transpose(self):
    return tensor(self.data.transpose())

  def flatten(self):
    return tensor(self.data.flatten())
    
  def reval(self):
    self.data = self.data.flatten()

  def premute(self, dim:Tuple[int, ...]):
    # dim (1, 0, 2): numbers refer to the dimensions of the original array assuming it's a 3 dimensional array
    return tensor(self.data.transpose(dim))

  def slice(self, start:Tuple[int, ...], end:Tuple[int, ...]):
    # ex: a.slice((1, 2), (3, 3))
    return tensor(self.data[start[0]:start[1], end[0]:end[1]])
    
  def expand(self, axis=None):
    if axis is None or isinstance(axis, tuple) or isinstance(axis, int):
      return tensor(np.expand_dims(self.data, axis=axis))
    else:
      raise RuntimeError("axis must be an int or a tuple")
    
  def flip(self, axis=None):
    if axis is None or isinstance(axis, tuple) or isinstance(axis, int):
      return tensor(np.flip(self.data, axis=axis))
    else:
      raise RuntimeError("axis must be an int or a tuple")
    
  
  # ***** slicing and indexing *****
  def __getitem__(self, val):
    return tensor(self.data[val])
  

  # ***** broadcasting mechanism *****
  @staticmethod
  def broadcasted(fxn, tx, ty):
    if isinstance(tx, int) or isinstance(tx, float): tx = tensor([tx])
    elif isinstance(ty, int) or isinstance(ty, float): ty = tensor([ty])

    txl = len(tx.shape); tyl = len(ty.shape); tdl = abs(txl - tyl)
    if txl > tyl and txl != 1 and tyl != 1: ty = tensor(np.resize(ty.data, (1,) * tdl + ty.shape))
    if txl < tyl and txl != 1 and tyl != 1: tx = tensor(np.resize(tx.data, (1,) * tdl + tx.shape))

    shp1 = np.array(tx.shape); shp2 = np.array(ty.shape)
    broadcast_shape = tuple((np.maximum(shp1, shp2)).astype(int))

    txx = tensor(np.resize(tx.data, broadcast_shape))
    tyy = tensor(np.resize(ty.data, broadcast_shape))

    return fxn(txx, tyy)
  

  # ***** arithmetic operations*****
  def __neg__(self): return ops.neg(self)

  def add(self, y): return self.broadcasted(ops.add, self, y)
  def sub(self, y): return self.broadcasted(ops.sub, self, y)
  def mul(self, y): return self.broadcasted(ops.mul, self, y)
  def div(self, y): return self.broadcasted(ops.div, self, y)
  def pow(self, y): return self.broadcasted(ops.pow, self, y)
  def sum(self, axis=None): return ops.sum(self, axis)
  def max(self, axis=None): return ops.max(self, axis)

  def __add__(self, y): return self.add(y)
  def __sub__(self, y): return self.sub(y)
  def __mul__(self, y): return self.mul(y)
  def __pow__(self, y): return self.pow(y)
  def __truediv__(self, y): return self.div(y)

  def __radd__(self, y): return self.add(y)
  def __rsub__(self, y): return self.sub(y)
  def __rmul__(self, y): return self.mul(y)
  def __rpow__(self, y): return self.pow(y)
  def __rtruediv__(self, y): return self.div(y)

  def __iadd__(self, y): return self.add(y)
  def __isub__(self, y): return self.sub(y)
  def __imul__(self, y): return self.mul(y)
  def __ipow__(self, y): return self.pow(y)
  def __itruediv__(self, y): return self.div(y)
