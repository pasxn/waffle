import math
import numpy as np
from typing import Tuple, List, Union, Callable

from waffle import ops


class tensor:
  def __init__(self, data:Union[int, float, List, Tuple, np.ndarray]):
    if isinstance(data, list):
      if isinstance(data[0], tensor):
        datalist = []
        for element in data: datalist.append(element.data)
        self.data = np.array(datalist).astype(np.float32)
      else: self.data = np.array(data, dtype=np.float32)
    elif isinstance(data, int) or isinstance(data, float) or isinstance(data, np.float32):
      self.data = np.array([data], dtype=np.float32)
    elif isinstance(data, np.ndarray):
      if data.shape == tuple(): data = data.reshape((1,))
      self.data = data.astype(np.float32) if data.shape else data.reshape((1,)).astype(np.float32)
    else:
      raise RuntimeError(f"can't create Tensor from {data}")
    
  def __repr__(self) -> str:
    return f"<tensor {self.data!r}>"


  # ***** data handlers *****
  @property
  def shape(self) -> Tuple[int, ...]: return self.data.shape

  @property
  def len(self) -> int: return self.data.shape[0]

  @property
  def dtype(self) -> str: return str(np.float32).split("'")[1]

    
  # ***** creation helper functions *****
  @staticmethod
  def zeros(*shape, **kwargs) -> 'tensor':
    return tensor(np.zeros(shape, dtype=np.float32), **kwargs)

  @staticmethod
  def ones(*shape, **kwargs) -> 'tensor':
    return tensor(np.ones(shape, dtype=np.float32), **kwargs)

  @staticmethod
  def randn(*shape, **kwargs) -> 'tensor':
    return tensor(np.random.randn(*shape).astype(np.float32), **kwargs)
    
  @staticmethod
  def arange(start:int, stop:int, step: int, **kwargs) -> 'tensor':
    return tensor(np.arange(start, stop, step, dtype=np.float32).astype(np.float32), **kwargs)

  @staticmethod
  def uniform(*shape, **kwargs) -> 'tensor':
    return tensor((np.random.uniform(-1., 1., size=shape)/np.sqrt(math.prod(shape))).astype(np.float32), **kwargs)
  
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> 'tensor': 
    return tensor.uniform(*shape, **kwargs).mul((6/(shape[0]+math.prod(shape[1:])))**0.5)

  @staticmethod
  def eye(dim:int, **kwargs) -> 'tensor':
    return tensor(np.eye(dim).astype(np.float32), **kwargs)
  

  # ***** CPU explicit helper functions *****
  def resize(self, *shape, **kwargs): # order='C' or order='F'
    self.data = self.data.reshape(shape, **kwargs)

  def reshape(self, *shape, **kwargs) -> 'tensor': # order='C' or order='F'
    return tensor(self.data.reshape(shape, **kwargs))
    
  def concat(self, y:'tensor', axis=0, order=0) -> 'tensor':
    if not isinstance(y, tensor): raise RuntimeError("input must be a waffle tensor")
    if order == 0:
      return tensor(np.concatenate((self.data, y.data), axis=axis))
    elif order == 1:
      return tensor(np.concatenate((y.data, self.data), axis=axis))
    else:
      raise RuntimeError(f"order is goven as {order}. order must be 0 or 1")
        
  def pad2d(self, arg:Union[Tuple[int, ...], int], mode='constant') -> 'tensor':
    if isinstance(arg, tuple):
      return tensor(np.pad(self.data, arg, mode=mode, constant_values=0))
    elif isinstance(arg, int):
      return tensor(np.pad(self.data, pad_width=arg, mode=mode, constant_values=0))
    else:
      raise RuntimeError("argument must be an int or a tuple ((top, bottom), (left, right))")
        
  def transpose(self) -> 'tensor':
    return tensor(self.data.transpose())

  def flatten(self) -> 'tensor':
    return tensor(self.data.flatten())
    
  def reval(self):
    self.data = self.data.flatten()

  def permute(self, dim:Tuple[int, ...]) -> 'tensor':
    # dim (1, 0, 2): numbers refer to the dimensions of the original array assuming it's a 3 dimensional array
    return tensor(self.data.transpose(dim))

  def slice(self, start:Tuple[int, ...], end:Tuple[int, ...]) -> 'tensor':
    # ex: a.slice((1, 2), (3, 3))
    return tensor(self.data[start[0]:start[1], end[0]:end[1]])
    
  def expand(self, axis=None) -> 'tensor':
    if axis is None or isinstance(axis, tuple) or isinstance(axis, int):
      return tensor(np.expand_dims(self.data, axis=axis))
    else:
      raise RuntimeError("axis must be an int or a tuple")

  def squeeze(self) -> 'tensor':
    return tensor(np.squeeze(self.data))    
    
  def flip(self, axis=None) -> 'tensor':
    if axis is None or isinstance(axis, tuple) or isinstance(axis, int):
      return tensor(np.flip(self.data, axis=axis))
    else:
      raise RuntimeError("axis must be an int or a tuple")
    
  def where(self, val:'tensor') -> Union['tensor', int]:
    indices = np.where(self.data == val.data)
    return tensor(indices[0]) if indices[0].shape[0] > 1 else indices[0][0]
  
  
  # ***** slicing and indexing *****
  def __getitem__(self, val:int) -> Union['tensor', np.float32]:
    sliced = self.data[val]
    return tensor(sliced) if isinstance(sliced, np.ndarray) else sliced
  

  # ***** broadcasting mechanism *****
  @staticmethod
  def broadcasted(fxn:Callable[['tensor'], 'tensor'], tx:'tensor', ty:'tensor') -> 'tensor':
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
  def __neg__(self) -> 'tensor': return ops.neg(self)

  def add(self, y:'tensor') -> 'tensor': return self.broadcasted(ops.add, self, y)
  def sub(self, y:'tensor') -> 'tensor': return self.broadcasted(ops.sub, self, y)
  def mul(self, y:'tensor') -> 'tensor': return self.broadcasted(ops.mul, self, y)
  def div(self, y:'tensor') -> 'tensor': return self.broadcasted(ops.div, self, y)
  def pow(self, y:'tensor') -> 'tensor': return self.broadcasted(ops.pow, self, y)
  def dot(self, y:'tensor') -> 'tensor': return ops.gemm(self, y)
  def sum(self, axis=None) -> 'tensor': return ops.sum(self, axis)
  def max(self, axis=None) -> 'tensor': return ops.max(self, axis)

  def __add__(self, y:'tensor') -> 'tensor': return self.add(y)
  def __sub__(self, y:'tensor') -> 'tensor': return self.sub(y)
  def __mul__(self, y:'tensor') -> 'tensor': return self.mul(y)
  def __pow__(self, y:'tensor') -> 'tensor': return self.pow(y)
  def __truediv__(self, y:'tensor') -> 'tensor': return self.div(y)
  def __matmul__(self, x:'tensor') -> 'tensor': return self.dot(x)

  def __radd__(self, y:'tensor') -> 'tensor': return self.add(y)
  def __rsub__(self, y:'tensor') -> 'tensor': return self.sub(y)
  def __rmul__(self, y:'tensor') -> 'tensor': return self.mul(y)
  def __rpow__(self, y:'tensor') -> 'tensor': return self.pow(y)
  def __rtruediv__(self, y:'tensor') -> 'tensor': return self.div(y)
  def __rmatmul__(self, x:'tensor') -> 'tensor': return self.dot(x)

  def __iadd__(self, y:'tensor') -> 'tensor': return self.add(y)
  def __isub__(self, y:'tensor') -> 'tensor': return self.sub(y)
  def __imul__(self, y:'tensor') -> 'tensor': return self.mul(y)
  def __ipow__(self, y:'tensor') -> 'tensor': return self.pow(y)
  def __itruediv__(self, y:'tensor') -> 'tensor': return self.div(y)
  def __imatmul__(self, x:'tensor') -> 'tensor': return self.dot(x)
