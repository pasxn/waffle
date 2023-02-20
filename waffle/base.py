import numpy as np
from typing import Tuple
from waffle.util import prod

class tensor:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, int) or isinstance(data, float):
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
        
    def transpose(self, axes=None):
        '''
        axes (1, 0, 2): numbers refer to the dimensions of the original array
        assuming it's a 3 dimensional array
        '''
        if axes is not None and isinstance(axes, tuple):
            return tensor(self.data.transpose(axes))
        elif axes is None:
            return tensor(self.data.transpose())
        else:
            RuntimeError("axes must be a tuple with dimensions equal to the dimensions of the tensor")

    def flatten(self):
        return tensor(self.data.flatten())
    
    def reval(self):
        self.data = self.data.flatten()
