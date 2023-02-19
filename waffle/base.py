import numpy as np
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
        return f"<Tensor {self.data!r}>"

    
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
    def resize(self, *shape):
        self.data.resize((shape))

    def reshape(self, *shape, **kwargs): # order='C' or order='F'
        return tensor(self.data.reshape(shape, **kwargs))
    
    def concat(self, y, axis=0, order=0):
        if not isinstance(y, tensor): raise RuntimeError("input is not a waffle tensor")
        if order == 0:
            return np.concatenate((self.data, y.data), axis=axis)
        elif order == 1:
            return np.concatenate((y.data, self.data), axis=axis)
        else:
            raise RuntimeError(f"order is goven as {order}. order should be 0 or 1")
