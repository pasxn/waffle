import numpy as np
from waffle.util import prod

class Tensor:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data if data.shape else data.reshape((1,))
        else:
            raise RuntimeError(f"can't create Tensor from {data}")
    
    def __repr__(self):
        return f"<Tensor {self.data!r}>"

    
    # ***** data handlers ****
    @property
    def shape(self): return self.data.shape

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