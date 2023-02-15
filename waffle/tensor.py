import numpy as np
from waffle.util import prod

class Tensor:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, int) or isinstance(data, float):
            self.data = np.array([data], dtype=np.float32)
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


    # ***** slising and indexing *****
    def __getitem__(self, val):
        arg = []; new_shape = []
        if val is not None:
            for i, s in enumerate(val if isinstance(val, (list, tuple)) else [val]):
                if isinstance(s, int): arg.append((s, s + 1))
                else: arg.append((s.start if s.start is not None else 0,(s.stop if s.stop >=0 else
                                self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
                new_shape.append(arg[-1][1] - arg[-1][0])
                assert s.step is None or s.step == 1
        new_shape += self.shape[len(arg):]
        if len(new_shape) == 0: new_shape = (1,)
        ret = self.slice(arg = arg + [(0,self.shape[i]) for i in range(len(arg), len(self.shape))])
        
        return ret.reshape(shape=new_shape) if tuple(ret.shape) != tuple(new_shape) else ret
