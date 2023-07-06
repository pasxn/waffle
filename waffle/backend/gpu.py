from waffle import tensor
from waffle.backend.gpu_backend import gpu_ops

class gpu:
  add_lib:gpu_ops.add_kernel = gpu_ops.add_kernel()

  @staticmethod
  def add(x:tensor, y:tensor) -> tensor:
    return tensor(gpu.add_lib(x, y))
