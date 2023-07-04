from waffle.backend.gpu_backend import gpu_ops


class gpu:

  add_lib:gpu_ops.add_kernels = gpu_ops.add_kernels()

  @staticmethod
  def compile():
    gpu.add_lib.compile()

  @staticmethod
  def add():
    pass
