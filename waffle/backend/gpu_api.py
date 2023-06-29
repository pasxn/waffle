from waffle.backend.gpu_backend import gpu_ops


class gpu:

  add:gpu_ops.kernels = gpu_ops.add_kernels()

  @staticmethod
  def compile():
    print(1)

  @staticmethod
  def add():
    pass
