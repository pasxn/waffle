from waffle import tensor


class Module:
  def __init__(self, *args, **kwargs):
    pass

  def load_onnx(self):
    pass

  def kernel_search(self):
    pass

  def compile(self):
    pass

  def run(self, x):
    return self.forward(x)
  