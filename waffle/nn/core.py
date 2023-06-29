from waffle import tensor


class Module:
  def __init__(self, *args, **kwargs):
    pass

  def load_onnx(self):
    pass

  def run(self, x:tensor) -> tensor:
    return self.forward(x)
  