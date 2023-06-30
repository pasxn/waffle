from waffle import tensor
from waffle import ops


class Module:
  def __init__(self, *args, **kwargs):
    pass

  def load(self):
    pass
    # get the linearized shit as a list of dicts

  def compile(self):
    ops.compile()
    # create the in memory graph

  def run(self, x:tensor) -> tensor:
    return self.forward(x)
  