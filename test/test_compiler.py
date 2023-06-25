import unittest
import numpy as np

from waffle import tensor
from waffle import ops
from waffle.backend import shapetracker

class test_compiler(unittest.TestCase):

  @unittest.skip("There may not be a compiler")
  def test_shapetracker(self):
    
    ops.compile = True

    t1 = tensor.randn(2, 3, 4)
    t2 = tensor.randn(2, 3, 4)

    for i in range(10):
      t1 + t2

    np.testing.assert_allclose(10, len(shapetracker.shapes))

    ops.compile = False
    
    shapetracker.clear()
    for i in range(10):
      t1 + t2
    
    np.testing.assert_allclose(0, len(shapetracker.shapes))

    ops.compile = True
    for i in range(2):
      t1 + t2
    np.testing.assert_allclose((2, 3, 4), shapetracker.pop()['x_shape'])
    np.testing.assert_allclose(3, shapetracker.pop()['x_dim'])


if __name__ == '__main__':
  unittest.main()
