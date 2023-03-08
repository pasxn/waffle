import unittest
import numpy as np

from waffle import tensor
from waffle.backends import cpu

class test_cpu_ops(unittest.TestCase):
    
  def test_ops(self):
    t1 = tensor.ones(2)
    t2 = tensor.ones(2)

    np.testing.assert_allclose(np.array(  [-1., -1.]            ), cpu.neg(t1).data)
    np.testing.assert_allclose(np.array(  [1., 1.]              ), cpu.relu(t1).data)
    np.testing.assert_allclose(np.array(  [2.718282, 2.718282]  ), cpu.exp(t1).data)
    np.testing.assert_allclose(np.array(  [0, 0]                ), cpu.log(t1).data)
    np.testing.assert_allclose(np.array(  [2., 2.]              ), cpu.add(t1,t2).data)
    np.testing.assert_allclose(np.array(  [0, 0]                ), cpu.sub(t1,t2).data)
    np.testing.assert_allclose(np.array(  [1., 1]               ), cpu.mul(t1,t2).data)
    np.testing.assert_allclose(np.array(  [1., 1.]              ), cpu.div(t1,t2).data)
    np.testing.assert_allclose(np.array(  [1., 1.]              ), cpu.pow(t1,t2).data)
    np.testing.assert_allclose(np.array(  [2.0]                 ), cpu.sum(t1).data)
    np.testing.assert_allclose(np.array(  [1.0]                 ), cpu.max(t1).data)

if __name__ == '__main__':
  unittest.main()
