import unittest
import numpy as np

from waffle import tensor
from waffle.ops import fops

class test_ops(unittest.TestCase):
    
  def test_basic_functional_ops(self):
    t1 = tensor.ones(2)
    t2 = tensor.ones(2)

    # TODO: Remove .data when the support for tensor == tensor is added
    np.testing.assert_allclose(tensor(  [-1., -1.]            ).data, fops.neg(t1).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, fops.relu(t1).data)
    np.testing.assert_allclose(tensor(  [2.718282, 2.718282]  ).data, fops.exp(t1).data)
    np.testing.assert_allclose(tensor(  [0, 0]                ).data, fops.log(t1).data)
    np.testing.assert_allclose(tensor(  [2., 2.]              ).data, fops.add(t1,t2).data)
    np.testing.assert_allclose(tensor(  [0, 0]                ).data, fops.sub(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1]               ).data, fops.mul(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, fops.div(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, fops.pow(t1,t2).data)
    np.testing.assert_allclose(tensor(  [2.0]                 ).data, fops.sum(t1).data)
    np.testing.assert_allclose(tensor(  [1.0]                 ).data, fops.max(t1).data)

if __name__ == '__main__':
  unittest.main()
