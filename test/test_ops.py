import unittest
import numpy as np

from waffle import tensor
from waffle import ops

class test_ops(unittest.TestCase):
    
  def test_basic_functional_ops(self):
    t1 = tensor.ones(2)
    t2 = tensor.ones(2)

    np.testing.assert_allclose(tensor(  [-1., -1.]            ).data, ops.neg(t1).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, ops.relu(t1).data)
    np.testing.assert_allclose(tensor(  [2.718282, 2.718282]  ).data, ops.exp(t1).data)
    np.testing.assert_allclose(tensor(  [0, 0]                ).data, ops.log(t1).data)
    np.testing.assert_allclose(tensor(  [2., 2.]              ).data, ops.add(t1,t2).data)
    np.testing.assert_allclose(tensor(  [0, 0]                ).data, ops.sub(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1]               ).data, ops.mul(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, ops.div(t1,t2).data)
    np.testing.assert_allclose(tensor(  [1., 1.]              ).data, ops.pow(t1,t2).data)
    np.testing.assert_allclose(tensor(  [2.0]                 ).data, ops.sum(t1).data)
    np.testing.assert_allclose(tensor(  [1.0]                 ).data, ops.max(t1).data)

    t1 = tensor.uniform(512, 1024)
    t2 = tensor.uniform(1024, 512)

    np.testing.assert_allclose(tensor(t1.data @ t2.data).data, ops.gemm(t1,t2).data)

if __name__ == '__main__':
  unittest.main()
