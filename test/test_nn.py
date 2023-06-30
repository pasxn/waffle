import unittest
import numpy as np

from waffle import tensor
from waffle import nn
from waffle import ops

class test_nn(unittest.TestCase):
  @unittest.skip("Fix after defining nn.Module")  
  def test_nn_arithmetic(self):

    class nn_arith(nn.Module):
      def __init__(self, input):
        super(nn_arith, self).__init__()
        self.t0 = input
        self.t1 = tensor.randn(5)
        self.t2 = tensor.randn(4, 6)
        self.t3 = tensor.randn(2, 3, 4, 1)

      def forward(self):
        x1 = ops.exp(self.t1 + self.t2) + self.t0
        x2 = self.t1 + x1
        x3 = x2**1
        x4 = ops.relu(self.t3 * x3.sum())
        x5 = ops.log(x4.max())

        return x5

    t0 = tensor.randn(2, 3, 6)
    n1 = nn_arith(t0)
    for i in range(100):
      n1.forward()
  
  @unittest.skip("Fix after defining nn.Module")
  def test_nn_linear(self):
    from examples.mnist_fully_connected.mnist_model import MnistNN
    model = MnistNN(784, 10)
    x = tensor.randn(784, 1)

    np.testing.assert_allclose((10, 1), model.run(x).shape)

if __name__ == '__main__':
  unittest.main()
