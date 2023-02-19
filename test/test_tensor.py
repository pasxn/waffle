import unittest
import numpy as np

from waffle.base import tensor

class test_tensor(unittest.TestCase):
    
  def test_slicing_n_indexing(self):
    t1 = tensor.ones(10)
    t2 = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      
    np.testing.assert_allclose(1.0, t1.data[0:5])
    np.testing.assert_allclose(tensor.ones(5).data, t1.data[0:5])

    np.testing.assert_allclose(tensor([1, 2, 3]).data, t2.data[1:4])
    np.testing.assert_allclose(tensor([0, 1, 2]).data, t2.data[0:3])
    np.testing.assert_allclose(tensor([0, 1, 2]).data, t2.data[:3])
    np.testing.assert_allclose(t2.data, t2.data[:])

  def test_helper_functions(self):
    t1 = tensor.ones(2)
    t2 = tensor([[1, 2],[3, 4]])

    t1.resize(3); np.testing.assert_allclose(tensor([1, 1, 0]).data, t1.data)
    t2.resize(5); np.testing.assert_allclose(tensor([1, 2, 3, 4, 0]).data, t2.data)

    t1 = tensor([1, 2, 3, 4, 5, 6])
    t2 = tensor([[1, 2], [3, 4]])
    
    np.testing.assert_allclose(tensor([[1, 2, 3],[4, 5, 6]]).data, t1.reshape(2, 3).data)
    np.testing.assert_allclose(tensor([1, 2, 3, 4]).data, t2.reshape(4).data)
    np.testing.assert_allclose(tensor([1, 2, 3, 4]).data, t2.reshape(4, order='C').data)
    np.testing.assert_allclose(tensor([1, 3, 2, 4]).data, t2.reshape(4, order='F').data)
    np.testing.assert_allclose(tensor([[1, 3, 2, 4]]).data, t2.reshape(1, 4, order='F').data)

    t1 = tensor.ones(2)
    t2 = tensor.zeros(2)

    np.testing.assert_allclose(tensor([1, 1, 0, 0]).data, t1.concat(t2).data)
    np.testing.assert_allclose(tensor([0, 0, 1, 1]).data, t1.concat(t2, order=1).data)


if __name__ == '__main__':
  unittest.main()
  