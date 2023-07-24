import unittest
import numpy as np
from waffle import tensor

class test_tensor(unittest.TestCase):
    
  def test_slicing_n_indexing_numpy(self):
    t1 = tensor.ones(10)
    t2 = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      
    np.testing.assert_allclose(1.0, t1.data[0:5])
    np.testing.assert_allclose(tensor.ones(5).data, t1.data[0:5])

    np.testing.assert_allclose(tensor([1, 2, 3]).data, t2.data[1:4])
    np.testing.assert_allclose(tensor([0, 1, 2]).data, t2.data[0:3])
    np.testing.assert_allclose(tensor([0, 1, 2]).data, t2.data[:3])
    np.testing.assert_allclose(t2.data, t2.data[:])

  def test_slicing_n_indexing_in_built(self):
    t1 = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    np.testing.assert_allclose(tensor([0]).data, t1[0].data)
    np.testing.assert_allclose(tensor([9]).data, t1[-1].data)
    np.testing.assert_allclose(tensor([0, 1, 2]).data, t1[0:3].data)

  def test_helper_functions(self):
    t1 = tensor([1, 2, 3, 4, 5, 6])
    t2 = t1
    t3 = tensor([[1, 2], [3, 4]])
    t4 = t3
    t1.resize(2, 3)
    np.testing.assert_allclose(t2.reshape(2, 3).data, t1.data)
    t3.resize(4)
    np.testing.assert_allclose(t4.reshape(4).data, t3.data)

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

    t1 = tensor([[1], [2]])
    np.testing.assert_allclose(tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 0]]).data, t1.pad2d(1).data)
    np.testing.assert_allclose(tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0],
                                       [0, 0, 0, 0], [0, 0, 0, 0]]).data, t1.pad2d(((1, 2), (2, 1))).data)
    t1 = tensor([1, 2])
    t2 = tensor([[1, 2], [3, 4]])
    np.testing.assert_allclose(tensor([1 ,2]).data, t1.transpose().data)
    np.testing.assert_allclose(tensor([[1 ,3], [2, 4]]).data, t2.transpose().data)
    
    t1 = tensor.randn(20, 16, 4)
    np.testing.assert_allclose(t1.data.flatten(), t1.flatten().data)

    t1 = tensor.randn(20, 16, 4)
    t2 = t1
    t1.reval()
    np.testing.assert_allclose(t2.data.flatten(), t1.data)

    t1 = tensor.randn(2, 3, 4)
    np.testing.assert_allclose(t1.data.transpose((1, 0, 2)), t1.permute((1, 0, 2)).data)

    t1 = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(tensor([[6], [9]]).data, t1.slice((1, 3), (2, 4)).data)

    t1 = tensor.randn(4)
    np.testing.assert_allclose((1, 4), t1.expand(0).shape)
    np.testing.assert_allclose((1, 4, 1), t1.expand((0, 2)).shape)

    t1 = tensor([[1, 2], [3, 4]])
    np.testing.assert_allclose(tensor([[4, 3], [2, 1]]).data, t1.flip().data)

  def test_broadcasting(self):
    t1 = tensor.ones(2)
    t2 = tensor.ones(2)
    t3 = tensor.ones(1, 2)
    t4 = tensor.randn(4, 4, 2)
    t5 = tensor.randn(1, 4)

    np.testing.assert_allclose(tensor([2, 2]).data, t1.add(t2).data)
    np.testing.assert_allclose(tensor([[2, 2]]).data, t3.add(tensor(1)).data)
    np.testing.assert_allclose(tuple([4, 4, 3]), t4.add(tensor.ones(3)).shape)
    np.testing.assert_allclose(tuple([4, 4, 4]), t4.add(t5).shape)

  def test_arithmetic(self):
    t1 = tensor.ones(2)
    t2 = tensor.ones(2)

    np.testing.assert_allclose(tensor([2, 2]).data, (t1 + t2).data)
    np.testing.assert_allclose(tensor([0, 0]).data, (t1 - t2).data)
    np.testing.assert_allclose(tensor([1, 1]).data, (t1 * t2).data)
    np.testing.assert_allclose(tensor([1, 1]).data, (t1 / t2).data)
    np.testing.assert_allclose(tensor([1, 1]).data, (t1 ** t2).data)

    t1 = tensor([[1, 2], [3, 4]])
    t2 = tensor([[5, 6], [7, 8]])

    np.testing.assert_allclose(tensor([10]).data, t1.sum().data)
    np.testing.assert_allclose(tensor([8]).data, t2.max().data)

    t1 = tensor.randn(2048, 1024)
    t2 = tensor.randn(1024, 2048)

    np.testing.assert_allclose((2048, 2048), (t1@t2).shape)


if __name__ == '__main__':
  unittest.main()
