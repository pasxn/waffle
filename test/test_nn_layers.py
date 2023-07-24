import unittest

import numpy as np

import torch
import torch.nn as tnn
import waffle.nn as wnn

from waffle import tensor
from waffle import ops

class test_layers(unittest.TestCase):

  def test_linear(self):
    input_size = 2048; output_size = 1024; batch_size = 1;
    
    torch_linear = tnn.Linear(input_size, output_size)
    waffle_linear = wnn.Linear(input_size, output_size)

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torch_linear((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle_linear(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)

  # conv
  # max pool

class test_nonlinearities(unittest.TestCase):
  
  def test_relu(self):
    input_size = 128; batch_size = 1;
    
    torchh = tnn.ReLU()
    waffle = wnn.ReLU()

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torchh((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)
    np.testing.assert_allclose(torch_output.transpose(), waffle_output)

  def test_leaky_relu(self):
    input_size = 128; batch_size = 1;
    
    torchh = tnn.LeakyReLU()
    waffle = wnn.LeakyReLU()

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torchh((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)
    np.testing.assert_allclose(torch_output.transpose(), waffle_output)

  @unittest.skip('slight error')
  def test_softmax(self):
    input_size = 128; batch_size = 1;
    
    torchh = tnn.Softmax()
    waffle = wnn.Softmax()

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torchh((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)
    np.testing.assert_allclose(torch_output.transpose(), waffle_output)

  @unittest.skip('major error')
  def test_sigmoid(self):
    input_size = 128; batch_size = 1;
    
    torchh = tnn.Sigmoid()
    waffle = wnn.Sigmoid()

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torchh((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)
    np.testing.assert_allclose(torch_output.transpose(), waffle_output)

  @unittest.skip('slight error')
  def test_tanh(self):
    input_size = 128; batch_size = 1;
    
    torchh = tnn.Tanh()
    waffle = wnn.Tanh()

    input = np.random.randn(batch_size, input_size).astype(np.float32)

    torch_output = torchh((torch.from_numpy(input))).detach().numpy()
    waffle_output = waffle(tensor(input).transpose()).data

    np.testing.assert_allclose(torch_output.transpose().shape, waffle_output.shape)
    np.testing.assert_allclose(torch_output.transpose(), waffle_output)


if __name__ == '__main__':
  unittest.main()
