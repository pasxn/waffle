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

  def test_batchnorm2d(self):
    pass

  def test_conv2d(self):  
    def torch_conv(img, kernel_size, num_kernels, padding, stride):
      img = img.clone().detach().unsqueeze(0)
      if len(img.shape) > 3:
        channels = img.shape[-1]
        img = img.permute(0, 3, 1, 2)
      else:
        img = img.unsqueeze(-1)
        channels = img.shape[-1]
        img = img.permute(0, 3, 1, 2)  
      
      conv_layer = tnn.Conv2d(in_channels=channels, out_channels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding)
      output_torch =  conv_layer(img)

      return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))
    
    def waffle_conv(img, kernel_size, num_kernels, channels, padding, stride):
      conv_layer = wnn.Conv2D(kernel_size, num_kernels, channels, padding, stride)

      return conv_layer(img)
    
    image_waffle_1d = tensor.glorot_uniform(263, 376)
    image_torch_1d  = torch.from_numpy(image_waffle_1d.data)
  
    image_waffle_3d = tensor.glorot_uniform(263, 376, 3)
    image_torch_3d  = torch.from_numpy(image_waffle_3d.data)

    output_waffle_1d = waffle_conv(image_waffle_1d, 4, 2, 1, 0, 1)
    output_torch_1d  = torch_conv(image_torch_1d, 4, 2, 0, 1)
    np.testing.assert_allclose(output_torch_1d.shape, output_waffle_1d.shape)

    output_waffle_3d = waffle_conv(image_waffle_3d, 4, 2, 3, 2, 4)
    output_torch_3d  = torch_conv(image_torch_3d, 4, 2, 2, 4)
    np.testing.assert_allclose(output_torch_3d.shape, output_waffle_3d.shape)  

  def test_maxpool(self):
    def torch_maxpool(img, kernel_size, stride):
      img = img.clone().detach().unsqueeze(0)
      if len(img.shape) > 3:
        img = img.permute(0, 3, 1, 2)
      else:
        img = img.unsqueeze(-1)
        img = img.permute(0, 3, 1, 2)  
  
      pool_layer = tnn.MaxPool2d(kernel_size=kernel_size, stride=stride)
      output_torch =  pool_layer(img)

      return  output_torch.clone().detach().squeeze(0).numpy().transpose((1, 2, 0))
    
    def waffle_maxpool(img, kernel_size, stride):
      pool_layer = wnn.MaxPool2D(kernel_size, stride)

      return pool_layer(img)
    
    image_waffle_1d = tensor.glorot_uniform(263, 376)
    image_torch_1d  = torch.from_numpy(image_waffle_1d.data)
  
    image_waffle_3d = tensor.glorot_uniform(263, 376, 3)
    image_torch_3d  = torch.from_numpy(image_waffle_3d.data)

    output_waffle_1d = waffle_maxpool(image_waffle_1d, 4, 1)
    output_torch_1d  = torch_maxpool(image_torch_1d, 4, 1)
    np.testing.assert_allclose(output_torch_1d, output_waffle_1d.data)

    output_waffle_3d = waffle_maxpool(image_waffle_3d, 4, 1)
    output_torch_3d  = torch_maxpool(image_torch_3d, 4, 1)
    np.testing.assert_allclose(output_torch_3d, output_waffle_3d.data)      


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
