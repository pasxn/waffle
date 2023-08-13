import unittest
import numpy as np

from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image

from extra.models.mnist_mlp.mlp_infer import predict_image_mlp
from extra.models.mnist_cnn.cnn_infer import predict_image_cnn
from extra.models.resnetlike.resnetlike_infer import predict_image_resnetlike

class test_nn_core(unittest.TestCase):
    
  def test_mlp(self):
    image = Image.open('./extra/images/mnist.jpg')
    transform = transforms.Compose([transforms.Resize((1, 28*28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image_torch = transform(image).unsqueeze(0)
    image_waffle = tensor(image_torch.numpy()).flatten().transpose().expand(1)

    # torch
    y_torch = predict_image_mlp(image_torch)

    # waffle
    model = nn.Module('mnist_mlp', './extra/models/mnist_mlp/mnist_mlp.onnx')
    model.compile()

    y_waffle = model.run(image_waffle)

    np.testing.assert_allclose(np.round(y_torch.numpy().flatten(), 2), np.round(y_waffle.data.flatten(), 2))

  def test_cnn(self):
    image = Image.open('./extra/images/mnist.jpg')
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image_torch = transform(image).unsqueeze(0)
    image_waffle = tensor(image_torch.numpy()).squeeze()

    # torch
    y_torch = predict_image_cnn(image_torch)

    # waffle
    model = nn.Module('mnist_cnn', './extra/models/mnist_cnn/mnist_cnn.onnx')
    model.compile()

    y_waffle = model.run(image_waffle)

    np.testing.assert_allclose(y_torch.argmax(dim=1, keepdim=True).item(), y_waffle.where(y_waffle.max()))

  def test_resnetlike(self):
    image = Image.open('./extra/images/mnist.jpg')
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image_torch = transform(image).unsqueeze(0)
    image_waffle = tensor(image_torch.numpy()).squeeze()

    # torch
    y_torch = predict_image_resnetlike(image_torch)

    # waffle
    model = nn.Module('resnetlike', './extra/models/resnetlike/resnetlike_mnist.onnx')
    model.compile()

    y_waffle = model.run(image_waffle)

    np.testing.assert_allclose(y_torch.argmax(dim=1, keepdim=True).item(), y_waffle.where(y_waffle.max()))    

if __name__ == '__main__':
  unittest.main()
