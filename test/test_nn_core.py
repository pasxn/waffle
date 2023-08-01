import unittest
import numpy as np

from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image

from models.mnist_mlp.mlp_infer import predict_image_mlp

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
    model = nn.Module('mnist_mlp', './models/mnist_mlp/mnist_mlp.onnx')
    model.compile()

    y_waffle = model.run(image_waffle)

    np.testing.assert_allclose(np.round(y_torch.numpy().flatten(), 2), np.round(y_waffle.data.flatten(), 2))

if __name__ == '__main__':
  unittest.main()
