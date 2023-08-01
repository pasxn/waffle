from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image
import time

image = Image.open('./extra/images/mnist.jpg')
transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

image_torch = transform(image).unsqueeze(0)
image_waffle = tensor(image_torch.numpy()).squeeze()

model = nn.Module('mnist_cnn', './models/mnist_cnn/mnist_cnn.onnx')
model.compile()
y = model.run(image_waffle)

print(y)
