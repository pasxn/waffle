from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image


image = Image.open('./extra/images/mnist.jpg')
transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

image = tensor(transform(image).numpy()).squeeze()

model = nn.Module('mnist_resnetlike', './extra//models/resnetlike/resnetlike_mnist.onnx')
model.compile()

output = model.run(image)
print(f"result: {output.where(output.max())}")
