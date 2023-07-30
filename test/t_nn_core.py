from waffle import nn
from waffle import tensor

from PIL import Image
import numpy as np

img = Image.open('./extra/images/mnist.jpg')
img = tensor(np.array(img))

model = nn.Module('mnist_cnn', './models/mnist_cnn/mnist_cnn.onnx')
model.compile()
y = model.run(img)
print(y)

# model = nn.Module('resnet18', './models/resnet18/resnet18.onnx')
# model.compile()
# model.run()
