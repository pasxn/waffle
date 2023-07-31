from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

from models.mnist_mlp.mlp_infer import predict_image_mlp


N = 10000

image = Image.open('./extra/images/mnist.jpg')
transform = transforms.Compose([transforms.Resize((1, 28*28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

image_torch = transform(image).unsqueeze(0)
image_waffle = tensor(image_torch.numpy()).flatten().transpose().expand(1)

# torch
start_time = time.perf_counter_ns()
for i in range(N):
  y_torch = predict_image_mlp(image_torch)
end_time = time.perf_counter_ns()
execution_time_torch = end_time - start_time

# waffle
model = nn.Module('mnist_mlp', './models/mnist_mlp/mnist_mlp.onnx')
model.compile()

start_time = time.perf_counter_ns()
for i in range(N):
  y_waffle = model.run(image_waffle)
end_time = time.perf_counter_ns()
execution_time_waffle = end_time - start_time

print(f"torch Time : {execution_time_torch/1000} ms")
print(f"waffle Time: {execution_time_waffle/1000} ms")

# print(y_waffle)
# print(y_torch)

# model = nn.Module('mnist_cnn', './models/mnist_cnn/mnist_cnn.onnx')
# model.compile()
# y = model.run(img)
# print(y)

# model = nn.Module('resnet18', './models/resnet18/resnet18.onnx')
# model.compile()
# model.run()
