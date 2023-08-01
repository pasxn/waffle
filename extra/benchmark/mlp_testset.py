from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image
import time

from models.mnist_mlp.mlp_infer import predict_image_mlp
from models.mnist_mlp.mlp_util import test_loader

# torch
start_time = time.perf_counter_ns()
for xx, _ in test_loader:
  xx = xx.reshape(xx.shape[0], -1)
  for x in xx:
    y_torch = predict_image_mlp(x.unsqueeze(0))
end_time = time.perf_counter_ns()
execution_time_torch = end_time - start_time

# waffle
model = nn.Module('mnist_mlp', './models/mnist_mlp/mnist_mlp.onnx')
model.compile()

start_time = time.perf_counter_ns()
for xx, _ in test_loader:
  xx = xx.reshape(xx.shape[0], -1)
  for x in xx:
    x = tensor(x.numpy()).flatten().transpose().expand(1)
    y_waffle = model.run(x)
end_time = time.perf_counter_ns()
execution_time_waffle = end_time - start_time

print(f"torch Time : {execution_time_torch/1000000} ms")
print(f"waffle Time: {execution_time_waffle/1000000} ms")
