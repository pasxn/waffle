from waffle import nn
from waffle import tensor

import torchvision.transforms as transforms
from PIL import Image
import time
import os

from extra.models.mnist_mlp.mlp_infer import predict_image_mlp
path = os.path.abspath(os.path.dirname(__file__))

def run_loop_mlp(n, dpath):
  image = Image.open(path + '/../images/mnist.jpg')
  transform = transforms.Compose([transforms.Resize((1, 28*28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

  image_torch = transform(image).unsqueeze(0)
  image_waffle = tensor(image_torch.numpy()).flatten().transpose().expand(1)

  # torch
  start_time = time.perf_counter_ns()
  for i in range(n):
    y_torch = predict_image_mlp(image_torch)
  end_time = time.perf_counter_ns()
  execution_time_torch = end_time - start_time

  # waffle
  model = nn.Module('mnist_mlp', dpath)
  model.compile()

  start_time = time.perf_counter_ns()
  for i in range(n):
    y_waffle = model.run(image_waffle)
  end_time = time.perf_counter_ns()
  execution_time_waffle = end_time - start_time

  return execution_time_torch, execution_time_waffle

if __name__ == '__main__':
  N = 1

  execution_time_torch, execution_time_waffle = run_loop_mlp(N, './extra/models/mnist_mlp/mnist_mlp.onnx')
  speedup = max(execution_time_torch, execution_time_waffle)/min(execution_time_torch, execution_time_waffle)

  print(f"torch Time : {execution_time_torch/1000000} ms")
  print(f"waffle Time: {execution_time_waffle/1000000} ms")
  print(f"waffle is x{speedup:.2f} {'slower' if execution_time_torch < execution_time_waffle else 'faster'} than torch!")
