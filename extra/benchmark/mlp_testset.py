from waffle import nn
from waffle import tensor

import time

from extra.models.mnist_mlp.mlp_infer import predict_image_mlp
from extra.models.mnist_mlp.mlp_util import test_loader

counter = 0; N = 2

# torch
start_time = time.perf_counter_ns()
for xx, _ in test_loader:
  xx = xx.reshape(xx.shape[0], -1)
  for x in xx:
    y_torch = predict_image_mlp(x.unsqueeze(0))
  counter +=1
  if counter == N: break
end_time = time.perf_counter_ns()
execution_time_torch = end_time - start_time
counter = 0

# # waffle
model = nn.Module('mnist_mlp', './extra/models/mnist_mlp/mnist_mlp.onnx')
model.compile()

start_time = time.perf_counter_ns()
for xx, _ in test_loader:
  xx = xx.reshape(xx.shape[0], -1)
  for x in xx:
    x = tensor(x.numpy()).flatten().transpose().expand(1)
    y_waffle = model.run(x)
  counter +=1
  if counter == N: break
end_time = time.perf_counter_ns()
execution_time_waffle = end_time - start_time
counter = 0

# eval
waffle_output = []; torch_output = []
for xx, _ in test_loader:
  xx = xx.reshape(xx.shape[0], -1)
  for x in xx:
    y_torch = predict_image_mlp(x.unsqueeze(0))
    x = tensor(x.numpy()).flatten().transpose().expand(1)
    y_waffle = model.run(x)

    torch_output.append(y_torch.argmax(dim=1, keepdim=True).item())
    waffle_output.append(y_waffle.where(y_waffle.max()))
  counter +=1
  if counter == N: break
counter = 0

sum = 0
for i in range(len(waffle_output)):
  sum = sum + 1 if torch_output[i] == waffle_output[i] else sum

accuracy = (sum/len(waffle_output))*100
speedup = max(execution_time_torch, execution_time_waffle)/min(execution_time_torch, execution_time_waffle)

print(f"accuracy with respect to torch: {accuracy:.2f}%")
print(f"torch Time : {(execution_time_torch/1000000000):.2f} s")
print(f"waffle Time: {(execution_time_waffle/1000000000):.2f} s")
print(f"waffle is x{speedup:.2f} {'slower' if execution_time_torch < execution_time_waffle else 'faster'} than torch!")
