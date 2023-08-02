# %%
from matplotlib import pyplot as plt
from extra.benchmark.mlp_loop import run_loop_mlp

# layers, increase tensor size and calculate time, flops for each layer

# %%
# this also keep time and add flops
input_times = [i for i in range(0, 1001, 10)]; torch_times = []; waffle_times = []
for input in input_times:
  time_torch, time_waffle = run_loop_mlp(input, '../../models/mnist_mlp/mnist_mlp.onnx')
  torch_times.append(time_torch/1000000); waffle_times.append(time_waffle/1000000)

# %%
plt.figure()
plt.plot(input_times, torch_times, label="torch")
plt.plot(input_times, waffle_times, label="waffle")
plt.legend()
plt.show()
