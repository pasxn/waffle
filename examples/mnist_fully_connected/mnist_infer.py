from mnist_model import MnistNN
from waffle import tensor

import torch
from models.mnist_fully_connected.mnist_model import NN

print("waffle engine->\n")
model = MnistNN(784, 10)
print(model.run(tensor.randn(784, 1)).data)
print("\n-------------------\n")

print("torch engine->\n")
model = NN(784, 10)
model.eval()
x = torch.randn(1, 784)
with torch.no_grad():
  print(model(x).numpy().transpose())
