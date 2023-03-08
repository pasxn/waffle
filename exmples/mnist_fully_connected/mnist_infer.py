from mnist_model import MnistNN
from waffle import tensor

model = MnistNN(784, 10)
print(model.forward(tensor.randn(784, 1)).shape)