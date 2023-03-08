from waffle import tensor
from waffle import nn

class MnistNN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(MnistNN, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)
    self.a1 = nn.Relu()
    self.fc2 = nn.Linear(50, num_classes)
    
  def forward(self, x):
    x = self.fc1(x); print(x.shape)
    x = self.a1(x); print(x.shape)
    x = self.fc2(x); print(x.shape)

    return x