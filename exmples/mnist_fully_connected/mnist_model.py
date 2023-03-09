from waffle import tensor
from waffle import nn

class MnistNN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(MnistNN, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)
    self.ac1 = nn.Relu()
    self.fc2 = nn.Linear(50, num_classes)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.ac1(x)
    x = self.fc2(x)

    return x
  