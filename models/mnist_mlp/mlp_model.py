import torch.nn as nn    
import torch.nn.functional as F  

class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(input_size, 2048)
    self.fc2 = nn.Linear(2048, 1024)
    self.fc3 = nn.Linear(1024, 512)
    self.fc4 = nn.Linear(512, 128)
    self.fc5 = nn.Linear(128, 32)
    self.fc6 = nn.Linear(32, num_classes)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(F.relu(x))
    x = self.fc3(F.relu(x))
    x = self.fc4(F.relu(x))
    x = self.fc5(F.relu(x))
    x = self.fc6(F.relu(x))
    x = F.relu(x)

    return x
    