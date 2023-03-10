import torch.nn as nn    
import torch.nn.functional as F  

class NN(nn.Module):
  def __init__(self, input_size, num_classes):  #28x28=784 size of the Mnist data
    super(NN, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)      #2 lAYERS
    self.fc2 = nn.Linear(50, num_classes)
    
  def forward(self, x):
    x = F.relu(self.fc1(x))      #ACTIVATION FUNCTION
    x = self.fc2(x)
    x = F.relu(x)

    return x
    