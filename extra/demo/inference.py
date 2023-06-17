#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Define the CNN architecture
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.fc1 = nn.Linear(64 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 10)


  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = x.view(-1, 64 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
        
    return F.log_softmax(x, dim=1)

# Load the saved model weights
model = Net()
model.load_state_dict(torch.load('./models/mnist_cnn/MNIST_CNN.ckpt'))
model.eval()

# Preprocess the test image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

test_image_path = '../../testimg1.jpeg'
test_image = Image.open(test_image_path)
test_image = transform(test_image).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(test_image)
    _, predicted = torch.max(output.data, 1)

predicted_label = predicted.item()

print(f"Predicted Label: {predicted_label}")

# %%
