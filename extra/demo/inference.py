#%% mnist_cnn test                   Load the .ckpt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Define the CNN architecture
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
model.load_state_dict(torch.load('../../models/mnist_cnn/MNIST_CNN.ckpt'))
model.eval()

# Preprocess the test image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

test_image_path = '../../../../testSample/img_13.jpg'
test_image = Image.open(test_image_path)
test_image = transform(test_image).unsqueeze(0)

#plot image
image = mpimg.imread(test_image_path)
plt.imshow(image)
plt.axis('off')  # Remove axis ticks
plt.show()

# Make a prediction
with torch.no_grad():
    output = model(test_image)
    _, predicted = torch.max(output.data, 1)

predicted_label = predicted.item()

print(f"Predicted Label: {predicted_label}")




#%% Mnist Fully connected Test            Load the .ckpt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F  

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 500
num_classes = 10

# Load the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Load the saved model parameters
model.load_state_dict(torch.load('../../models/mnist_fully_connected/model.ckpt'))
print("Model parameters loaded successfully.")

# Load the custom image
custom_image = Image.open('../../../../testSample/img_10.jpg')  # Replace 'path_to_your_image.jpg' with the actual path to your image
custom_image = custom_image.convert('L')  # Convert the image to grayscale
custom_image = custom_image.resize((28, 28))  # Resize the image to 28x28 pixels

# Convert the image to a numpy array
custom_image_array = np.array(custom_image)

# Flatten the image array and normalize its values
custom_image_tensor = torch.tensor(custom_image_array.flatten() / 255.0, dtype=torch.float32)

# Reshape the tensor to match the input size expected by the model
custom_image_tensor = custom_image_tensor.view(1, -1).to(device)

# Pass the custom image through the model
model.eval()
with torch.no_grad():
    output = model(custom_image_tensor)

# Get the predicted class label
_, predicted = torch.max(output.data, 1)
predicted_label = predicted.item()

print('Predicted label:', predicted_label)

