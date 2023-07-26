import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image

from models.mnist_cnn.cnn_model import Net
model = Net()

state_dict = torch.load('mnist_cnn.ckpt')
model.load_state_dict(state_dict)

# Define a function to load an image from a given path and predict its label
def predict_image(image_path):
  image = Image.open(image_path).convert('L')
  transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
  image = transform(image).unsqueeze(0)
  with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction

# Define the path of the image to be tested
image_path = '../../extra/images/mnist.jpg'

# Predict the label of the image
predicted_label = predict_image(image_path)

# Print the predicted label
print('Predicted label:', predicted_label)
