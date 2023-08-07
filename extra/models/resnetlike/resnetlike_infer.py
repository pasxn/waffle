import os
import torch
from torchvision import transforms
from PIL import Image

from extra.models.resnetlike.resnetlike_model import ResNet

model = ResNet()
path = os.path.abspath(os.path.dirname(__file__))

state_dict = torch.load(path + '/resnetlike_mnist.ckpt')
model.load_state_dict(state_dict)

def predict_image_cnn(image):
  with torch.no_grad():
    output = model(image)
    return output

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

if __name__ == '__main__':
  # Define the path of the image to be tested
  image_path = './extra/images/mnist.jpg'

  # Predict the label of the image
  predicted_label = predict_image(image_path)

  # Print the predicted label
  print('Predicted label:', predicted_label)
