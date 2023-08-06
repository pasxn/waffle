import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

from extra.models.resnetlike.resnetlike_model import ResNet
from extra.models.resnetlike.resnetlike_util import train_loader, test_loader 

# Initialize the ResNet model
model = ResNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

#save the model  Checkpoint
torch.save(model.state_dict(), 'resnetlike.ckpt')
print("Model successfully saved in CKPT format!")

# Save the model in ONNX format
dummy_input = torch.randn(1, 1, 28, 28).to(device)
input_names = ['input']
output_names = ['output']
onnx_path = 'resnetlike_mnist.onnx'
torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names)
print("Model successfully saved in ONNX format!")
