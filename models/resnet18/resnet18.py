import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = ',device)

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


#Load pre-trained ResNet-18 model
model = resnet18(pretrained=True)

# Modify final layer to have 10 outputs
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Freeze all layers except the final fully connected layer
for name, param in model.named_parameters():
  if name not in ['fc.weight', 'fc.bias']:
    param.requires_grad = False

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#Training loop
model.train()
num_epochs = 10

for epoch in range(num_epochs):
  running_loss = 0.0
  correct = 0
  total = 0

  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    if i % 2000 == 1999:
      batch_loss = running_loss / 2000
      batch_acc = 100 * correct / total
      print('[%d, %5d] loss: %.3f, accuracy: %.2f%%' % (epoch + 1, i + 1, batch_loss, batch_acc))
      running_loss = 0.0
      correct = 0
      total = 0

print('Training finished.')

# Evaluation loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Print accuracy
print('Accuracy of the network on the 10000 test images: {:.2f}%'.format(100 * correct / total))

#save in onnx format
input_shape = torch.randn(4, 3, 224, 224).to(device)
torch.onnx.export(model, input_shape, 'resnet18.onnx')