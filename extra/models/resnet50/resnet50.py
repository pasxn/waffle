import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50 as ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform_train = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(32, padding=4),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

net = ResNet50(10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

EPOCHS = 50
for epoch in range(EPOCHS):
  losses = []
  running_loss = 0
  correct = 0
  total = 0
  for i, inp in enumerate(trainloader):
    inputs, labels = inp
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
        
    running_loss += loss.item()
        
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total

    if i%100 == 0 and i > 0:
      print(f'Loss [{epoch+1},Accuracy = {train_acc:.2f}, {i}](epoch, minibatch): ', running_loss / 100)
      running_loss = 0.0

  avg_loss = sum(losses)/len(losses)
  scheduler.step(avg_loss)
            
print('Training Done')

net.eval()
correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
        
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')

input_shape = torch.randn(128, 3, 32, 32).to(device)
torch.onnx.export(net, input_shape, 'resnet50.onnx')

import torch

cuda_version = torch.version.cuda
print("PyTorch CUDA version:", cuda_version)
print('device = ',device)