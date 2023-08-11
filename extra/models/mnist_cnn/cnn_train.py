import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from extra.models.mnist_cnn.cnn_model import Net
from extra.models.mnist_cnn.cnn_util import train_loader, test_loader 

# Instantiate the model and define the optimizer
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# Train the model
def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

# Test the model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))


# Train the model for 10 epochs and test it
for epoch in range(1):
  train(epoch)
  test()

#save the model  Checkpoint
torch.save(model.state_dict(), 'mnist_cnn.ckpt')

#save in onnx format
input_shape = (1, 1, 28, 28)
torch.onnx.export(model, torch.randn(*input_shape), 'mnist_cnn.onnx')
