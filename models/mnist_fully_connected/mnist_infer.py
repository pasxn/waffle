import torch
from torch.utils.data import DataLoader  #dataset management
import torchvision.datasets as datasets  #to import MNist 
import torchvision.transforms as transforms 
import torch.onnx

from models.mnist_fully_connected.mnist_model import NN
from models.mnist_fully_connected.mnist_util import input_size, num_classes, device, train_loader, test_loader


model = NN(input_size=input_size, num_classes=num_classes).to(device)

state_dict = torch.load('MNist.ckpt')
model.load_state_dict(state_dict)

def check_accuracy(loader, model):
  num_correct = 0
  num_samples = 0
  model.eval()
    
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.reshape(x.shape[0], -1)
            
      scores = model(x)
      _, predictions = scores.max(1)  #geting the maximum values from the scores (digit)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
            
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
   
  model.train()

print("accuracy train set: ")
check_accuracy(train_loader, model)
print("accuracy test set: ")
check_accuracy(test_loader, model)
