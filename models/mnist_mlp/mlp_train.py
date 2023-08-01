import torch
import torch.nn as nn    
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  #dataset management
import torchvision.datasets as datasets  #to import MNist 
import torchvision.transforms as transforms
import torch.onnx

from models.mnist_mlp.mlp_model import NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64 #64 imgs at a time
num_epochs = 1

#load data using torchvision lib
train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#arrange data into the model... initializing the network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss & Optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training the network
for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(train_loader):
    data = data.to(device=device)
    targets = targets.to(device=device)
        
    #converting the img metrix into a single dimention vector 
    data =data.reshape(data.shape[0], -1)
        
    #fowarding the data
    scores = model(data)
    loss = criterion(scores, targets)
        
    #backword
    optimizer.zero_grad()
    loss.backward()
        
    optimizer.step()

#save the model  Checkpoint
torch.save(model.state_dict(), 'models/mnist_fully_connected/mnist_mlp.ckpt')

#save in onnx format
ex_input = torch.randn(1, input_size)   #onnx require size and shape of a input
onnx_path = 'models/mnist_fully_connected/mnist_mlp.onnx'
torch.onnx.export(model,ex_input, onnx_path)
