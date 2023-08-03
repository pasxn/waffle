import torch
import torch.nn.functional as F  
from torch.utils.data import DataLoader  #dataset management
import torchvision.datasets as datasets  #to import MNist 
import torchvision.transforms as transforms
import torch.onnx

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
