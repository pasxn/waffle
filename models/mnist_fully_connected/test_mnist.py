import torch
from torch.utils.data import DataLoader  #dataset management
import torchvision.datasets as datasets  #to import MNist 
import torchvision.transforms as transforms 
import torch.onnx

from model_mnist import NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784; num_classes = 10; batch_size = 64 #64 imgs at a time

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

#load data using torchvision lib
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print("accuracy,train set: ")
check_accuracy(train_loader, model)
print("accuracy,test set: ")
check_accuracy(test_loader, model)
