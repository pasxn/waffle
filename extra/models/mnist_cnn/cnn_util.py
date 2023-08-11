import torch
from torchvision import datasets, transforms, models

# Define transforms for the data
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])
#values (0.1307,) and (0.3081,) are the mean and standard deviation of the MNIST dataset, 


# Load the MNIST dataset
train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data/', train=False, download=True, transform=transform)

# Define data loaders for the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
