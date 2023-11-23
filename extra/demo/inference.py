# %%

'''
An ML inferemce framework as a python package
-------------------------------------------------

How to Download?

  git clone https://github.com/pasxn/waffle.git
  cd waffle

How to install?

  python -m pip install -e .    <-- This command registers the code as a pip package


'''

# %%

import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms

# our own API
from waffle import tensor
from waffle import nn





# %%

# Datalodaer utilities

# Define transforms for the data
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])
# (0.1307,) and (0.3081,) are the mean and standard deviation of the MNIST dataset 


# Load the MNIST dataset
test_dataset = datasets.MNIST('data/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)





# %%

# Fetiching a random input from the dataset and plotting
image = test_loader.dataset[int(np.random.random()*len(test_loader.dataset))]
image_tensor = image[0].squeeze().numpy()
plt.imshow(image_tensor)


# importing the model graphs
mlp = nn.Module('mlp', '../models/mnist_cnn/mnist_cnn.onnx')
cnn = nn.Module('cnn', '../models/mnist_cnn/mnist_cnn.onnx')
resnetlike = nn.Module('resnetlike', '../models/mnist_cnn/mnist_cnn.onnx')

# compiling the models
mlp.compile()
cnn.compile()
resnetlike.compile()

# infering the models
output_mlp = mlp.run(tensor(image_tensor))
output_cnn = cnn.run(tensor(image_tensor))
output_resnetlike = resnetlike.run(tensor(image_tensor))

# printing out the results
print(f"result from mlp: {output_mlp.where(output_mlp.max())}")
print(f"result from cnn: {output_cnn.where(output_cnn.max())}")
print(f"result from resnetlike: {output_resnetlike.where(output_resnetlike.max())}")
