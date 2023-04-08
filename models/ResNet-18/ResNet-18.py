# %%
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn 

# %%
# Load the pre-trained ResNet-18 model from PyTorch's model zoo
resnet = models.resnet18(pretrained=True)

# Replace the last layer of the ResNet-18 model
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)

# %%
# Set the model to evaluation mode
resnet.eval()

# %%
# Define the test dataset and data loader
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# %%
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# %%
# Evaluate the model on the test dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# %%
# Print the accuracy on the test dataset
print('Accuracy on the test dataset: %d %%' % (100 * correct / total))

# %%
#export the model into ONNX format
import torch.onnx as onnx
dummy_input = torch.randn(1, 3, 224, 224)
output_file = "resnet18.onnx"
onnx.export(resnet, dummy_input, output_file, verbose=True)

# %%
