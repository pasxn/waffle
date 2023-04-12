#%%
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

#%%
# Define the data transformation
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#%%
# Load the Tiny ImageNet dataset
dataset = ImageFolder('tiny-imagenet-200/test', transform=data_transform)

#%%
# Create a data loader for the dataset
data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

#%%
# Load the pre-trained ResNet-50 model
#model = models.resnet50(pretrained=True)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

#%%
# Set the model to evaluation mode
model.eval()

#%%
# Create a loss criterion 
criterion = nn.CrossEntropyLoss()

#%%
# variables for tracking accuracy and loss
total_correct = 0
total_loss = 0
total_images = 0

#%%
# Evaluate the model on the dataset
with torch.no_grad():
    for images, labels in data_loader:
        # Move the data to GPU (available)
        #if torch.cuda.is_available():
        #   images = images.cuda()
        #  labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Update total loss
        total_loss += loss.item()

        # Get predicted labels
        _, predicted = torch.max(outputs, 1)

        # Update total correct predictions
        total_correct += (predicted == labels).sum().item()

        # Update total images
        total_images += labels.size(0)
        print("Total number of evaluated images: ", total_images)

#%%
# Calculate accuracy and average loss
accuracy = total_correct / total_images
avg_loss = total_loss / len(data_loader)

# Print results
print('Accuracy: {:.2%}'.format(accuracy))
print('Average Loss: {:.4f}'.format(avg_loss))

# %%
# Export the model to ONNX format
input_tensor = torch.randn(1, 3, 224, 224)  # Assumes input images of size 224x224 with 3 channels
onnx_file_path = "resnet50.onnx"
torch.onnx.export(model, input_tensor, onnx_file_path, export_params=True, opset_version=11)

print(f"ResNet50 model has been saved to '{onnx_file_path}' in ONNX format.")
# %%
