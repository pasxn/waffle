import torch
import torch.onnx

import os
from extra.models.mnist_mlp.mlp_model import NN
from extra.models.mnist_mlp.mlp_util import input_size, num_classes, device, train_loader, test_loader


model = NN(input_size=input_size, num_classes=num_classes).to(device)
path = os.path.abspath(os.path.dirname(__file__))

state_dict = torch.load(path + '/mnist_mlp.ckpt')
model.load_state_dict(state_dict)

def predict_image_mlp(image):
  with torch.no_grad():
    output = model(image)
    return output

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
            
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} %')
   
  model.train()
if __name__ == '__main__':
  print("accuracy train set: ", end='')
  check_accuracy(train_loader, model)
  print("accuracy test set : ", end='')
  check_accuracy(test_loader, model)
