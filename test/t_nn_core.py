from waffle import nn

model = nn.Module('mnist_cnn', './models/mnist_cnn/mnist_cnn.onnx')
model.compile()

model = nn.Module('resnet18', './models/resnet18/resnet18.onnx')
model.compile()
