from waffle.onnx.onnxread import read_onnx

model_path = 'models/resnet50/resnet50.onnx'
graph_nodes = read_onnx(model_path)

print(graph_nodes)
print()
print()
print()
print(len(graph_nodes))
print()
print()
