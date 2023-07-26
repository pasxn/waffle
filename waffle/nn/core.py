from waffle import tensor
from waffle import ops
from waffle.onnx.onnxread import read_onnx
from waffle.onnx.onnxgraph import onnxGraph
from waffle.onnx.onnxlist import onnxList

class Module:
  def __init__(self, name:str, path:str, mode:str='list'):
    self.model_name = name
    self.onnx_obj = None
    self.linearized_model = read_onnx(path)
    self.onnx_obj = onnxList(self.linearized_model) if mode == 'list' else onnxGraph(self.linearized_model) 

  def compile(self):
    self.onnx_obj.hard_traverse()

  def run(self, x:tensor) -> tensor:
    return self.onnx_obj.run(x)


if __name__ == '__main__':
  model_path = './models/resnet50/resnet50.onnx'
  graph_nodes = read_onnx(model_path)

  for i in graph_nodes:
    print('==========================================')
    print(i.name)
    print(i.input)
    print(i.output)
    #print(i.params)