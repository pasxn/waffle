
from waffle.onnx import Node

# return the whole linearized graph as a list of Nodes
import onnx
import numpy as np

def read_onnx(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Store node details
    nodes = []

    # Iterate over nodes
    for i, node in enumerate(model.graph.node):
        attributes = []
        for attr in node.attribute:
            attribute = {}
            attribute['name'] = attr.name
            attribute['type'] = attr.type

            if attr.type == onnx.AttributeProto.FLOATS:
                attribute['values'] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                attribute['values'] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.STRING:
                attribute['values'] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.TENSOR:
                tensor_data = np.frombuffer(attr.t.raw_data, dtype=np.float32)
                attribute['values'] = tensor_data.reshape(attr.t.dims)

            attributes.append(attribute)

        # Input weight tensors
        weight_tensors = []
        for input_name in node.input:
            for initializer in model.graph.initializer:
                if input_name == initializer.name:
                    weight_tensor = {}
                    weight_tensor['name'] = initializer.name
                    weight_tensor['shape'] = initializer.dims

                    weight_array = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    weight_tensor['values'] = weight_array.reshape(initializer.dims)

                    weight_tensors.append(weight_tensor)

        # node_weight = None; node_bias = None
        # assert len(weight_tensors) == 2, 'there are more tan 2 weight tensors!'
        # for i in weight_tensors:
        #     node_weight = i if i['name'] == 'weight' else None
        #     node_bias   = i if i['name'] == 'bias' else None

        nodes.append(Node(node.name, node.input, node.output, node.op_type, attributes, weight_tensors, bias = None))

    return nodes

# # Access the graph nodes
# for i, node_info in enumerate(graph_nodes):
#     print("Node Details:")
#     print(f"========================================== Node {i}:==========================================")
#     print("Name:", node_info['name'])
#     print()
#     print("Input Nodes:", node_info['input'])
#     print()
#     print("Output Nodes:", node_info['output'])
#     print()
#     print("Op Type:", node_info['op_type'])
#     print()
#     print("Attributes:")
#     for attr in node_info['attributes']:
#         print(f"- Name: {attr['name']}")
#         print(f"  Type: {attr['type']}")
#         if 'values' in attr:
#             if isinstance(attr['values'], list):
#                 print(f"  Values: {attr['values']}")
#             else:
#                 print(f"  Value: {attr['values']}")
#         print()
#     print()
#     print("Weight Tensors:")
#     for weight_tensor in node_info['weight_tensors']:
#         print("Weight tensor name:", weight_tensor['name'])
#         print("Shape:", weight_tensor['shape'])
#         print("Weight Values:", weight_tensor['values'])
#         print()
#     print()