from typing import List
from waffle import tensor
from waffle.onnx.node import Node  
import os

WFLDBG = os.environ.get('WFLDBG')
#edit spaces for 2
class onnxGraph:
    def __init__(self, linearized_list):
        self.nodes = [Node(node.name, node.input, node.output, node.op_type, node.attributes, node.params) for node in linearized_list]
        self.build_graph()
        self.hard_traverse()

    def build_graph(self):
        for i, node in enumerate(self.nodes):
            node.input_indices = []
            for input_name in node.input:
                for j, other_node in enumerate(self.nodes):
                    if input_name in other_node.output:
                        node.input_indices.append(j)    

    def hard_traverse(self):
        for i, node in enumerate(self.nodes):
            inputs = []
            if i == 0:
                inputs.append(-1)
            for input_index in node.input_indices:
                inputs.append(input_index)

            node.set_traverse_input(inputs)
            node.search_layer()
            
    def run(self, input):
        i = 0
        while self.nodes[-1].output_computed is None:
            current_node = self.nodes[i]
            current_node_traverse_input_len = len(current_node.traverse_input) if current_node.traverse_input is not None else 9999

            if current_node_traverse_input_len == 1:
                if current_node.traverse_input[0] == -1:
                    current_node.compute_node(input)
                else:
                    current_input_node = self.nodes[current_node.traverse_input[0]]
                    if current_input_node.output_computed is not None:
                        current_node.compute_node(current_input_node.output_computed)

            elif current_node_traverse_input_len == 2:
                current_input_node_0 = self.nodes[current_node.traverse_input[0]]
                current_input_node_1 = self.nodes[current_node.traverse_input[1]]
                if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None:
                    current_node.compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed)

            elif current_node_traverse_input_len == 3:
                current_input_node_0 = self.nodes[current_node.traverse_input[0]]
                current_input_node_1 = self.nodes[current_node.traverse_input[1]]
                current_input_node_2 = self.nodes[current_node.traverse_input[2]]
                if current_input_node_0.output_computed is not None and current_input_node_1.output_computed is not None and current_input_node_2.output_computed is not None:
                    current_node.compute_node(x=current_input_node_0.output_computed, y=current_input_node_1.output_computed, z=current_input_node_2.output_computed)

            else:
                current_node.compute_node(self.nodes[i - 1].output_computed)

            i = i + 1 if i < len(self.nodes) - 1 else 0

        result = self.nodes[-1].output_computed

        for node in self.nodes:
            node.output_computed = None

        return result
