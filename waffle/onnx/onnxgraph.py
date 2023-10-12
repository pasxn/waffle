from typing import List
from waffle import tensor
from waffle.onnx.node import Node  
import os
from collections import deque
import numpy as np
WFLDBG = os.environ.get('WFLDBG')
WFLDBG = 1
class onnxGraph:
    def __init__(self, linearized_list: List[Node]):
        self.linearized_list = linearized_list
        self.graph = {}  # Name the graph dictionary as 'graph'

    def hard_traverse(self):
        # Initialize the graph dictionary with nodes as keys and Node objects as values
        for node in self.linearized_list:
            self.graph[node] = []

        # Iterate through nodes to establish connections based on input and output names
        for node in self.linearized_list:
            for output_name in node.output:
                # Find nodes that have this output name as an input
                for connected_node in self.linearized_list:
                    if output_name in connected_node.input:
                        # Add the connected Node object to the current node's connections
                        self.graph[node].append(connected_node)

        # Set the compute_node method for each Node object based on the self.graph structure
        for node in self.graph:
            node.search_layer()

        if WFLDBG:
            print('\n------------------ ONNX Graph Connections ------------------')
            for node, connected_nodes in self.graph.items():
                print(f"{node.name} -> {[n.name for n in connected_nodes]}")

    def run(self, input: tensor) -> tensor:
        print('\n------------------ ONNX Graph Computation ------------------')

        # Initialize the result as the input tensor
        result = input

        # Keep track of nodes that need to be revisited
        nodes_to_revisit = []

        for node in self.linearized_list:
            if node in self.graph:
                # Calculate the input tensors based on the number of inputs (0 to 3)
                inputs = []
                for connected_node in self.graph[node]:
                    if connected_node.output_computed is not None:
                        inputs.append(connected_node.output_computed)

                # Check if all inputs are computed
                if all(input_tensor is not None for input_tensor in inputs):
                    # Calculate the output of the current node
                    if len(inputs) == 1:
                        node.compute_node(inputs[0])
                    elif len(inputs) == 2:
                        node.compute_node(inputs[0], inputs[1])
                    elif len(inputs) == 3:
                        node.compute_node(inputs[0], inputs[1], inputs[2])
                    else:
                        node.compute_node(result)  # Use the overall result as input

                    # Update the result with the current node's output
                    result = node.output_computed

                    # Print the computation progress
                    print(f'Node: {node.name.rjust(15)}   Output shape: {node.output_computed.shape if node.output_computed is not None else "None"}')

                    # Clear any nodes to revisit since the calculation is complete
                    nodes_to_revisit.clear()
                else:
                    # Not all inputs are computed, add the node to the revisit list
                    nodes_to_revisit.append(node)

        return result

