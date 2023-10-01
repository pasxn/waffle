from typing import List
from waffle import tensor
from waffle.onnx.node import Node  
import os
from collections import deque
WFLDBG = os.environ.get('WFLDBG')
WFLDBG = 1
class onnxGraph:
    def __init__(self, linearized_list: List[Node]):
        self.linearized_list = linearized_list
        self.graph = {}  # Initialize an empty graph as a dictionary

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
        # Find the nodes with no incoming connections to start DFS
        start_nodes = [node for node, connections in self.graph.items() if not connections]

        # Dictionary to store computed node outputs
        computed_outputs = {}

        # Define a DFS function
        def dfs(node):
            # If the output is already computed, return it
            if node in computed_outputs:
                return computed_outputs[node]

            # Calculate the input tensors based on the number of inputs (0 to 3)
            inputs = []
            for connected_node in self.graph[node]:
                input_tensor = dfs(connected_node)
                inputs.append(input_tensor)

            # Calculate the output of the current node
            if len(inputs) == 1:
                node.compute_node(inputs[0])
            elif len(inputs) == 2:
                node.compute_node(inputs[0], inputs[1])
            elif len(inputs) == 3:
                node.compute_node(inputs[0], inputs[1], inputs[2])
            else:
                node.compute_node(input)

            # Store the computed output
            computed_outputs[node] = node.output_computed
            print(f'Node: {node.name.rjust(15)}   Output shape: {node.output_computed.shape if node.output_computed is not None else "None"}')
            return node.output_computed

        # Initialize the result by performing DFS on start nodes
        result = dfs(start_nodes[0])

        return result
