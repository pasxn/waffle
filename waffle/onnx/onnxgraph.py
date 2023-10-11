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
        visited = set()  # To keep track of visited nodes
        stack = []       # Stack to store nodes to be processed
        result = None    # To store the final result

        # Start from the first node in the graph
        start_node = next(iter(self.graph.keys()))

        while start_node:
            # If the node has not been visited yet
            if start_node not in visited:
                visited.add(start_node)

                # Determine the inputs for the current node
                inputs = []
                for input_name in start_node.input:
                    for connected_node in self.graph[start_node]:
                        if input_name in connected_node.output:
                            if connected_node not in visited:
                                # If a connected node has not been visited, add it to the stack
                                stack.append(connected_node)
                            inputs.append(connected_node.output_computed)

                # Compute the node based on the number of inputs
                if len(inputs) == 1:
                    start_node.compute_node(inputs[0])
                elif len (inputs) == 2:
                    start_node.compute_node(inputs[0], inputs[1])
                elif len(inputs) == 3:
                    start_node.compute_node(inputs[0], inputs[1], inputs[2])
                else:
                    start_node.compute_node(input)

                # Update the result
                result = start_node.output_computed

            # Check if there are more nodes in the stack to visit
            if stack:
                start_node = stack.pop()
            else:
                # If there are no more nodes in the stack, break the loop
                break

        return result
