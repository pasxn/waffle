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
        # Initialize the graph dictionary with nodes as keys and empty lists as values
        for node in self.linearized_list:
            self.graph[node] = []
        
        # Iterate through nodes to establish connections based on input and output names
        for node in self.linearized_list:
            for output_name in node.output:
                # Find nodes that have this output name as an input
                for connected_node in self.linearized_list:
                    if output_name in connected_node.input:
                        # Add the connected node to the current node's connections
                        self.graph[node].append(connected_node)
        node.search_layer()
        
        if WFLDBG:
            print('\n------------------ ONNX Graph Connections ------------------')
            for node, connected_nodes in self.graph.items():
              print(f"{node.name} -> {[n.name for n in connected_nodes]}")


    def run(self, input: tensor) -> tensor:
            self.input_tensor = input
            visited = {node: False for node in self.graph.keys()}

            def dfs(node):
                visited[node] = True

                # Recursively visit connected nodes
                for connected_node in self.graph[node]:
                    if not visited[connected_node]:
                        dfs(connected_node)

                # Compute the output of the current node using its compute_node method
                if node.traverse_input is None:
                    node.compute_node(self.input_tensor)
                elif len(node.traverse_input) == 1:
                    input_node = self.graph[node][0]
                    node.compute_node(input_node.output_computed)
                elif len(node.traverse_input) == 2:
                    input_node1, input_node2 = self.graph[node]
                    node.compute_node(input_node1.output_computed, input_node2.output_computed)
                elif len(node.traverse_input) == 3:
                    input_node1, input_node2, input_node3 = self.graph[node]
                    node.compute_node(input_node1.output_computed, input_node2.output_computed, input_node3.output_computed)

            # Find the starting nodes (nodes with no incoming connections)
            start_nodes = [node for node, connected_nodes in self.graph.items() if not any(connected_node in self.graph for connected_node in connected_nodes)]

            for start_node in start_nodes:
                if not visited[start_node]:
                    dfs(start_node)

            result = self.linearized_list[-1].output_computed

            # Reset output_computed for all nodes
            for node in self.linearized_list:
                node.output_computed = None

            return result