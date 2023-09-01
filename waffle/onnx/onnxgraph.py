from typing import List
from waffle import tensor
from waffle.onnx.node import Node  
import os
from collections import deque
WFLDBG = os.environ.get('WFLDBG')
#WFLDBG = 1
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
      pass