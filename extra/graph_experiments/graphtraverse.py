import random

class Graph:
  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    self.adj_list = {i: [] for i in range(num_nodes)}
    self.node_values = [random.randint(0, 10) for _ in range(num_nodes)]

  def add_edge(self, u, v):
    self.adj_list[u].append(v)

  def generate_random_connections(self):
    remaining_nodes = list(range(self.num_nodes))
    for node in range(self.num_nodes - 1):
      neighbors_count = random.randint(1, min(10, len(remaining_nodes) - 1))
      neighbors = random.sample(remaining_nodes, neighbors_count)
      remaining_nodes.remove(node)    #  no self-loop
      for neighbor in neighbors:
        self.add_edge(node, neighbor)

  def dfs(self, node, visited, summation):
    visited[node] = True
    summation[0] += self.node_values[node]

    for neighbor in self.adj_list[node]:
      if not visited[neighbor]:
        self.dfs(neighbor, visited, summation)

  def traverse(self, start_node=0, end_node=None):
    if end_node is None:
      end_node = self.num_nodes - 1

    visited = [False] * self.num_nodes
    summation = [0]
    self.dfs(start_node, visited, summation)

    return summation[0], sum(visited)


# main
graph = Graph(1000)
graph.generate_random_connections()

# Calculate the known summation of the random values
known_summation = sum(graph.node_values)

# Traverse the graph and get the summation of random values during traversal
traversal_summation = graph.traverse()

# Traverse the graph and get the summation of random values during traversal
traversal_summation, num_traversed_nodes = graph.traverse()

# Compare the results
print("Known Summation:", known_summation)
print("Traversed Summation:", traversal_summation)
print("Number of Nodes Traversed:", num_traversed_nodes)