import random

class Graph:
  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    self.adj_list = {i: [] for i in range(num_nodes)}
    self.node_values = [random.randint(1, 10) for _ in range(num_nodes)]
    self.total_value = sum(self.node_values)
    self.visited = [False] * num_nodes
    self.traversal_order = []
    self.traversal_count = 0

  def add_edge(self, u, v):
    self.adj_list[u].append(v)
    self.adj_list[v].append(u)

  def generate_random_connections(self):
    remaining_nodes = list(range(self.num_nodes))
    for node in range(self.num_nodes - 1):
      neighbors_count = random.randint(1, min(10, len(remaining_nodes) - 1))
      neighbors = random.sample(remaining_nodes, neighbors_count)
      remaining_nodes.remove(node)    #  no self-loop
      for neighbor in neighbors:
        self.add_edge(node, neighbor)

  def dfs(self, node):
    if not self.visited[node]:
      self.visited[node] = True
      self.traversal_order.append(node)
      self.traversal_count += 1
      for neighbor in self.adj_list[node]:
        self.dfs(neighbor)

  def traverse(self):
    self.dfs(0) #(0th node)
    
    if all(self.visited) and len(self.traversal_order) == self.num_nodes:
      print("Traversal successful.")
    else:
      print("Traversal failed.")

    traversal_sum = sum(self.node_values[node] for node in self.traversal_order)
    print(f"Known value: {self.total_value}, Traversal sum: {traversal_sum}")
    print(f"Number of traversal nodes: {self.traversal_count}")

# main:
graph = Graph(1000)
graph.generate_random_connections()
graph.traverse()
