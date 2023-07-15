import random

class Node:
  def __init__(self, Value):
    self.value= Value
    self.neighbours = []

class Graph:
  def __init__(self, numberofnodes):
    self.numberofnodes= numberofnodes
    self.nodes =(Node(random.randint(1, 10)) for Node in range(numberofnodes)) 
    #number of nodes with random value 1-10 & in range of numofnodes
    
  def Nodeconnections(self):
    for Node1 in range (self.numberofnodes):
      for Node2 in range (Node1+1, self.numberofnodes):
        self.nodes[Node1].neighbours.append(Node2)
        self.nodes[Node2].neighbours.append(Node1)
    
  def Graphtravesal(self):
    
    pass