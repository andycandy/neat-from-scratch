import dataclasses
from neat.genes import ConnGene, NodeType, NodeGene 
from neat.innovation import InnovationTracker

currentBox = InnovationTracker(5)


inp1 = NodeGene(id=1, activation=None, type=NodeType.INPUT, bias=0.0)
inp2 = NodeGene(id=2, activation=None, type=NodeType.INPUT, bias=0.0)
out1 = NodeGene(id=3, activation="sigmoid", type=NodeType.OUTPUT, bias=0.0)
hidden1 = NodeGene(id=4, activation="tanh", type=NodeType.HIDDEN, bias=0.0)

conn1 = ConnGene(src=1, dest=4, weight=-0.5, enabled=True, inno_id=currentBox.conn_innovation(1,4))
conn2 = ConnGene(src=2, dest=4, weight=-1, enabled=True, inno_id=currentBox.conn_innovation(2,4))
conn3 = ConnGene(src=4, dest=3, weight=2, enabled=True, inno_id=currentBox.conn_innovation(4,3))
conn4 = ConnGene(src=1, dest=3, weight=1, enabled=True, inno_id=currentBox.conn_innovation(1,3))
