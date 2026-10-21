from enum import Enum
from dataclasses import dataclass

class NodeType(Enum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2

@dataclass(frozen=True)
class NodeGene:
    id: int
    type : NodeType
    activation : str | None
    bias: float

@dataclass(frozen=True)
class ConnGene:
    inno_id : int
    src : int
    dest : int
    weight : float
    enabled: bool

@dataclass
class Genome:
    id: int
    nodes: dict[int, NodeGene]
    conns: dict[int, ConnGene]
    conns_broken: dict[int,int]
    fitness: float | None = None
