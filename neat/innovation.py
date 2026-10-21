from neat.genes import Genome
class InnovationTracker: 
    def __init__(self, first_hidden_id: int):
        self.conn_registry : dict[tuple[int,int],int] = {}
        self.split_registry : dict[tuple[int,int],int] = {} 
        self.next_node_id : int = first_hidden_id 
        self.next_innovation : int = 0
    
    def conn_innovation(self, src: int, dst:int) -> int:
        if (src, dst) in self.conn_registry:
            return self.conn_registry[(src,dst)]
        val = self.next_innovation
        self.next_innovation += 1
        self.conn_registry[(src,dst)] = val
        return val

    def split_node_id(self, conn_innovation: int, genome: Genome) -> int:
        key = (conn_innovation, genome.conns_broken.get(conn_innovation, 0))
        nid = self.split_registry.get(key)
        if nid is not None and nid not in genome.nodes:
            return nid
        fresh = self.next_node_id
        self.next_node_id += 1
        if nid is None:  # collision ids stay out of the registry
            self.split_registry[key] = fresh
        return fresh
