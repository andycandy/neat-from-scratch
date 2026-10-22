import dataclasses

import numpy as np

from neat.config import NeatConfig
from neat.genes import Genome


def _pick_one_parent(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    return a if rng.random() < 0.5 else b

def crossover(fitter: Genome, other: Genome, cfg: NeatConfig,
              rng: np.random.Generator) -> Genome:
    """Child structure comes from `fitter`. Caller orders the parents (random if tied)."""
    conns = dict(fitter.conns)
    nodes = dict(fitter.nodes)
    commonConn = fitter.conns.keys() & other.conns.keys()
    for c in commonConn:
        a = _pick_one_parent(fitter, other, rng)
        faulty = fitter.conns[c].enabled & other.conns[c].enabled
        if faulty is False:
            faulty = rng.random() >= 0.75   # disabled in either parent: 75% stay disabled
        conns[c] = dataclasses.replace(a.conns[c], enabled=faulty)
        
    commonNodes = fitter.nodes.keys() & other.nodes.keys()
    for c in commonNodes:
        a = _pick_one_parent(fitter, other, rng)
        nodes[c] = a.nodes[c]
    return Genome(id=fitter.id, nodes=nodes, conns=conns,
                  conns_broken=dict(fitter.conns_broken))
