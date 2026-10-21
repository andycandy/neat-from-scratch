import dataclasses

import numpy as np
from neat.innovation import InnovationTracker
from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType, NodeGene


def _mutate_value(old: float, cfg: NeatConfig, rng: np.random.Generator) -> float:
    if rng.random() < cfg.p_perturb:
        new = old + rng.normal(0, cfg.sigma)
    else:
        new = rng.normal(0, 1)
    return float(np.clip(new, -cfg.w_max, cfg.w_max))


def mutate_weights(genome: Genome, cfg: NeatConfig, rng: np.random.Generator,
                   new_id: int | None = None) -> Genome:
    """Return a new genome with every weight and non-input bias perturbed or replaced."""
    conns = {k: dataclasses.replace(c, weight=_mutate_value(c.weight, cfg, rng))
             for k, c in genome.conns.items()}
    nodes = {k: n if n.type == NodeType.INPUT
             else dataclasses.replace(n, bias=_mutate_value(n.bias, cfg, rng))
             for k, n in genome.nodes.items()}
    return Genome(id=genome.id if new_id is None else new_id, nodes=nodes, conns=conns,
                  conns_broken=dict(genome.conns_broken))

def reachable(genome: Genome, a: int, b: int) -> bool:
    adj : dict[int, list[int]] = {k: [] for k in genome.nodes}
    for i in genome.conns.values():
        adj[i.src].append(i.dest)
    stack, seen = [a], {a}
    while stack:
        n = stack.pop()
        if n == b:
            return True
        for m in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False

def mutate_add_connection(genome: Genome, cfg: NeatConfig,
                          tracker: InnovationTracker, rng: np.random.Generator) -> Genome:
    """Try up to 20 random (src, dst) pairs; add the first valid one, else return a copy."""
    sources = [n.id for n in genome.nodes.values() if n.type != NodeType.OUTPUT]
    targets = [n.id for n in genome.nodes.values() if n.type != NodeType.INPUT]
    existing = {(c.src, c.dest) for c in genome.conns.values() if c.enabled}
    conns = dict(genome.conns)
    for _ in range(20):
        src, dst = int(rng.choice(sources)), int(rng.choice(targets))
        if src == dst or (src, dst) in existing or reachable(genome, dst, src):
            continue
        inno = tracker.conn_innovation(src, dst)
        w = float(np.clip(rng.normal(0, 1), -cfg.w_max, cfg.w_max))
        conns[inno] = ConnGene(inno, src, dst, w, True)
        break
    return Genome(id=genome.id, nodes=dict(genome.nodes), conns=conns,
                  conns_broken=dict(genome.conns_broken))

def add_node(genome: Genome, cfg: NeatConfig,
            tracker: InnovationTracker, rng: np.random.Generator) -> Genome :
    """Add a node between existing connection"""
    existing : list[ConnGene] = [c for c in genome.conns.values() if c.enabled]
    if len(existing) == 0: return genome
    chosen : ConnGene = existing[int(rng.integers(len(existing)))]
    src = chosen.src
    dest = chosen.dest
    conns = dict(genome.conns)
    nodes = dict(genome.nodes)
    nid = tracker.split_node_id(chosen.inno_id, genome)
    l1,l2 = tracker.conn_innovation(src, nid), tracker.conn_innovation(nid, dest)
    conns[l1] = ConnGene(inno_id=l1, src=src, dest=nid, weight=1, enabled=True)
    conns[l2] = ConnGene(inno_id=l2, src=nid, dest=dest, weight=chosen.weight, enabled=True)
    conns[chosen.inno_id] =  dataclasses.replace(chosen, enabled=False)
    nodes[nid] = NodeGene(activation="tanh", bias=0.0, id=nid, type=NodeType.HIDDEN)
    broken = dict(genome.conns_broken)
    broken[chosen.inno_id] = broken.get(chosen.inno_id, 0) + 1
    return Genome(id=genome.id, nodes=nodes, conns=conns, conns_broken=broken)


def mutate_activation(genome: Genome, cfg: NeatConfig, rng: np.random.Generator) -> Genome:
    """Give one random hidden node a random allowed activation."""
    hidden = [n.id for n in genome.nodes.values() if n.type == NodeType.HIDDEN]
    nodes = dict(genome.nodes)
    if hidden:
        nid = hidden[int(rng.integers(len(hidden)))]
        act = cfg.activations[int(rng.integers(len(cfg.activations)))]
        nodes[nid] = dataclasses.replace(nodes[nid], activation=act)
    return Genome(id=genome.id, nodes=nodes, conns=dict(genome.conns),
                  conns_broken=dict(genome.conns_broken))


def toggle_enable(genome: Genome, rng: np.random.Generator) -> Genome:
    """Flip one random connection's enabled flag. Safe: the all-edges graph is always acyclic."""
    conns = dict(genome.conns)
    if conns:
        k = list(conns)[int(rng.integers(len(conns)))]
        conns[k] = dataclasses.replace(conns[k], enabled=not conns[k].enabled)
    return Genome(id=genome.id, nodes=dict(genome.nodes), conns=conns,
                  conns_broken=dict(genome.conns_broken))


def mutate(genome: Genome, cfg: NeatConfig, tracker: InnovationTracker,
           rng: np.random.Generator) -> Genome:
    """Apply each mutation with its own probability. Never changes its input."""
    if rng.random() < cfg.p_weight:
        genome = mutate_weights(genome, cfg, rng)
    if rng.random() < cfg.p_add_conn:
        genome = mutate_add_connection(genome, cfg, tracker, rng)
    if rng.random() < cfg.p_add_node:
        genome = add_node(genome, cfg, tracker, rng)
    if rng.random() < cfg.p_activation:
        genome = mutate_activation(genome, cfg, rng)
    if rng.random() < cfg.p_toggle:
        genome = toggle_enable(genome, rng)
    return genome
