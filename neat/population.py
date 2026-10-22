import csv
from collections.abc import Callable

import numpy as np

from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.innovation import InnovationTracker
from neat.reproduction import reproduce
from neat.species import Species, adjust_threshold, speciate


def make_initial_population(cfg: NeatConfig, tracker: InnovationTracker,
                            rng: np.random.Generator) -> list[Genome]:
    n_in, n_out = cfg.num_inputs, cfg.num_outputs
    pop = []
    for key in range(cfg.pop_size):
        nodes = {i: NodeGene(i, NodeType.INPUT, None, 0.0) for i in range(n_in)}
        for o in range(n_in, n_in + n_out):
            nodes[o] = NodeGene(o, NodeType.OUTPUT, cfg.output_activation, 0.0)
        conns = {}
        for i in range(n_in):
            for o in range(n_in, n_in + n_out):
                inno = tracker.conn_innovation(i, o)
                w = float(np.clip(rng.normal(0, 1), -cfg.w_max, cfg.w_max))
                conns[inno] = ConnGene(inno, i, o, w, True)
        pop.append(Genome(key, nodes, conns, {}))
    return pop


def _hidden(g: Genome) -> list[NodeGene]:
    return [n for n in g.nodes.values() if n.type == NodeType.HIDDEN]


def _enabled(g: Genome) -> int:
    return sum(c.enabled for c in g.conns.values())


class Population:
    def __init__(self, cfg: NeatConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker(cfg.num_inputs + cfg.num_outputs)
        self.genomes = make_initial_population(cfg, self.tracker, self.rng)
        self.species: list[Species] = []
        self.generation = 0
        self.next_genome_key = cfg.pop_size       # ids 0..pop_size-1 are taken
        self.next_species_key = 0
        self.best: Genome | None = None           # best ever
        self.champ: Genome | None = None          # best of the latest evaluated generation
        self.stats: list[dict] = []

    def run(self, fitness_fn: Callable[[list[Genome]], np.ndarray], generations: int,
            stop: Callable[[Genome], bool] | None = None) -> Genome:
        """Evolve. If any genome satisfies `stop`, return the fittest one that did
        (self.generation is then the first generation where one passed); else the best ever."""
        for _ in range(generations):
            fits = fitness_fn(self.genomes)
            for g, f in zip(self.genomes, fits):
                g.fitness = float(f)
            self.champ = max(self.genomes, key=lambda g: g.fitness)
            if self.best is None or self.champ.fitness > self.best.fitness:
                self.best = self.champ            # safe: reproduce makes new objects

            passed = [g for g in self.genomes if stop(g)] if stop else []
            if passed:
                self._log(fits)
                return max(passed, key=lambda g: g.fitness)

            self.species, self.next_species_key = speciate(
                self.genomes, self.species, self.cfg, self.generation, self.next_species_key)
            self._log(fits)
            if self.cfg.dynamic_threshold:
                self.cfg = adjust_threshold(self.cfg, len(self.species))
            self.genomes, self.species, self.next_genome_key = reproduce(
                self.species, self.cfg, self.tracker, self.rng, self.generation,
                self.next_genome_key)
            self.generation += 1
        return self.best

    def _log(self, fits: np.ndarray) -> None:
        """One stats row per generation (feeds the complexification/activation plots)."""
        acts = {f"act_{a}": 0 for a in self.cfg.activations}
        for g in self.genomes:
            for n in _hidden(g):
                acts[f"act_{n.activation}"] = acts.get(f"act_{n.activation}", 0) + 1
        self.stats.append({
            "gen": self.generation,
            "best": self.best.fitness,
            "champ": self.champ.fitness,
            "mean": float(np.mean(fits)),
            "n_species": len(self.species),
            "threshold": self.cfg.compat_threshold,
            "mean_hidden": float(np.mean([len(_hidden(g)) for g in self.genomes])),
            "mean_enabled_conns": float(np.mean([_enabled(g) for g in self.genomes])),
            "champ_hidden": len(_hidden(self.champ)),
            "champ_enabled_conns": _enabled(self.champ),
            **acts,
        })

    def save_stats(self, path: str) -> None:
        fields = list(dict.fromkeys(k for row in self.stats for k in row))   # union, in order
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval=0)
            w.writeheader()
            w.writerows(self.stats)
