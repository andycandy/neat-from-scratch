import csv
import dataclasses
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from neat.config import NeatConfig
from neat.genes import ConnGene, Genome, NodeGene, NodeType
from neat.innovation import InnovationTracker
from neat.reproduction import reproduce
from neat.species import Species, speciate


def make_initial_population(cfg: NeatConfig, tracker: InnovationTracker,
                            rng: np.random.Generator) -> list[Genome]:
    n_in, n_out = cfg.num_inputs, cfg.num_outputs
    pop = []
    nodes : dict[int, NodeGene] = {}
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


def adjust_threshold(cfg: NeatConfig, n_species: int) -> NeatConfig:
    """Raise the species threshold if there are too many species, lower it if too few."""
    lo, hi = cfg.target_species
    t = cfg.compat_threshold
    if n_species > hi:
        t = min(cfg.threshold_max, t + cfg.threshold_step)
    elif n_species < lo:
        t = max(cfg.threshold_min, t - cfg.threshold_step)
    return dataclasses.replace(cfg, compat_threshold=t)   # cfg is frozen: build a new one


def _hidden(g: Genome) -> list[NodeGene]:
    return [n for n in g.nodes.values() if n.type == NodeType.HIDDEN]


def _enabled(g: Genome) -> int:
    return sum(c.enabled for c in g.conns.values())


class Population:
    def __init__(self, cfg: NeatConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker(cfg.num_inputs + cfg.num_outputs)
        self.genomes: list[Genome] = make_initial_population(cfg=cfg, tracker=self.tracker, rng=self.rng)
        self.species: list[Species] = []
        self.generation = 0
        self.next_genome_key = cfg.pop_size       # ids 0..pop_size-1 are taken
        self.next_species_key = 0
        self.best: Genome | None = None
        self.stats: list[dict] = []

    def run(self, fitness_fn: Callable[[list[Genome]], np.ndarray], generations: int,
            stop: Callable[[Genome], bool] | None = None,
            snapshot_dir: str | None = None, snapshot_every: int = 1) -> Genome:
        """Evolve. If any genome satisfies `stop`, returns the fittest such genome, and
        self.generation is the first generation where one passed. Otherwise returns the best ever."""
        if snapshot_dir is not None:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        for _ in range(generations):
            fits = fitness_fn(self.genomes)
            for g, f in zip(self.genomes, fits):
                g.fitness = float(f)

            champ = max(self.genomes, key=lambda g: g.fitness)
            if self.best is None or champ.fitness > self.best.fitness:
                self.best = champ                 # safe: reproduce makes new objects

            if snapshot_dir is not None and self.generation % snapshot_every == 0:
                with open(Path(snapshot_dir) / f"gen_{self.generation:04d}.pkl", "wb") as f:
                    pickle.dump(champ, f)         # this generation's best, not best-ever

            # any genome may pass, not just the top-fitness one
            passed = [g for g in self.genomes if stop(g)] if stop is not None else []
            stopping = bool(passed)
            if not stopping:
                self.species, self.next_species_key = speciate(
                    cfg=self.cfg, generation=self.generation, genomes=self.genomes,
                    next_key=self.next_species_key, species_list=self.species
                )
            self._log(fits, champ)
            if stopping:
                return max(passed, key=lambda g: g.fitness)
            if self.cfg.dynamic_threshold:
                self.cfg = adjust_threshold(self.cfg, len(self.species))

            self.genomes, self.species, self.next_genome_key = reproduce(
                species_list=self.species, cfg=self.cfg, generation=self.generation,
                rng=self.rng, tracker=self.tracker, next_key=self.next_genome_key,
            )
            self.generation += 1
        assert self.best is not None
        return self.best

    def _log(self, fits: np.ndarray, champ: Genome) -> None:
        """One stats row for the current generation."""
        acts = {f"act_{a}": 0 for a in self.cfg.activations}   # every column in every row
        for g in self.genomes:
            for n in _hidden(g):
                key = f"act_{n.activation}"
                acts[key] = acts.get(key, 0) + 1
        self.stats.append({
            "gen": self.generation,
            "best": self.best.fitness if self.best else float("nan"),
            "champ": champ.fitness,
            "mean": float(np.mean(fits)),
            "n_species": len(self.species),
            "threshold": self.cfg.compat_threshold,
            "mean_hidden": float(np.mean([len(_hidden(g)) for g in self.genomes])),
            "mean_enabled_conns": float(np.mean([_enabled(g) for g in self.genomes])),
            "champ_hidden": len(_hidden(champ)),
            "champ_enabled_conns": _enabled(champ),
            **acts,
        })

    def save_stats(self, path: str) -> None:
        fields: list[str] = []                    # union of keys, first-seen order
        for row in self.stats:
            fields += [k for k in row if k not in fields]
        with open(path, "w", newline="") as f:    # with = RAII: file closes automatically
            w = csv.DictWriter(f, fieldnames=fields, restval=0)
            w.writeheader()
            w.writerows(self.stats)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)                  # whole object: rng, tracker, species, all

    @staticmethod
    def load(path: str) -> "Population":
        with open(path, "rb") as f:
            return pickle.load(f)
