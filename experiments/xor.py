"""XOR benchmark: run NEAT on several seeds and report generations-to-solve.

Run with:  uv run python experiments/xor.py
"""
import sys
from pathlib import Path

import numpy as np

# lets `import neat` work when this file is run directly from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neat.config import NeatConfig
from neat.genes import Genome, NodeType
from neat.network import compile_genome, run
from neat.population import Population
from neat.visualize import draw_genome

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)


def outputs(g: Genome) -> np.ndarray:
    """The genome's 4 outputs, one per XOR case. Shape (4,)."""
    net = compile_genome(g)
    return run(net, X)[:, 0]  


def fitness(genomes: list[Genome]) -> np.ndarray:
    """One score per genome: (4 - total absolute error) ** 2. Max 16."""
    return np.array([(4 - np.abs(outputs(g) - Y).sum()) ** 2 for g in genomes])


def solved(g: Genome) -> bool:
    """All 4 outputs on the correct side of 0.5."""
    return bool(np.all((outputs(g) > 0.5) == (Y > 0.5)))


def main(seeds=range(10), generations: int = 300) -> None:
    cfg = NeatConfig(num_inputs=2, num_outputs=1)   # output_activation defaults to sigmoid
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    solved_at = []
    for s in seeds:
        pop = Population(cfg, s)
        best = pop.run(fitness, generations, stop=solved)   # stops at first solved gen
        hidden_count = sum(n.type == NodeType.HIDDEN for n in best.nodes.values())
        ok = solved(best)
        print(f"seed {s}: gen {pop.generation:3d}  fitness {best.fitness:6.3f}  "
              f"hidden {hidden_count}  solved {ok}  outputs {np.round(outputs(best), 3)}")
        pop.save_stats(str(out_dir / f"xor_seed{s}.csv"))
        draw_genome(best, str(out_dir / f"xor_seed{s}.png"), input_names=["x1", "x2"],
                    output_names=["xor"], title=f"XOR seed {s}, solved at gen {pop.generation}")
        if ok:
            solved_at.append(pop.generation)
    print(f"solved {len(solved_at)}/{len(seeds)}")
    if solved_at:
        print(f"generations to solve: {np.mean(solved_at):.1f} +- {np.std(solved_at):.1f}")


if __name__ == "__main__":   # runs main() only when the file is executed, not imported
    main()
