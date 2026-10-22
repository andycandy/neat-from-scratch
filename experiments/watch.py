"""Watch an evolved agent play: saves a GIF and prints how it uses its buttons.

uv run python experiments/watch.py --run runs/slime3                 # best game of 20
uv run python experiments/watch.py --run runs/slime3 --best-of 50    # search more seeds
uv run python experiments/watch.py --run runs/slime3 --seed 123      # one specific game
  --rank 0 = best hall-of-fame agent (1 = second best, ...)
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle, Wedge  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neat.network import compile_genome, run  # noqa: E402
import __main__  # noqa: E402
import run_slime  # noqa: E402
from run_slime import INPUT_SCALE, load_latest_checkpoint, play  # noqa: E402

# checkpoints were written by run_slime.py running AS __main__, so pickle recorded
# the class as __main__.HallOfFame; point that name at the real class before loading
__main__.HallOfFame = run_slime.HallOfFame
from slime.slimevolley import SlimeVolleyEnv  # noqa: E402


def record_game(genome, seed: int):
    """Play one game; return per-step positions, actions and the final score."""
    env = SlimeVolleyEnv()
    env.seed(seed)
    obs = env.reset()
    env.policy.reset()
    net = compile_genome(genome)
    frames, actions, total, done = [], [], 0.0, False
    while not done:
        out = run(net, obs[None, :] / INPUT_SCALE)[0]
        action = (out > 0).astype(int)
        g = env.game
        frames.append((g.agent_right.x, g.agent_right.y, g.agent_left.x, g.agent_left.y,
                       g.ball.x, g.ball.y, g.agent_right.life, g.agent_left.life))
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        total += reward
    return np.array(frames), np.array(actions), total


def save_gif(frames: np.ndarray, path: Path, every: int = 3, fps: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_xlim(-24, 24)
    ax.set_ylim(0, 18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-24, 0), 48, 1.5, color="0.6"))                 # ground
    ax.add_patch(Rectangle((-0.5, 1.5), 1.0, 2.0, color="saddlebrown"))     # fence
    me = Wedge((0, 0), 1.5, 0, 180, color="tab:orange")                    # our agent (right)
    ai = Wedge((0, 0), 1.5, 0, 180, color="tab:blue")                      # built-in AI (left)
    ball = Circle((0, 0), 0.5, color="crimson")
    for p in (me, ai, ball):
        ax.add_patch(p)
    label = ax.text(0, 17, "", ha="center", va="top", fontsize=9)

    def draw(i):
        mx, my, ax_, ay, bx, by, my_lives, ai_lives = frames[i * every]
        me.set_center((mx, my))
        ai.set_center((ax_, ay))
        ball.center = (bx, by)
        label.set_text(f"AI (blue) lives {int(ai_lives)}    NEAT (orange) lives {int(my_lives)}"
                       f"    step {i * every}")
        return me, ai, ball, label

    anim = FuncAnimation(fig, draw, frames=len(frames) // every, blit=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slime2")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="play this seed (skips the search)")
    ap.add_argument("--best-of", type=int, default=20, help="search this many fresh seeds")
    args = ap.parse_args()

    pop, hof = load_latest_checkpoint(Path(args.run))
    score_hof, genome = hof.ranked()[args.rank]

    if args.seed is None:                       # find this agent's best game
        net = compile_genome(genome)
        env = SlimeVolleyEnv()
        seeds = range(10_000, 10_000 + args.best_of)   # never used in training
        games = [(play(net, env, s), s) for s in seeds]
        scores = [g[0][0] for g in games]
        print(f"searched {args.best_of} games (gen {pop.generation}): mean score {np.mean(scores):+.2f}, "
              f"best {max(scores):+.0f}, returns/game {np.mean([g[0][2] for g in games]):.1f}")
        (score, _, returns, _), args.seed = max(games, key=lambda g: (g[0][0], g[0][2]))
        print(f"best game: seed {args.seed}, score {score:+.0f}, {returns} returns")

    frames, actions, total = record_game(genome, args.seed)

    names = ["forward", "backward", "jump"]
    pressed = actions.mean(axis=0)
    print(f"hall-of-fame #{args.rank}: reliable score {score_hof:.2f}; this game {total:+.0f} "
          f"in {len(actions)} steps")
    print("button pressed (fraction of steps): " +
          ", ".join(f"{n} {p:.0%}" for n, p in zip(names, pressed)))
    combos, counts = np.unique(actions, axis=0, return_counts=True)
    top = sorted(zip(counts.tolist(), [tuple(int(v) for v in c) for c in combos]), reverse=True)[:3]
    print("most common [fwd, back, jump]: " + ", ".join(f"{c} {n / len(actions):.0%}" for n, c in top))
    x_moved = frames[:, 0].max() - frames[:, 0].min()
    print(f"agent x range over the game: {x_moved:.1f} (its whole half spans 2.0 to 22.5, i.e. 20.5)")

    out = Path(args.run) / f"gen{pop.generation:05d}_rank{args.rank}_seed{args.seed}.gif"
    save_gif(frames, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
