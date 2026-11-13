"""
train.py - train the Q-learning agent, track its win rate, save a graph.

    python train.py                      # default: 10,000 episodes (about 5 seconds)
    python train.py --seed 3             # different random seed
"""
import argparse

import matplotlib
matplotlib.use("Agg")   # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from agent import QAgent, RandomAgent
from env import ArenaEnv, N_ACTIONS, TRAIN_STYLES


def play_episode(env, agent, style=None, learn=False):
    """Play one full episode. Returns the outcome: 'win', 'loss' or 'draw'."""
    state = env.reset(style)
    done = False
    info = {}
    while not done:
        action = agent.act(state, greedy=not learn)
        next_state, reward, done, info = env.step(action)
        if learn:
            agent.update(state, action, reward, next_state, done)
        state = next_state
    if learn:
        agent.end_episode()
    return info["outcome"]


def win_rate(env, agent, style, n):
    """Fraction of n greedy (no exploration) episodes won against one style."""
    wins = sum(play_episode(env, agent, style) == "win" for _ in range(n))
    return wins / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Learning happens fast, so evaluate often early on and rarely later
    # (log-spaced checkpoints, starting from the untrained agent at episode 0).
    checkpoint_set = {c for c in (0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
                      if c <= args.episodes} | {args.episodes}

    env = ArenaEnv(seed=args.seed)
    eval_env = ArenaEnv(seed=args.seed + 1)   # separate env so evaluation never disturbs training
    agent = QAgent(env.state_shape, N_ACTIONS, seed=args.seed)

    # Baseline: how often does a random agent win?
    rand = RandomAgent(N_ACTIONS, seed=args.seed)
    baseline = {s: win_rate(eval_env, rand, s, 1000) for s in TRAIN_STYLES}
    print("Random-agent win rate:", {s: f"{v:.0%}" for s, v in baseline.items()})

    checkpoints, history = [], {s: [] for s in TRAIN_STYLES}
    for ep in range(0, args.episodes + 1):
        if ep > 0:
            play_episode(env, agent, learn=True)      # opponent style is random each episode
        if ep in checkpoint_set:
            checkpoints.append(ep)
            row = []
            for s in TRAIN_STYLES:
                w = win_rate(eval_env, agent, s, args.eval_episodes)
                history[s].append(w)
                row.append(f"{s} {w:.0%}")
            print(f"episode {ep:>6}  eps={agent.epsilon:.2f}  win rate vs " + " | ".join(row))

    # Final, larger evaluation
    final = {s: win_rate(eval_env, agent, s, 2000) for s in TRAIN_STYLES}
    print("\nFINAL (2000 greedy episodes each)")
    for s in TRAIN_STYLES:
        print(f"  vs {s:<10} learned {final[s]:.0%}   random {baseline[s]:.0%}")

    # Graph = the evidence that learning happened
    plt.figure(figsize=(8, 4.5))
    for s, color in zip(TRAIN_STYLES, ("tab:red", "tab:blue")):
        plt.plot(checkpoints, history[s], color=color, marker="o", markersize=4,
                 label=f"Q-learning vs {s}")
        plt.axhline(baseline[s], color=color, linestyle="--", alpha=0.5,
                    label=f"random agent vs {s}")
    plt.xscale("symlog", linthresh=10)   # log scale (0 allowed): learning is fast
    plt.xlabel("Training episodes (log scale)")
    plt.ylabel("Win rate (greedy policy)")
    plt.xlim(0, args.episodes)
    plt.ylim(0, 1.02)
    plt.title("Adaptive Arena RL: win rate during training")
    plt.legend(loc="center right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    np.save("q_table.npy", agent.q)
    print("\nSaved results.png and q_table.npy")


if __name__ == "__main__":
    main()
