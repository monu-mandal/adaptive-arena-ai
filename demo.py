"""
demo.py - watch the trained agent fight, one ASCII frame per step.

    python demo.py                    # vs aggressive
    python demo.py --style defensive
Run train.py first: it creates q_table.npy.
"""
import argparse

import numpy as np

from agent import QAgent
from env import ACTION_NAMES, ArenaEnv, N_ACTIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="aggressive", choices=["aggressive", "defensive"])
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    env = ArenaEnv(seed=args.seed)
    agent = QAgent(env.state_shape, N_ACTIONS)
    agent.q = np.load("q_table.npy")

    state, done = env.reset(args.style), False
    print(f"Agent (A) vs {args.style} opponent (O)\n")
    print(f"step  0   {env.render()}")
    while not done:
        action = agent.act(state, greedy=True)
        state, _, done, info = env.step(action)
        print(f"step {env.steps:>2}   {env.render()}   agent: {ACTION_NAMES[action]:<6}"
              f" opp: {ACTION_NAMES[env.opp_last]}")
    print(f"\nResult: {info['outcome']}")


if __name__ == "__main__":
    main()
