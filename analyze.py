"""
analyze.py - WHY does the agent win? Two checks (run train.py first).

1. Compare simple hand-written strategies against the opponents.
2. Look inside the learned Q-table and see what it prefers in key situations.
"""
import numpy as np

from env import ATTACK, BLOCK, IDLE, RIGHT, ArenaEnv, TRAIN_STYLES
from train import play_episode


class BlockThenCounter:
    """Walk in, block, and attack only when the opponent has just attacked (so it is helpless)."""
    def act(self, state, greedy=True):
        dist, _, _, opp_last, _ = state
        if dist > 1:
            return RIGHT
        return ATTACK if opp_last == ATTACK else BLOCK


class AlwaysAttack:
    def act(self, state, greedy=True):
        return RIGHT if state[0] > 1 else ATTACK


def main():
    env = ArenaEnv(seed=5)
    n = 2000
    print("1) Hand-written strategies (2000 episodes each)")
    for name, agent in (("block then counter", BlockThenCounter()), ("always attack", AlwaysAttack())):
        for style in TRAIN_STYLES:
            res = [play_episode(env, agent, style) for _ in range(n)]
            print(f"   {name:<18} vs {style:<10} win {res.count('win')/n:4.0%}  "
                  f"draw {res.count('draw')/n:4.0%}  loss {res.count('loss')/n:4.0%}")

    q = np.load("q_table.npy")
    visited = np.abs(q).sum(axis=-1) > 0     # states the agent actually experienced
    counter = [(a == ATTACK) for idx, a in _greedy(q, visited, just_attacked=True)]
    wait = [(a == BLOCK) for idx, a in _greedy(q, visited, just_attacked=False)]
    print("\n2) Learned Q-table, fighters adjacent, both alive, visited states only")
    print(f"   opponent just attacked (it is helpless): greedy move is ATTACK in "
          f"{np.mean(counter):.0%} of {len(counter)} states")
    print(f"   opponent did not just attack:            greedy move is BLOCK  in "
          f"{np.mean(wait):.0%} of {len(wait)} states")


def _greedy(q, visited, just_attacked):
    for idx in zip(*np.nonzero(visited)):
        dist, my_hp, opp_hp, opp_last, my_rec = idx
        if dist <= 1 and my_hp > 0 and opp_hp > 0 and my_rec == 0:
            if (opp_last == ATTACK) == just_attacked and (just_attacked or opp_last in (BLOCK, IDLE)):
                yield idx, int(q[idx].argmax())


if __name__ == "__main__":
    main()
