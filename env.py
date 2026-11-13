"""
env.py - a tiny 1D fighting arena (Gym-style: reset() and step()).

Two fighters stand on a line of cells. The AGENT starts on the left, a
scripted OPPONENT starts on the right. Each step both choose an action at the
same time.

Rules
-----
* Actions: 0 = left, 1 = right, 2 = attack, 3 = block.
* An attack hits if the fighters are adjacent AND the target is not blocking.
  A hit removes 1 HP.
* Recovery: after attacking, a fighter is helpless for the NEXT step (its
  action is forced to IDLE, so it cannot block). Attacking into a block
  therefore leaves you open. This is what makes "block, then counter-attack"
  the winning idea.
* An episode ends when someone reaches 0 HP or after max_steps.

State (what the agent sees) - all small whole numbers, so a table can hold it:
    (distance, my_hp, opp_hp, opp_last_action, my_recovery)
"""
import numpy as np

LEFT, RIGHT, ATTACK, BLOCK = 0, 1, 2, 3
IDLE = 4  # not choosable; used when a fighter is in recovery
ACTION_NAMES = ["left", "right", "attack", "block", "idle"]
N_ACTIONS = 4

# Styles used for training/evaluation. "idle" and "blocker" exist only so the
# unit tests have a predictable opponent.
TRAIN_STYLES = ("aggressive", "defensive")


class ArenaEnv:
    def __init__(self, size=10, max_hp=5, max_steps=60, seed=None):
        self.size = size
        self.max_hp = max_hp
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.style = "aggressive"
        # (distance, my_hp, opp_hp, opp_last_action, my_recovery)
        self.state_shape = (size, max_hp + 1, max_hp + 1, 5, 2)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, style=None):
        """Start a new episode. If style is None, pick a random training style."""
        self.style = style if style is not None else self.rng.choice(TRAIN_STYLES)
        self.agent_pos = 0
        self.opp_pos = self.size - 1
        self.agent_hp = self.max_hp
        self.opp_hp = self.max_hp
        self.agent_rec = 0   # 1 means "must idle next step"
        self.opp_rec = 0
        self.opp_last = IDLE
        self.steps = 0
        return self._obs()

    def step(self, action):
        """Apply the agent's action. Returns (state, reward, done, info)."""
        self.steps += 1

        # 1) Choose actions. A fighter in recovery is forced to idle.
        a_act = IDLE if self.agent_rec else action
        o_act = IDLE if self.opp_rec else self._opponent_action()
        self.agent_rec = 0
        self.opp_rec = 0

        # 2) Movement (agent first; fighters can never share a cell).
        if a_act in (LEFT, RIGHT):
            self.agent_pos = self._move(self.agent_pos, self.opp_pos, a_act)
        if o_act in (LEFT, RIGHT):
            self.opp_pos = self._move(self.opp_pos, self.agent_pos, o_act)

        # 3) Combat. Both attacks are resolved from the same positions.
        adjacent = abs(self.agent_pos - self.opp_pos) <= 1
        dmg_to_opp = int(a_act == ATTACK and adjacent and o_act != BLOCK)
        dmg_to_agent = int(o_act == ATTACK and adjacent and a_act != BLOCK)
        self.opp_hp -= dmg_to_opp
        self.agent_hp -= dmg_to_agent

        # 4) Attacking (hit or miss) makes you helpless next step.
        if a_act == ATTACK:
            self.agent_rec = 1
        if o_act == ATTACK:
            self.opp_rec = 1
        self.opp_last = o_act

        # 5) Reward: damage dealt - damage taken - small time penalty.
        reward = dmg_to_opp - dmg_to_agent - 0.01

        # 6) Episode end.
        done, outcome = False, None
        if self.agent_hp <= 0 or self.opp_hp <= 0 or self.steps >= self.max_steps:
            done = True
            if self.agent_hp > self.opp_hp:
                outcome, reward = "win", reward + 10
            elif self.agent_hp < self.opp_hp:
                outcome, reward = "loss", reward - 10
            else:
                outcome = "draw"
        return self._obs(), reward, done, {"outcome": outcome}

    def render(self):
        """One-line ASCII picture, e.g.  A . . . O . . . . .   HP 5 vs 3"""
        cells = ["."] * self.size
        cells[self.agent_pos] = "A"
        cells[self.opp_pos] = "O"
        return " ".join(cells) + f"   HP {self.agent_hp} vs {self.opp_hp}"

    # ------------------------------------------------------------- internals
    def _obs(self):
        dist = abs(self.agent_pos - self.opp_pos)
        return (dist, max(0, self.agent_hp), max(0, self.opp_hp),
                self.opp_last, self.agent_rec)

    def _move(self, pos, other, action):
        new = pos + (-1 if action == LEFT else 1)
        new = min(max(new, 0), self.size - 1)
        return pos if new == other else new

    def _toward_agent(self):
        return LEFT if self.opp_pos > self.agent_pos else RIGHT

    def _opponent_action(self):
        """Scripted opponent. Each style is a few if-statements."""
        dist = abs(self.agent_pos - self.opp_pos)
        r = self.rng.random()
        if self.style == "aggressive":
            # Rushes in and attacks most of the time.
            if dist > 1:
                return self._toward_agent()
            return ATTACK if r < 0.8 else BLOCK
        if self.style == "defensive":
            # Lets you come to it, blocks a lot, attacks now and then.
            if dist > 2:
                return BLOCK if r < 0.5 else self._toward_agent()
            if dist == 2:
                return BLOCK
            return BLOCK if r < 0.6 else ATTACK
        if self.style == "blocker":   # test helper: always blocks
            return BLOCK
        return IDLE                   # "idle": test helper, does nothing
