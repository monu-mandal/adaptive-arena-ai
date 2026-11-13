r"""
agent.py - tabular Q-learning.

The Q-table stores, for every (state, action), our current estimate of
"how much total reward will I get from here if I take this action and then
play well". Learning = repeatedly nudging each entry toward what we just saw.

The whole algorithm is the update in QAgent.update():

    Q(s,a) <- Q(s,a) + alpha * ( r + gamma * max_a' Q(s',a')  -  Q(s,a) )
                                  \________ target ________/   \ current /
                                  \______________ TD error ______________/
"""
import numpy as np


class QAgent:
    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9997, seed=0):
        self.q = np.zeros(state_shape + (n_actions,))
        self.n_actions = n_actions
        self.alpha = alpha          # learning rate: how far to move per update
        self.gamma = gamma          # discount: how much future reward matters
        self.epsilon = eps_start    # exploration rate (chance of a random move)
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        """Epsilon-greedy: usually the best known action, sometimes a random one."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[state]
        best = np.flatnonzero(values == values.max())   # break ties randomly
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done):
        """One Q-learning step (see the formula at the top of the file)."""
        target = reward if done else reward + self.gamma * self.q[next_state].max()
        td_error = target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error

    def end_episode(self):
        """Explore less as training goes on."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


class RandomAgent:
    """Baseline: picks any action uniformly. The learner must beat this."""
    def __init__(self, n_actions, seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        return int(self.rng.integers(self.n_actions))
