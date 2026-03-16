"""Abstract base for RL agents."""

import numpy as np
from abc import ABC, abstractmethod

from .state_encoding import TOTAL_STATES, state_to_index
from ..env.constants import NUM_ACTIONS


class BaseAgent(ABC):
    def __init__(self, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=500000, q_init=-5.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_init = q_init
        self.Q = np.full(
            (TOTAL_STATES, NUM_ACTIONS), q_init, dtype=np.float64)
        self.steps = 0
        self.rng = np.random.RandomState(0)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.epsilon_decay_steps)
        return (self.epsilon_start
                + (self.epsilon_end - self.epsilon_start) * frac)

    def select_action(self, state, valid_actions, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return valid_actions[
                self.rng.randint(len(valid_actions))]
        s = state_to_index(state)
        best_q = max(self.Q[s, a] for a in valid_actions)
        best = [a for a in valid_actions
                if abs(self.Q[s, a] - best_q) < 1e-10]
        return best[self.rng.randint(len(best))]

    @abstractmethod
    def update(self, state, action, reward, next_state,
               next_valid, next_action=None):
        ...

    def set_rng(self, rng):
        self.rng = rng

    def copy_q_table(self):
        return self.Q.copy()

    def load_q_table(self, qt):
        self.Q = qt.copy()

    def get_hyperparameters(self):
        return {
            'agent_type': self.name,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'q_init': self.q_init,
            'steps_trained': self.steps,
        }