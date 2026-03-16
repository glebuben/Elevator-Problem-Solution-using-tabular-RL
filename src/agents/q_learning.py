"""Off-policy Q-learning agent (value-iteration style)."""

from .base_agent import BaseAgent
from .state_encoding import state_to_index


class QLearningAgent(BaseAgent):
    @property
    def name(self):
        return "Q-learning"

    def update(self, state, action, reward, next_state,
               next_valid, next_action=None):
        s = state_to_index(state)
        ns = state_to_index(next_state)
        max_next_q = max(self.Q[ns, a] for a in next_valid)
        target = reward + self.gamma * max_next_q
        self.Q[s, action] += self.alpha * (
            target - self.Q[s, action])
        self.steps += 1