"""On-policy SARSA agent (policy-iteration style)."""

from .base_agent import BaseAgent
from .state_encoding import state_to_index


class SARSAAgent(BaseAgent):
    @property
    def name(self):
        return "SARSA"

    def update(self, state, action, reward, next_state,
               next_valid, next_action=None):
        s = state_to_index(state)
        ns = state_to_index(next_state)
        next_q = self.Q[ns, next_action]
        target = reward + self.gamma * next_q
        self.Q[s, action] += self.alpha * (
            target - self.Q[s, action])
        self.steps += 1