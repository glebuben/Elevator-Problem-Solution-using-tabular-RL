class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, state, valid_actions):
        return valid_actions[
            self.rng.randint(len(valid_actions))]