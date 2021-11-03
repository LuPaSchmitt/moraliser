from mesa import Agent

from lib.agent import PDAgent


class TitForTatAgent(PDAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.other_prev_actions = {}  # Map other's id to its previous action

    def initialize(self, starting_action=None):
        super().initialize(starting_action)
        self.other_prev_actions = {other.unique_id: 0 for other in self.neighbors}

    def make_action(self, other: Agent):
        return self.other_prev_actions[other.unique_id]

    def feedback(self):
        super().feedback()
        # Update history
        self.other_prev_actions = {other.unique_id: other.action[self.unique_id] for other in self.neighbors}