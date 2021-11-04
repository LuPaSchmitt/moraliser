from mesa import Agent

from agent import PDAgent


class TitForTatAgent(PDAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.other_prev_actions = {}  # Map other's id to its previous action

    def clone(self):
        return TitForTatAgent(self.unique_id, self.model)

    def initialize(self, neighbor_type, starting_action=None):
        super().initialize(neighbor_type, starting_action)
        self.other_prev_actions = {other.unique_id: 0 for other in self.neighbors}

    def make_action(self, other: Agent):
        return self.other_prev_actions[other.unique_id]

    def feedback(self):
        # Update history
        self.other_prev_actions = {other.unique_id: other.action[self.unique_id] for other in self.neighbors}
