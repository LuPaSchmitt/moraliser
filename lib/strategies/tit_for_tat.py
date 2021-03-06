from mesa import Agent

from lib.agent import PDAgent
from lib.config import TFT_REPRODUCABLE


class TitForTatAgent(PDAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.other_prev_actions = {}  # Map other's id to its previous action

    def reproducable(self) -> bool:
        return TFT_REPRODUCABLE

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

    def cross(self, other: PDAgent):
        assert type(self) == type(other), f"{type(self)} cannot cross with {type(self)} agent"
        c1, c2 = self.clone(), other.clone()
        c1.inherited_attr = self.inherited_attr
        c2.inherited_attr = other.inherited_attr

        return c1, c2
