from mesa import Agent

from agent import PDAgent


class SimpleAgent(PDAgent):

    def make_action(self, other: Agent):
        # return self.action[other.unique_id]  # Simply copy my previous action
        return 1

    def clone(self):
        return SimpleAgent(self.unique_id, self.model)
