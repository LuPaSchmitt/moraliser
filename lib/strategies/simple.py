from mesa import Agent

from lib.agent import PDAgent


class SimpleAgent(PDAgent):

    def make_action(self, other: Agent):
        return self.action[other.unique_id]  # Simply copy my previous action
