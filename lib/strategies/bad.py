from mesa import Agent

from lib.agent import PDAgent


class BadAgent(PDAgent):
    def reproducable(self) -> bool:
        return False

    def make_action(self, other: Agent):
        return 1

    def clone(self):
        return BadAgent(self.unique_id, self.model)
