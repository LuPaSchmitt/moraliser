from mesa import Agent

from lib.agent import PDAgent


class GoodAgent(PDAgent):
    def reproducable(self) -> bool:
        return False

    def make_action(self, other: Agent):
        return 0

    def clone(self):
        return GoodAgent(self.unique_id, self.model)
