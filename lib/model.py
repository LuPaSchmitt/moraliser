from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from strategies import *
from agent import PDAgent
from config import *


class PDModel(Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        """
        Create a new Spatial Prisoners' Dilemma Model.
        """
        super().__init__()
        self.random.seed(0)
        self.grid = SingleGrid(width, height, torus=TORUS_GRID)
        self.schedule = SimultaneousActivation(self)

        # Create agents
        for x in range(width):
            for y in range(height):
                agent = TitForTatAgent(self.next_id(), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        agent: PDAgent
        for agent in self.schedule.agents:
            agent.initialize()

        self.data_collector = DataCollector(
            model_reporters={
                'Cooperating_Agents': lambda m: sum(a.is_cooperating for a in m.schedule.agents),
                'Total_Scores': lambda m: sum(a.score for a in m.schedule.agents),
            },
            agent_reporters={
                'Current_Actions': 'action',
                'Current_Scores': 'cur_score',
                'Scores': 'score',
            }
        )

        self.data_collector.collect(self)

    @property
    def num_cooperating_agents(self):
        return sum(a.is_cooperating for a in self.schedule.agents)

    @property
    def total_scores(self):
        return sum(a.score for a in self.schedule.agents)

    def step(self):
        print('steps:', self.schedule.steps)
        if self.schedule.steps >= 1:
            # Play the tournament and payoff
            self.schedule.step()
        else:
            # In the first tournament agents have their action set already up in the initialize()
            self.schedule.steps += 1
            self.schedule.time += 1

        agent: PDAgent
        for agent in self.schedule.agents:
            agent.update_scores()
            agent.feedback()
        # TODO Mutate

        # Collect data
        self.data_collector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()

