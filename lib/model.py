from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from strategies import *
from agent import PDAgent
from config import *
from typing import List


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

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize()

        self.total_scores = 0
        self.num_cooperating_agents = 0
        self.max_score = 0
        self.min_score = 0

        self.data_collector = DataCollector(
            model_reporters={
                'Cooperating_Agents': 'num_cooperating_agents',
                'Total_Scores': 'total_scores',
                'Max_Score': 'max_score',
                'Min_Score': 'min_score',
            },
            agent_reporters={
                'Current_Actions': 'action',
                'Current_Scores': 'cur_score',
                'Scores': 'score',
            }
        )

        self.data_collector.collect(self)

    def update_stats(self):
        self.num_cooperating_agents = sum(a.is_cooperating for a in self.agents)
        self.total_scores = sum(a.score for a in self.agents)
        self.max_score = max(a.score for a in self.agents)
        self.min_score = min(a.score for a in self.agents)

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
        for agent in self.agents:
            agent.update_scores()
            agent.feedback()

        self.update_stats()
        # TODO Mutate

        # Collect data
        self.data_collector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()
