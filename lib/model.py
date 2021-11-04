from typing import List

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation

from agent import PDAgent
from genetic import evolute
from config import *
from strategies import *


class PDModel(Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, num_substeps=NUM_SUBSTEPS, seed=None):
        """
        Create a new Spatial Prisoners' Dilemma Model.
        """
        super().__init__()
        if seed is not None:
            self.random.seed(seed)
        self.grid = SingleGrid(width, height, torus=TORUS_GRID)
        self.num_substeps = num_substeps  # substeps within each generation
        self.schedule = SimultaneousActivation(self)
        self.generations = 0

        # Create agents
        for x in range(width):
            for y in range(height):
                # agent = SimpleAgent(self.next_id(), self)
                # agent = TitForTatAgent(self.next_id(), self)
                agent = NeuralAgent(self.next_id(), self, stochastic=True)
                agent.random_weights()
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize()

        self.total_scores = 0
        self.num_cooperating_agents = 0
        self.max_score = 0
        self.min_score = 0

        # Collect for each step
        self.data_collector = DataCollector(
            model_reporters={
                'Cooperating_Agents': 'num_cooperating_agents',
                'Total_Scores': 'total_scores',
                'Mean_Score': lambda m: m.total_scores / len(m.agents),
                'Max_Score': 'max_score',
                'Min_Score': 'min_score',
            },
            agent_reporters={
                'Scores': 'score',
            }
        )

        # # Collect for each substep
        # self.substep_data_collector = DataCollector(
        #     agent_reporters={
        #         'Current_Action': 'action',
        #         'Current_Score': 'cur_score',
        #         'Score': 'score',
        #     }
        # )

        self.data_collector.collect(self)

    def update_stats(self):
        self.num_cooperating_agents = sum(a.is_cooperating for a in self.agents)
        self.total_scores = sum(a.score for a in self.agents)
        self.max_score = max(a.score for a in self.agents)
        self.min_score = min(a.score for a in self.agents)

    def substep(self):
        """
        In each substep, every agent plays the PD tournament with its neighbors once
        """
        if self.schedule.steps >= 1:  # scheduler.steps mean the substeps
            # Play the tournament and advance the actions
            self.schedule.step()
        else:
            # In the first tournament agents have their action set already up in the initialize()
            self.schedule.steps += 1
            self.schedule.time += 1

        for agent in self.agents:
            agent.update_scores()
            agent.feedback()

        # self.substep_data_collector.collect(self)

    def next_generation(self):
        """
        Apply genetic algorithm to select dominant agents
        """
        children = evolute(self.agents)

        # Recreate agents
        width, height = self.grid.width, self.grid.height
        self.grid = SingleGrid(width, height, torus=TORUS_GRID)
        self.schedule = SimultaneousActivation(self)
        self.current_id = 0
        for x in range(width):
            for y in range(height):
                agent = children.pop()
                agent.unique_id = self.next_id()
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize()

        self.generations += 1

    def step(self):
        """
        Each generation consists of several substeps. Genetic algorithm is applied at the end of the step.
        """
        if self.schedule.steps >= self.num_substeps:
            self.next_generation()

        self.substep()
        self.update_stats()
        self.data_collector.collect(self)

    def run(self, n):
        """Run the model for n steps (generations)"""
        for _ in range(n):
            self.step()
