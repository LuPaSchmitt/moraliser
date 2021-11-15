import dill as pickle

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from tqdm import tqdm
import copy

from lib.config import *
from lib.genetic import *
from lib.strategies import *


class PDModel(Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 num_substeps=NUM_SUBSTEPS, seed=None, neighbor_type=NEIGHBOR_TYPE, fitness_type='score', agent_type='neural',
                 agent_type_map=None):
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
        self.neighbor_type = neighbor_type

        if fitness_type == 'score':
            self.fitness_function = lambda a: a.score
        elif fitness_type == 'cooperating_ratio':
            self.fitness_function = lambda a: a.cooperating_ratio
        elif fitness_type == 'defecting_ratio':
            self.fitness_function = lambda a: a.defecting_ratio
        else:
            raise ValueError(f'Unknown fitness type {fitness_type}')

        # Create agents
        for x in range(width):
            for y in range(height):
                if agent_type_map is not None:
                    type_str = agent_type_map(x, y)
                else:
                    if agent_type == 'mixed':
                        type_str = self.random.choices(['neural', 'tit_for_tat', 'simple'], [0.6, 0.3, 0.1], k=1)[0]
                    else:
                        type_str = agent_type

                agent = self.create_agent(type_str)
                agent.inherited_attr = '#' + ''.join(self.random.choices('ABCDEF0123456789', k=6))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize(self.neighbor_type, 0)

        self.initial_agents = copy.deepcopy(self.agents)

        # Statistics
        self.total_scores = 0
        self.num_cooperating_agents = 0
        self.max_score = 0
        self.min_score = 0
        self.num_simples = 0
        self.num_tit_for_tats = 0
        self.num_neurals = 0
        self.mean_feature_vector = None

        # Collect for each step
        self.data_collector = DataCollector(
            model_reporters={
                'Cooperating_Agents': 'num_cooperating_agents',
                'Total_Scores': 'total_scores',
                'Mean_Score': lambda m: m.total_scores / len(m.agents),
                'Max_Score': 'max_score',
                'Min_Score': 'min_score',
                'Simple_Agents': 'num_simples',
                'Tit_for_tat_Agents': 'num_tit_for_tats',
                'Neural_Agents': 'num_neurals',
                'Mean_Feature_Vector': 'mean_feature_vector',
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

        # self.data_collector.collect(self)

    def create_agent(self, type_str):
        if type_str == 'neural':
            agent = NeuralAgent(self.next_id(), self, stochastic=True)
            agent.random_weights()
        elif type_str == 'tit_for_tat':
            agent = TitForTatAgent(self.next_id(), self)
        elif type_str == 'simple':
            agent = SimpleAgent(self.next_id(), self)
        elif type_str == 'good':
            agent = GoodAgent(self.next_id(), self)
        elif type_str == 'bad':
            agent = BadAgent(self.next_id(), self)
        else:
            raise ValueError(f'Unknown agent type {type_str}')
        return agent

    def update_stats(self):
        self.num_cooperating_agents = sum(a.is_cooperating for a in self.agents)
        self.total_scores = sum(a.score for a in self.agents)
        self.max_score = max(a.score for a in self.agents)
        self.min_score = min(a.score for a in self.agents)
        self.num_simples = sum(1 for a in self.agents if isinstance(a, SimpleAgent))
        self.num_tit_for_tats = sum(1 for a in self.agents if isinstance(a, TitForTatAgent))
        self.num_neurals = sum(1 for a in self.agents if isinstance(a, NeuralAgent))
        fs = [a.feature_vector().squeeze() for a in self.agents if isinstance(a, NeuralAgent)]
        self.mean_feature_vector = np.mean(fs, axis=0)

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
        children = evolute(self.agents, self.fitness_function)

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
            agent.initialize(self.neighbor_type, 0)

        self.generations += 1

    def next_generation_local(self):
        """
        Apply genetic algorithm to select dominant agents
        """
        children = evolute_local(self.agents, self.fitness_function)

        # Recreate agents
        width, height = self.grid.width, self.grid.height
        self.grid = SingleGrid(width, height, torus=TORUS_GRID)
        self.schedule = SimultaneousActivation(self)
        self.current_id = 0
        for agent in children:
            agent.unique_id = self.next_id()
            self.grid.place_agent(agent, agent.pos)
            self.schedule.add(agent)

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize(self.neighbor_type, 0)

        self.generations += 1

    def step(self):
        """
        Each generation consists of several substeps. Genetic algorithm is applied at the end of the step.
        """
        # if self.schedule.steps >= self.num_substeps:
        #     self.next_generation_local()

        for i in range(self.num_substeps):
            self.substep()
        self.update_stats()
        self.data_collector.collect(self)

        self.next_generation_local()

    def run(self, n):
        """Run the model for n steps (generations)"""
        for _ in tqdm(range(n)):
            self.step()

    def dump(self, path):
        """
        Save the model and its data_collector to path
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def load_model(path) -> PDModel:
    with open(path, 'rb') as f:
        return pickle.load(f)
