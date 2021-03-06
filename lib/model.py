from typing import Callable

import dill as pickle
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from tqdm import tqdm

from lib.config import *
from lib.genetic import *
from lib.strategies import *


class PDModel(Model):
    """Model class for Iterated Prisoner's Dilemma."""

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
            self.raw_fitness = lambda a: a.score
        elif fitness_type == 'cooperating_ratio':
            self.raw_fitness = lambda a: a.cooperating_ratio
        elif fitness_type == 'defecting_ratio':
            self.raw_fitness = lambda a: a.defecting_ratio
        else:
            raise ValueError(f'Unknown fitness type {fitness_type}')

        # Create agents
        for x in range(width):
            for y in range(height):
                params_dict = {}
                if agent_type_map is not None:
                    packed = agent_type_map(x, y)
                    if isinstance(packed, str):
                        type_str = packed
                    else:
                        type_str, params_dict = packed
                else:
                    if agent_type == 'mixed':
                        # TODO: you may use other agent type and probability configurations
                        type_str = self.random.choices(['neural', 'string', 'tit_for_tat', 'simple'], [1, 1, 1, 1], k=1)[0]
                        # type_str = self.random.choices(['string', 'string_tit_for_tat', 'good'], [2, 3, 2], k=1)[0]
                    else:
                        type_str = agent_type

                agent = self.create_agent(type_str, params_dict)
                agent.inherited_attr = '#' + ''.join(self.random.choices('ABCDEF0123456789', k=6))
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.agents: List[PDAgent] = self.schedule.agents
        for agent in self.agents:
            agent.initialize(self.neighbor_type, 0)

        self.initial_agents_types = [type(agent) for agent in self.agents]

        # Statistics
        self.num_cooperating_agents = 0
        self.mean_score = 0
        self.max_score = 0
        self.min_score = 0
        self.num_simples = 0
        self.num_tit_for_tats = 0
        self.num_neurals = 0
        self.num_strings = 0
        self.mean_feature_vector = None
        self.f_max = 0
        self.f_avg = 0

        # Collect for each step
        self.data_collector = DataCollector(
            model_reporters={
                'Cooperating_Agents': 'num_cooperating_agents',
                'Mean_Score': 'mean_score',
                'Max_Score': 'max_score',
                'Min_Score': 'min_score',
                'Simple_Agents': 'num_simples',
                'Tit_for_tat_Agents': 'num_tit_for_tats',
                'Neural_Agents': 'num_neurals',
                'String_Agents': 'num_strings',
                'Mean_Feature_Vector': 'mean_feature_vector',
            },
            agent_reporters={
                'Scores': 'score',
            }
        )

    def create_agent(self, type_str, params_dict):
        if type_str == 'neural':
            agent = NeuralAgent(self.next_id(), self, stochastic=params_dict.get('stochastic', False))
            agent.random_weights()
        elif type_str == 'string':
            agent = StringAgent(self.next_id(), self)
            agent.random_chromosome()
            if 'starting_action' in params_dict:
                agent.chromosome[0] = params_dict['starting_action']
        elif type_str == 'string_tit_for_tat':
            agent = StringAgent.create_tit_for_tat(self.next_id(), self, starting_action=params_dict.get('starting_action', None))
        elif type_str == 'string_always_cooperating':
            agent = StringAgent.create_always_cooperating(self.next_id(), self)
        elif type_str == 'string_always_defecting':
            agent = StringAgent.create_always_defecting(self.next_id(), self)
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
        self.mean_score = sum(a.score for a in self.agents) / len(self.agents)

        scores = sorted(a.score for a in self.agents)
        k = int(0.1 * len(self.agents))
        self.max_score = sum(scores[-k:]) / k
        self.min_score = sum(scores[:k]) / k

        self.num_simples = 0
        self.num_tit_for_tats = 0
        self.num_neurals = 0
        self.num_strings = 0
        for a in self.agents:
            if isinstance(a, SimpleAgent):
                self.num_simples += 1
            elif isinstance(a, TitForTatAgent):
                self.num_tit_for_tats += 1
            elif isinstance(a, NeuralAgent):
                self.num_neurals += 1
            elif isinstance(a, StringAgent):
                self.num_strings += 1

        fs = np.array([a.feature_vector() for a in self.agents if isinstance(a, NeuralAgent)])
        self.mean_feature_vector = fs.mean(0) if len(fs) > 0 else None

        self.f_max = max(a.fitness for a in self.agents)
        self.f_avg = sum(a.fitness for a in self.agents) / len(self.agents)

    def substep(self):
        """
        In each substep, every agent plays one round of IPD with its neighbors
        """
        if self.schedule.steps >= 1:
            # Play the tournament and advance the actions
            self.schedule.step()
        else:
            # In the first tournament agents have their action set already up in the initialize()
            self.schedule.steps += 1
            self.schedule.time += 1

        for agent in self.agents:
            agent.update_scores()
            agent.feedback()

    def recompute_scaled_fitness(self) -> Callable:
        """
        Fitness scaling is used to avoid the situations where the most fitted agent has overwhelming advantage over average agents,
            or where the agents have too evenly distributed fitness.
            In such situations, genetic algorithm cannot effectively find the global maximal or is easily stuck in local maxima.
        Return the scaled fitness function, which maps the max fitness to FITNESS_MULTIPLIER times avg fitness
        see java manual P15 for more details
        """
        f_max = max(self.raw_fitness(a) for a in self.agents)
        f_avg = sum(self.raw_fitness(a) for a in self.agents) / len(self.agents)
        a = (FITNESS_MULTIPLIER - 1) * f_avg / (f_max - f_avg + EPS)
        b = f_avg * (f_max - FITNESS_MULTIPLIER * f_avg) / (f_max - f_avg + EPS)
        return lambda agent: a * self.raw_fitness(agent) + b

    def update_fitness(self):
        """
        Called before each evolution. Update the fitness field of all agents
        """
        f = self.recompute_scaled_fitness()
        for a in self.agents:
            a.fitness = f(a)

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
            agent.initialize(self.neighbor_type)

        self.generations += 1

    def next_generation_local(self):
        """
        Apply local genetic algorithm to select dominant agents
        """
        children = evolute_local(self.agents)

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
            agent.initialize(self.neighbor_type)

        self.generations += 1

    def step(self):
        """
        Each generation consists of several substeps. Genetic algorithm is applied at the end of the step.
        """
        if self.generations >= 1:
            if USE_LOCAL_GA:
                self.next_generation_local()
            else:
                self.next_generation()
        else:
            self.generations += 1

        for i in range(self.num_substeps):
            self.substep()

        self.update_fitness()
        self.update_stats()
        self.data_collector.collect(self)

    def run(self, num_generations, callback=None):
        """Run the model for many generations"""
        for i in tqdm(range(num_generations)):
            self.step()
            if callback is not None:
                callback(i)

    def dump(self, path):
        """
        Save the model and its data_collector to path
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def load_model(path) -> PDModel:
    with open(path, 'rb') as f:
        return pickle.load(f)
