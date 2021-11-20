from typing import List

from lib.agent import PDAgent
from lib.strategies import *


def evolute(population: List[PDAgent], fitness_function) -> List[PDAgent]:
    assert len(population) >= 2
    random = population[0].random

    weights = [a.fitness for a in population]

    children = []
    while len(children) < len(population):
        a, b = random.choices(population, weights, k=2)
        if a.reproducable() and b.reproducable() and type(a) == type(b):
            c1, c2 = a.cross(b)
        else:
            c1, c2 = a, b
        c1.mutate()
        c2.mutate()
        children += [c1, c2]

    children = children[:len(population)]
    return children


def evolute_local(population: List[PDAgent], fitness_function) -> List[PDAgent]:
    """
    A modified version of genetic algorithm that is specially designed for spatial society.
    In here we select an agent as one parent, and select its fittest neighbor as another parent to reproduce two children.
    The children are then placed at the position of the first parent's weakest neighbors. And the replaced neighbors are out.
    """
    children = population
    raise NotImplementedError
    # for agent in population:
    #     if not agent.reproducable():
    #         children.append(agent)
    #     else:
    #         candidates = [c for c in agent.neighbors if c.reproducable()]
    #         assert len(candidates) >= 1, f'Agent {agent} at {agent.pos} has too few reproducable neighbors'
    #
    #         weights = [fitness_function(n) for n in candidates]
    #
    #         other = agent.random.choices(candidates, weights, k=1)[0]
    #         c = cross(agent, other)
    #         c.mutate()
    #         c.pos = agent.pos
    #         children.append(c)

    return children
