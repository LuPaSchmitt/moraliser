from typing import List

from lib.agent import PDAgent


def evolute(population: List[PDAgent]) -> List[PDAgent]:
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


# def evolute_local_old(population: List[PDAgent]) -> List[PDAgent]:
#     children = []
#     random = population[0].random
#
#     for agent in population:
#         if not agent.reproducable():
#             children.append(agent)
#         else:
#             candidates = [agent] + agent.neighbors
#             candidates = [c for c in candidates if c.reproducable()]
#             assert len(candidates) >= 2, f'Agent {agent} at {agent.pos} has too few reproducable neighbors'
#
#             weights = [a.fitness for a in candidates]
#             a, b = random.choices(candidates, weights, k=2)
#             if type(a) != type(b):
#                 c = agent
#             else:
#                 c, _ = a.cross(b)
#                 c.pos = agent.pos
#
#             c.mutate()
#             children.append(c)
#
#     return children


def evolute_local(population: List[PDAgent]) -> List[PDAgent]:
    """
    A modified version of genetic algorithm that is specially designed for spatial society.
    In here we select an agent as one parent, and select its fittest neighbor as another parent to reproduce two children.
    The children are then placed at the position of the first parent's weakest neighbors. And the replaced neighbors are out.
    """
    assert len(population) >= 2
    random = population[0].random

    weights = [a.fitness for a in population]

    for i in range(len(population) // 2):
        father = random.choices(population, weights, k=1)[0]
        if not father.reproducable():
            continue
        candidates = [a for a in father.neighbors if a.reproducable() and type(a) == type(father)]
        if len(candidates) == 0:
            continue  # does nothing in this iteration
        mother = max(candidates, key=lambda a: a.fitness)  # find fittest reproducable neighbor
        children = father.cross(mother)

        # Place back and replace the weakest
        for child in children:
            if len(father.neighbors) == 0:
                continue
            weakest = min(father.neighbors, key=lambda a: a.fitness)
            index = weakest.unique_id - 1
            child.pos = weakest.pos
            child.mutate()
            population[index] = child
            weights[index] = 0
            father.neighbors.remove(weakest)

    return population  # in place operation
