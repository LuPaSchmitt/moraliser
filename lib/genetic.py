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
        candidates = [a for a in father.neighbors if a.reproducable()]
        if len(candidates) == 0:
            continue  # does nothing in this iteration
        mother = max(candidates, key=lambda a: a.fitness)  # find fittest reproducable neighbor
        if father.reproducable() and mother.reproducable() and type(father) == type(mother):
            children = father.cross(mother)
        else:
            continue

        # Place back and replace the weakest
        for child in children:
            # candidates = [a for a in father.neighbors if a.reproducable()]
            # if len(candidates) == 0:
            #     continue
            # weakest = min(candidates, key=lambda a: a.fitness)
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
