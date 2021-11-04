from typing import List

from agent import PDAgent
from strategies import *


def cross(a: PDAgent, b: PDAgent):
    rng = a.random.random
    mother = a if rng() < 0.5 else b
    c = mother.clone()
    # c.action = mother.action  # inherits memory
    # c.pos = mother.pos  # localize

    if type(a) != type(b):  # e.g. a is tit-for-tat and b is neural
        return c

    # a b are the same type
    if isinstance(a, NeuralAgent):
        # Two neural networks
        b: NeuralAgent
        c: NeuralAgent
        mother: NeuralAgent
        for wc, wa, wb in zip(c.data(), a.data(), b.data()):
            wc += wa if rng() < 0.5 else wb
        # for wc, wm in zip(c.data(), mother.data()):
        #     wc += wm

    return c


def evolute(population: List[PDAgent], fitness_function) -> List[PDAgent]:
    assert len(population) >= 2
    random = population[0].random

    weights = [fitness_function(a) for a in population]

    children = []
    for i in range(len(population)):
        a, b = random.choices(population, weights, k=2)
        c = cross(a, b)
        c.mutate()
        children.append(c)

    return children
