import numpy as np
import scipy.special
from mesa import Agent

from lib.agent import PDAgent
from lib.config import *


class NeuralAgent(PDAgent):
    def __init__(self, unique_id, model, stochastic=False, mut_prob=MUT_PROB, mut_strength=MUT_STRENGTH):
        super().__init__(unique_id, model)
        self.other_prev_actions = {}  # Map other's id to its previous action
        self.stochastic = stochastic
        self.a = np.array([10.0])
        self.b = np.array([-0.5 * self.a])
        self.activation = scipy.special.expit  # sigmoid
        self.thr = mut_prob
        self.strength = mut_strength

    def reproducable(self) -> bool:
        return True

    def clone(self):
        return NeuralAgent(self.unique_id, self.model, self.stochastic, self.thr, self.strength)

    def random_weights(self):
        self.a = np.random.normal(self.a[0], 0.1, size=1)
        self.b = np.random.normal(self.b[0], 0.1, size=1)

    def mutate(self):
        if np.random.rand() < self.thr:
            self.a += np.random.normal(0, 0.1 * self.strength)
        if np.random.rand() < self.thr:
            self.b += np.random.normal(0, 0.1 * self.strength)

    def forward(self, inputs: np.ndarray):
        """
        Forward the inputs through the neural network

        :param inputs: history of other's action, should be a list of 0 or 1
        :return: a 1x1 tensor in [0, 1], indicating the action tendency
        """
        x = self.a * inputs + self.b
        x = self.activation(x)  # [0, 1]

        return x

    def make_action(self, other: Agent):
        history = self.other_prev_actions[other.unique_id]
        output = np.asscalar(self.forward(np.array([history])))

        # Possibly make the action more stochastic
        if self.stochastic:
            return 0 if np.random.random() >= output else 1
        else:
            return 0 if output <= 0.5 else 1

    def feedback(self):
        # Update history
        self.other_prev_actions = {other.unique_id: other.action[self.unique_id] for other in self.neighbors}

    def feature_vector(self):
        """
        Given 0 or 1 as input, return the two outputs of the network
        """
        inputs = np.array([0, 1])
        return np.array([self.forward(x).max() for x in inputs])

    def data(self):
        """
        Iterate through all weight matrices
        """
        yield self.a
        yield self.b

    def cross(self, other: PDAgent):
        assert type(self) == type(other), f"{type(self)} cannot cross with {type(self)} agent"
        c1, c2 = self.clone(), other.clone()
        c1.inherited_attr = self.inherited_attr
        c2.inherited_attr = other.inherited_attr
        for w1, w2, ws, wo in zip(c1.data(), c2.data(), self.data(), other.data()):  # copy weights
            if np.random.random() < 0.5:
                w1 += ws
                w2 += wo
            else:
                w1 += wo
                w2 += ws
        return c1, c2
