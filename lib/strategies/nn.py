import numpy as np
import scipy.special
from mesa import Agent

from agent import PDAgent
from config import *

if NUMPY_SEED is not None:
    np.random.seed(NUMPY_SEED)


class NeuralAgent(PDAgent):
    def __init__(self, unique_id, model, stochastic=False, mut_prob=MUT_PROB, mut_strength=1):
        super().__init__(unique_id, model)
        self.other_prev_actions = {}  # Map other's id to its previous action
        self.stochastic = stochastic

        self.activation_function = lambda x: scipy.special.expit(x)

        # structure of the nn
        self.input = 1
        self.hidden1 = 3
        self.hidden2 = 3
        self.output = 1

        # Probability of mutation
        self.thr = mut_prob
        # strength of mutation
        self.strength = mut_strength

        # initialize weights
        self.wih1 = np.zeros((self.hidden1, self.input))
        self.wh1h2 = np.zeros((self.hidden2, self.hidden1))
        self.wh2o = np.zeros((self.output, self.hidden2))

    def clone(self):
        return NeuralAgent(self.unique_id, self.model, self.stochastic, self.thr, self.strength)

    def random_weights(self):
        self.wih1 = np.random.normal(0.0, pow(self.input, -0.5), (self.hidden1, self.input))
        self.wh1h2 = np.random.normal(0.0, pow(self.hidden1, -0.5), (self.hidden2, self.hidden1))
        self.wh2o = np.random.normal(0.0, pow(self.hidden2, -0.5), (self.output, self.hidden2))

    def mutate(self):
        if np.random.rand() < self.thr:
            self.strength += (0.5 - np.random.rand()) * 0.1

        if self.strength < 0:
            self.strength = 0

        if np.random.rand() < self.thr:
            self.wih1 += np.random.normal(0.0, pow(self.input, -0.5) * self.strength, (self.hidden1, self.input))

        if np.random.rand() < self.thr:
            self.wh1h2 += np.random.normal(0.0, pow(self.hidden1, -0.5) * self.strength, (self.hidden2, self.hidden1))

        if np.random.rand() < self.thr:
            self.wh2o += np.random.normal(0.0, pow(self.hidden2, -0.5) * self.strength, (self.output, self.hidden2))

    def forward(self, inputs: np.ndarray):
        """
        Forward the inputs through the neural network

        :param inputs: history of other's action, should be a list of 0 or 1
        :return: a 1x1 tensor in [0, 1], indicating the action tendency
        """
        x = self.wih1 @ inputs
        x = self.activation_function(x)

        x = self.wh1h2 @ x
        x = self.activation_function(x)

        x = self.wh2o @ x
        x = self.activation_function(x)  # [0, 1]

        return x

    def make_action(self, other: Agent):
        history = self.other_prev_actions[other.unique_id]
        output = np.asscalar(self.forward(np.array([history])))

        # Possibly make the action more stochastic
        if self.stochastic:
            return 0 if self.random.random() >= output else 1
        else:
            return 0 if output <= 0.5 else 1

    def feedback(self):
        # Update history
        self.other_prev_actions = {other.unique_id: other.action[self.unique_id] for other in self.neighbors}

    def data(self):
        """
        Iterate through all weight matrices
        """
        yield self.wih1
        yield self.wh1h2
        yield self.wh2o
