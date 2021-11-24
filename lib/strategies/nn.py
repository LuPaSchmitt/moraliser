import numpy as np
import scipy.special
from mesa import Agent

from lib.agent import PDAgent
from lib.config import *

from typing import Dict


class NeuralAgent(PDAgent):
    def __init__(self, unique_id, model, layer_dims=[6, 3, 1], stochastic=False, mut_prob=MUT_PROB, mut_strength=MUT_STRENGTH):
        super().__init__(unique_id, model)
        self.prev_actions: Dict[int, np.ndarray] = {}  # Map other's id to its previous action arrays
        self.stochastic = stochastic

        assert layer_dims[0] % 2 == 0
        self.mem_len = layer_dims[0] // 2  # input: length of memory (each round we have two numbers: ours, opponent's)
        assert layer_dims[-1] == 1  # output one dimension: C/D
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1

        # Network structure
        self.ws = []  # weights
        self.bs = []  # biases
        for i in range(self.num_layers):
            input_dim, output_dim = layer_dims[i], layer_dims[i + 1]
            self.ws.append(np.zeros((output_dim, input_dim), dtype=float))
            self.bs.append(np.zeros(output_dim, dtype=float))

        self.relu = lambda x: x * (x > 0)
        self.activation = scipy.special.expit  # sigmoid
        self.thr = mut_prob
        self.strength = mut_strength

    def reproducable(self) -> bool:
        return True

    def clone(self):
        return NeuralAgent(self.unique_id, self.model, self.layer_dims, self.stochastic, self.thr, self.strength)

    def initialize(self, neighbor_type, starting_action=None):
        super().initialize(neighbor_type, starting_action)
        # todo: random?
        self.prev_actions = {other.unique_id: np.zeros((self.mem_len, 2), dtype=int) for other in self.neighbors}

    def random_weights(self):
        for i in range(self.num_layers):
            self.ws[i] = np.random.normal(0, 0.1, self.ws[i].shape)
            # self.bs[i] = np.random.normal(0 if i == 0 else -0.3, 0.1, self.bs[i].shape)
            self.bs[i] = np.random.normal(0, 0.1, self.bs[i].shape)

    def mutate(self):
        for i in range(self.num_layers):
            if np.random.random() < self.thr:
                self.ws[i] += np.random.normal(0, 0.1 * self.strength, self.ws[i].shape)
            if np.random.random() < self.thr:
                self.bs[i] += np.random.normal(0, 0.1 * self.strength, self.bs[i].shape)

    def forward(self, inputs: np.ndarray):
        """
        Forward the inputs through the neural network

        :param inputs: flattened history of ours and other's action, should be a list of 0 or 1
        :return: a 1x1 tensor in [0, 1], indicating the action tendency
        """
        assert inputs.shape == (self.mem_len * 2,)

        x = inputs
        for i in range(self.num_layers):
            x = self.ws[i] @ x + self.bs[i]
            # x = self.relu(x) if i < self.num_layers - 1 else self.activation(x)
            x = self.activation(x)

        return x

    def make_action(self, other: Agent):
        history = self.prev_actions[other.unique_id]
        output = np.asscalar(self.forward(history.flatten()))

        # Possibly make the action more stochastic
        if self.stochastic:
            return 0 if np.random.random() >= output else 1
        else:
            return 0 if output <= 0.5 else 1

    def feedback(self):
        # Update history
        for other in self.neighbors:
            this_round = [(self.action[other.unique_id], other.action[self.unique_id])]
            history = self.prev_actions[other.unique_id]
            self.prev_actions[other.unique_id] = np.append(history[1:, :], this_round, axis=0)

    def feature_vector(self):
        """
        Given 0 or 1 as input, return the two outputs of the network
        """
        f0 = np.asscalar(self.forward(np.array([0] * (self.mem_len * 2))))
        if self.mem_len > 2:
            f1 = np.asscalar(self.forward(np.array([0] * (self.mem_len * 2 - 4) + [1] * 4)))
        else:
            f1 = np.asscalar(self.forward(np.array([0] * (self.mem_len * 2 - 2) + [1] * 2)))
        return np.array([f0, f1])

    def data(self):
        """
        Iterate through all weight matrices
        """
        for i in range(self.num_layers):
            yield self.ws[i]
            yield self.bs[i]

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
