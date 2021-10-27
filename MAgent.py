# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:55:03 2021

@author: Lukas
"""

import numpy as np
import scipy.special

names = ["Otto", "Rüdiger", "Hans", "Gustav", "Alfred", "Norbert", "Peter", "Thomas", "Egon", "Heinrich"]


class Agent:

    def __init__(self):

        # get a random name (needs to be improved)
        self.name = np.random.choice(names)
        # maps between (0,1)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.prev_action = 0
        self.opponent_prev_action = 0

        # structure of the nn
        self.input = 1
        self.hidden1 = 3
        self.hidden2 = 3
        self.output = 1
        # last interaction
        self.li = 0

        # Score
        self.score = 0

        # Number of Mutation
        self.mut = 0

        # initialize random weights
        self.wih1 = np.random.normal(0.0, pow(self.input, -0.5), (self.hidden1, self.input))
        self.wh1h2 = np.random.normal(0.0, pow(self.hidden1, -0.5), (self.hidden2, self.hidden1))
        self.wh2o = np.random.normal(0.0, pow(self.hidden2, -0.5), (self.output, self.hidden2))

    def update(self, name, gen1, gen2, gen3):

        self.name = name
        self.wih1 = gen1
        self.wh1h2 = gen2
        self.wh2o = gen3

    # just a prototype - probably has to be adjusted (not just the numbers and threshold)
    def mutate(self):

        self.mut += 1
        # strength of the mutation
        factor = 0.5
        # threshold of mutation
        thr = 0.3

        if (np.random.rand() > thr):
            self.wih1 += np.random.normal(0.0, pow(self.input, -0.5) * factor, (self.hidden1, self.input))

        if (np.random.rand() > thr):
            self.wh1h2 += np.random.normal(0.0, pow(self.hidden1, -0.5) * factor, (self.hidden2, self.hidden1))

        if (np.random.rand() > thr):
            self.wh2o += np.random.normal(0.0, pow(self.hidden2, -0.5) * factor, (self.output, self.hidden2))

    # calculate output given input
    # 0 means cooperation, 1 betraying
    def calc(self, inp_list):

        inp = np.array(inp_list, ndmin=2).T
        hidden1_inputs = np.dot(self.wih1, inp)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def get_data(self):
        return (self.name, self.wih1, self.wh1h2, self.wh2o)

    def action(self):
        res = self.calc(self.opponent_prev_action)
        # introduce randomness
        return 0 if np.random.random() < np.max(res) else 1
