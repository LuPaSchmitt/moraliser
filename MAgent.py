# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:55:03 2021

@author: Lukas
"""

import numpy as np
import scipy.special


class Agent:

    id_counter = 0
    def __init__(self):
        
        self.id = Agent.id_counter
        Agent.id_counter += 1
        # maps between (0,1)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.opponent_prev_action_map = {}  # id -> last action

        # structure of the nn
        self.input = 1
        self.hidden1 = 3
        self.hidden2 = 3
        self.output = 1
        # last interaction
        self.li = 0

        # Score
        self.score = 0

        # Probability of mutation 
        self.thr = 0.3

        # initialize random weights
        self.wih1 = np.random.normal(0.0, pow(self.input, -0.5), (self.hidden1, self.input))
        self.wh1h2 = np.random.normal(0.0, pow(self.hidden1, -0.5), (self.hidden2, self.hidden1))
        self.wh2o = np.random.normal(0.0, pow(self.hidden2, -0.5), (self.output, self.hidden2))

    def update(self, ID, gen1, gen2, gen3):
        self.id = ID
        self.wih1 = gen1
        self.wh1h2 = gen2
        self.wh2o = gen3

    # just a prototype - probably has to be adjusted (not just the numbers and threshold)
    def mutate(self):

        
        # strength of the mutation
        factor = 1

        if np.random.rand() > self.thr:
            self.wih1 += np.random.normal(0.0, pow(self.input, -0.5) * factor, (self.hidden1, self.input))

        if np.random.rand() > self.thr:
            self.wh1h2 += np.random.normal(0.0, pow(self.hidden1, -0.5) * factor, (self.hidden2, self.hidden1))

        if np.random.rand() > self.thr:
            self.wh2o += np.random.normal(0.0, pow(self.hidden2, -0.5) * factor, (self.output, self.hidden2))

    # calculate output given input
    # 0 cooperation, 1 defecting
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

    def action(self, id):
        opponent_prev_action = self.opponent_prev_action_map.get(id, 0)
        res = self.calc(opponent_prev_action)
        # introduce randomness
        #return 0 if np.random.random() > np.max(res) else 1
        
        #without randomness
        return 0 if np.max(res) <= 0.5 else 1
