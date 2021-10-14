# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:55:03 2021

@author: Lukas
"""

import numpy as np

names = ["Otto","Rüdiger","Hans","Gustav","Alfred","Norbert","Peter","Thomas","Egon","Heinrich"]

class Agent:
    
    def __init__(self):
        
        # get a random name (needs to be improved)
        self.name = np.random.choice(names)
        
        # structure of the nn
        self.input = 1
        self.hidden1 = 3
        self.hidden2 = 3
        self.output = 1

        # Score
        self.score = 0
        
        # Number of Mutation
        self.mut = 0

        # initialize random weights
        self.wih1 = np.random.normal(0.0, pow(self.input, -0.5), (self.hidden1, self.input))
        self.wh1h2 = np.random.normal(0.0, pow(self.hidden1, -0.5), (self.hidden2, self.hidden1))
        self.wh2o = np.random.normal(0.0, pow(self.hidden2, -0.5), (self.output, self.hidden2))
        
        pass
    
    def update(self,name,gen1,gen2,gen3):
        
        self.name = name
        self.wih1 = gen1
        self.wh1h2 = gen2
        self.wh2o = gen3

    # just a prototype - probably has to be adjusted (not just the numbers and threshold)
    def mutate(self):
        
        self.mut += 1
        # strength of the mutation
        factor = 0.05
        # threshold of mutation
        thr = 0.5
        
        if( np.random.rand() > thr):
            self.wih1 += np.random.normal(0.0, pow(self.input, -0.5)*factor, (self.hidden1, self.input))
            
        if( np.random.rand() > thr):   
            self.wh1h2 += np.random.normal(0.0, pow(self.hidden1, -0.5)*factor, (self.hidden2, self.hidden1))
            
        if( np.random.rand() > thr):
            self.wh2o += np.random.normal(0.0, pow(self.hidden2, -0.5)*factor, (self.output, self.hidden2))
            

    # calculate output given input
    def calc(self, inp_list):
        
        inp = np.array(inp_list, ndmin=2).T
        hidden1_inputs = np.dot(self.wih1, inp)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    
    def reward(self,z):
        self.score += z
        
    def reset_score(self):
        self.score = 0

