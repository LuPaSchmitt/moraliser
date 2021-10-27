# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:36:15 2021

@author: Lukas
"""
from MAgent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt
import strategies
from strategies import SCORE_MAP

np.random.seed(0)
N = 30
agents = [Agent() for _ in range(N)]

# create agents
for i in range(N):
    agents.append(Agent())

rounds = 10
gens = 1000
lin = np.arange(1, gens + 1, 1)
Info_avg = []
Info_max = []

for i in range(gens):
    for a in agents:
        a.score = 0

    for n in range(rounds):

        for a in agents:
            our_action = a.action()
            their_action = strategies.default(a.opponent_prev_action, a.prev_action)
            a.opponent_prev_action = their_action
            a.prev_action = our_action
            a.score += SCORE_MAP[our_action][their_action]

    # after the run
    score = []
    for a in agents:
        score.append(a.score)

    avg = sum(score) / len(score)
    mscore = max(score)
    # print("Maximum Score: "+ str(mscore)+" Average Score "+ str(np.round(avg,2)))
    Info_avg.append(avg)
    Info_max.append(mscore)

    # for simplicity
    for a in agents:
        if (a.score < avg):
            a.mutate()

plt.plot(lin, Info_avg)
plt.plot(lin, Info_max)
plt.savefig('test.png')
