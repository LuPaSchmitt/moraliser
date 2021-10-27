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


rounds = 10
gens = 1000
lin = np.arange(1, gens + 1, 1)
Info_avg = []
Info_max = []

for i in range(gens):
    for a in agents:
        a.score = 0

    for n in range(rounds):

        for j, me in enumerate(agents):
            opponent = agents[(j + 1) % N]
            my_action = me.action(opponent.id)
            their_action = opponent.action(me.id)
            me.opponent_prev_action_map[opponent.id] = their_action
            opponent.opponent_prev_action_map[me.id] = my_action

            me.score += SCORE_MAP[my_action][their_action]
            opponent.score += SCORE_MAP[their_action][my_action]

    # after the run
    score = [a.score for a in agents]

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
plt.savefig('ring.png')
