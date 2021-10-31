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


seed =  0
np.random.seed(seed)
N = 100
agents = [Agent() for _ in range(N)]


rounds = 10
gens = 100
lin = np.arange(1, gens + 1, 1)
Info_avg = []
Info_max = []
Info_min = []

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
    sum_score = sum(score)
    avg = np.median(score)
    max_score = max(score)
    min_score = min(score)
    # print("Maximum Score: "+ str(mscore)+" Average Score "+ str(np.round(avg,2)))
    Info_avg.append(avg)
    Info_max.append(max_score)
    Info_min.append(min_score)

    # Genetic Algorithm
    children = [Agent() for _ in range(N)]
    
    for i in range(N):
        S = 0
        pick = random.randrange(0,sum_score)
        
        for a in agents:
            S += a.score
            
            if S > pick :
                #parent a
                gen1a = a.wih1  
                gen2a = a.wh1h2
                gen3a =  a.wh2o
                break
        S = 0
        pick = random.randrange(0,sum_score)
        
        for a in agents:
            S += a.score
            
            if S > pick :
                #parent b
                gen1b = a.wih1  
                gen2b = a.wh1h2
                gen3b =  a.wh2o
                #create child
                children[i].update(i, gen1a if np.random.random() > 0.5 else gen1b , gen2a if np.random.random() > 0.5 else gen2b, gen3a if np.random.random() > 0.5 else gen3b)
                children[i].mutate()
                break
            
    #replace old generation with children
    for i in range(N):
        agents[i] = children[i]
        
            
            
        

plt.plot(lin, Info_avg)
plt.plot(lin, Info_max)
plt.plot(lin, Info_min)
text = "Ring_NoA_"+str(N)+"Gen_"+str(gens)+"Rounds_"+str(rounds)+"Seed_"+str(seed)
plt.savefig(text + '.png')
