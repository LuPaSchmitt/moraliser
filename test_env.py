# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:36:15 2021

@author: Lukas
"""
from MAgent import Agent
import numpy as np
import matplotlib.pyplot as plt
import strategies
from strategies import SCORE_MAP

prob = True
seed = 0
np.random.seed(seed)
N = 10
agents = [Agent() for _ in range(N)]
 

rounds = 10
gens = 1000

Info_avg = []
Info_max = []
Info_min = []

sum_score = 0
gen = 0



def GA(ag):
    # Genetic Algorithm
    children = [Agent() for _ in range(N)]
    
    for i in range(N):
       
        S = 0
        pick = np.random.random()*sum_score
        
        for a in agents:
            S += a.score
            
            if S > pick :
                data = a.get_data()
                #create child
                children[i].update(i, data[0], data[1], data[2], data[3])

                children[i].mutate()

                break
            
    return children.copy()

def GA2(ag):
    # Genetic Algorithm
    children = [Agent() for _ in range(N)]
    
    for i in range(N):
        S = 0
        pick = np.random.random()*sum_score
        
        for a in ag:
            S += a.score
            
            if S > pick :
                #parent a
                data_a = a.get_data()
                break
        S = 0
        pick = np.random.random()*sum_score
        
        for a in agents:
            S += a.score
            
            if S > pick :
                #parent b
                data_b = a.get_data()
                #create child
                children[i].update(i, data_a[0] if np.random.random() > 0.5 else data_b[0], 
                                   data_a[1] if np.random.random() > 0.5 else data_b[1], 
                                   data_a[2] if np.random.random() > 0.5 else data_b[2], 
                                   data_a[3] if np.random.random() > 0.5 else data_b[3])

                children[i].mutate()

                break
            
    return children.copy()



#for gen in range(gens):
while sum_score / N <= 2*rounds*3*0.98:
    gen+=1
    for a in agents:
        a.score = 0

    for n in range(rounds):

        for j, me in enumerate(agents):
            opponent = agents[(j + 1) % N]
            my_action = me.action(opponent.id,prob)
            their_action = opponent.action(me.id,prob)
            me.opponent_prev_action_map[opponent.id] = their_action
            opponent.opponent_prev_action_map[me.id] = my_action

            me.score += SCORE_MAP[my_action][their_action]
            opponent.score += SCORE_MAP[their_action][my_action]


    
    score = [a.score for a in agents]
    sum_score = sum(score)
    avg = np.median(score)
    max_score = max(score)
    min_score = min(score)
    # print("Maximum Score: "+ str(mscore)+" Average Score "+ str(np.round(avg,2)))
    Info_avg.append(avg)
    Info_max.append(max_score)
    Info_min.append(min_score)
    
    agents = GA2(agents)
    
        
    
lin = np.arange(0, gen, 1)        
            
            
        

plt.plot(lin, Info_avg)
plt.plot(lin, Info_max)
plt.plot(lin, Info_min)
text = "pictures/ring/NoA_"+str(N)+"Gen_"+str(gens)+"Rounds_"+str(rounds)+"Seed_"+str(seed)+"Random_"+str(prob)
plt.savefig(text + '.png')
for a in agents:
    print(a.calc(0),a.calc(1))
