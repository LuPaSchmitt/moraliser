# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 23:36:15 2021

@author: Lukas
"""
from MAgent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt



N = 30
agents = []

#create agents
for i in range(N):
    agents.append(Agent())
    
def strat():
    last = 0
    current = 0
    return (last,current)


rounds = 10
gens = 1000
lin = np.arange(1,gens+1,1)
Info_avg = []
Info_max = []


for i in range(gens):
    for a in agents:
        a.score = 0
        
    for n in range(rounds):
      
        for a in agents:
            #strategie of partner
            last,g1= strat()
            #strat2(a.li):
            
            #calculate own strategie
            res = a.calc(last)
            #introduce randomness
            if(random.random() < np.max(res)):
                g2 = 1
            else:
                g2 = 0
            #give points
            if(g1+g2==2):
                a.score+=1
            if(g1+g2 == 0):
                a.score+=3
            if(g2>g1):
                a.score+=5
    #after the run
    score = []
    for a in agents:
        score.append(a.score)
        
    avg = sum(score)/len(score)
    mscore = max(score)
    #print("Maximum Score: "+ str(mscore)+" Average Score "+ str(np.round(avg,2)))
    Info_avg.append(avg)
    Info_max.append(mscore)
    
    
    #for simplicity
    for a in agents: 
        if(a.score<avg):
            a.mutate()


plt.plot(lin,Info_avg)
plt.plot(lin,Info_max)
        
            
            
        
        
        
        

