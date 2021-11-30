from lib.model import *
from matplotlib import pyplot as plt
import numpy as np


def agent_type_map(x, y):
    if x in [0,4]:
        return 'tit_for_tat'
    return 'neural'


def plot_feature_map(i):
    feature_map0 = np.zeros((m.grid.width, m.grid.height))
    feature_map1 = np.zeros((m.grid.width, m.grid.height))
    for cell in m.grid.coord_iter():
        agent, x, y = cell
        if isinstance(agent, NeuralAgent):
            feature_map0[x][y] = agent.feature_vector()[0]
            feature_map1[x][y] = agent.feature_vector()[1]
    plt.clf()
    plt.imshow(feature_map0, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'figs/0feature_map_{i}.png')
    plt.clf()
    plt.imshow(feature_map1, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'figs/1feature_map_{i}.png')


m = PDModel(5,5, seed=MESA_SEED, agent_type='neural', agent_type_map=None)
m.run(1000) 
    
#plot_feature_map(10)

#
data = m.data_collector.get_model_vars_dataframe()
#
data['Mean_Score'].plot(legend=True)
data['Max_Score'].plot(legend=True)
data['Min_Score'].plot(legend=True)
plt.xlabel("Generations", fontsize = 16)
plt.ylabel("Score", fontsize = 16)

plt.savefig('2scores.pdf')

plt.clf()
vecs = data['Mean_Feature_Vector']
vec_0 = [v[0] for v in vecs]
vec_1 = [v[1] for v in vecs]
plt.plot(vec_0, label='0')
plt.plot(vec_1, label='1')
plt.xlabel("Generations", fontsize = 16)
plt.ylabel("Feature-Vector", fontsize = 16)
plt.legend()
plt.savefig('2feat_vec.pdf')
#
# plt.clf()
# data['Tit_for_tat_Agents'].plot(legend=True)
# data['Neural_Agents'].plot(legend=True)
# plt.savefig('agents.png')



