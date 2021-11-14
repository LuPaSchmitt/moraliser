from lib.model import *
from matplotlib import pyplot as plt
import numpy as np


def agent_type_map(x, y):
    if x in [0, 2, 4, 6, 8]:
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
    plt.imshow(feature_map0, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{i}_feature_map0.png')
    plt.clf()
    plt.imshow(feature_map1, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{i}_feature_map1.png')


m = PDModel(10, 10, seed=MESA_SEED, agent_type='mixed', agent_type_map=agent_type_map)
m.run(10)
plot_feature_map(10)
m.run(10)
plot_feature_map(20)
#
# data = m.data_collector.get_model_vars_dataframe()
#
# data['Mean_Score'].plot(legend=True)
# data['Max_Score'].plot(legend=True)
# data['Min_Score'].plot(legend=True)
# plt.savefig('scores.png')
#
# plt.clf()
# vecs = data['Mean_Feature_Vector']
# vec_0 = [v[0] for v in vecs]
# vec_1 = [v[1] for v in vecs]
# plt.plot(vec_0, label='0')
# plt.plot(vec_1, label='1')
# plt.legend()
# plt.savefig('feat_vec.png')
#
# plt.clf()
# data['Tit_for_tat_Agents'].plot(legend=True)
# data['Neural_Agents'].plot(legend=True)
# plt.savefig('agents.png')



