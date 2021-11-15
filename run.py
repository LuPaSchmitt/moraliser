from datetime import datetime
from os import makedirs

import numpy as np
from matplotlib import pyplot as plt

from lib.model import *

folder = f"out/{datetime.now().strftime('%d_%m_%Y_%H:%M')}"
makedirs(folder, exist_ok=False)

with open(f'{folder}/config.txt', 'w') as f:
    f.write(config_to_str())


def plot_feature_map(m, i):
    feature_map0 = np.zeros((m.grid.height, m.grid.width))
    feature_map1 = np.zeros((m.grid.height, m.grid.width))
    for cell in m.grid.coord_iter():
        agent, x, y = cell
        if isinstance(agent, NeuralAgent):
            v = agent.feature_vector()
            feature_map0[y, x] = v[0]
            feature_map1[y, x] = v[1]
    plt.clf()
    plt.imshow(feature_map0, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{folder}/{i}_feature_map0.png')
    plt.clf()
    plt.imshow(feature_map1, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{folder}/{i}_feature_map1.png')


def plot_agent_type_map(m, i):
    im = np.zeros((m.grid.height, m.grid.width, 3), dtype=int)
    cmap = {
        SimpleAgent: [0, 0, 255],
        TitForTatAgent: [235, 155, 52],  # Orange
        GoodAgent: [0, 255, 0],
        BadAgent: [255, 0, 0],
        NeuralAgent: [237, 14, 233],  # Pink
    }
    for cell in m.grid.coord_iter():
        agent, x, y = cell
        im[y, x, :] = cmap[type(agent)]
    plt.clf()
    plt.imshow(im, interpolation='nearest')
    plt.savefig(f'{folder}/{i}_agent_type.png')


def plot_scores(data):
    plt.clf()
    data['Mean_Score'].plot(legend=True)
    data['Max_Score'].plot(legend=True)
    data['Min_Score'].plot(legend=True)
    plt.savefig(f'{folder}/scores.png')


def plot_feat_vecs(data):
    plt.clf()
    vecs = data['Mean_Feature_Vector']
    vec_0 = [v[0] for v in vecs]
    vec_1 = [v[1] for v in vecs]
    plt.plot(vec_0, label='0')
    plt.plot(vec_1, label='1')
    plt.legend()
    plt.savefig(f'{folder}/feat_vec.png')


def agent_type_map(x, y):
    if x in [0, DEFAULT_WIDTH - 1]:
        return 'tit_for_tat'
    return 'neural'


m = PDModel(DEFAULT_WIDTH, DEFAULT_HEIGHT, seed=MESA_SEED, agent_type='mixed', agent_type_map=agent_type_map)
generations = 5000
m.run(generations)
m.dump(f'{folder}/model.pickle')
plot_feature_map(m, generations)
plot_agent_type_map(m, generations)
data = m.data_collector.get_model_vars_dataframe()
plot_scores(data)
plot_feat_vecs(data)

# plt.clf()
# data['Tit_for_tat_Agents'].plot(legend=True)
# data['Neural_Agents'].plot(legend=True)
# plt.savefig(f'{folder}/agents.png')
