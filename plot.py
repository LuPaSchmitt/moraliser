import numpy as np
from matplotlib import pyplot as plt
from lib.model import *

# start = 4400
# end = 4800


def plot_feature_map(m, i, folder):
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
    # plt.clf()
    # plt.imshow(feature_map1, interpolation='nearest')
    # plt.colorbar()
    # plt.savefig(f'{folder}/{i}_feature_map1.png')


def plot_agent_type_map(m, i, folder):
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


def plot_scores(data, folder, start=0, end=None):
    if end is None:
        end = len(data)
    plt.clf()
    x = np.arange(start, end)
    plt.plot(x, data['Mean_Score'][start:end], label='Mean')
    plt.plot(x, data['Max_Score'][start:end], label='Max')
    plt.plot(x, data['Min_Score'][start:end], label='Min')
    plt.xlabel('Generation')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig(f'{folder}/scores.png')


def plot_feat_vecs(data, folder, start=0, end=None):
    if end is None:
        end = len(data)
    plt.clf()
    vecs = data['Mean_Feature_Vector'][start:end]
    x = np.arange(start, end)
    vec_0 = [v[0] for v in vecs]
    vec_1 = [v[1] for v in vecs]
    plt.plot(x, vec_0, label='0')
    plt.plot(x, vec_1, label='1')
    plt.xlabel('Generation')
    plt.ylabel('Feature')
    plt.legend()
    plt.savefig(f'{folder}/feat_vec.png')
