import numpy as np
from matplotlib import pyplot as plt

from lib.model import *


def plot_attr_map(m, i, folder, output_name, agent_attr, im_channels=1, im_dtype=float, cmap=None):
    im = np.zeros((m.grid.height, m.grid.width, im_channels), dtype=im_dtype)
    for cell in m.grid.coord_iter():
        agent, x, y = cell
        attr = agent_attr(agent)
        im[y, x, :] = cmap[attr] if cmap is not None else attr
    plt.clf()
    plt.imshow(im, vmin=0, vmax=1, interpolation='nearest')
    if cmap is None: plt.colorbar()
    plt.savefig(f'{folder}/{i}_{output_name}.png')


def plot_defecting_ratio_map(m, i, folder):
    plot_attr_map(m, i, folder, 'defecting_ratio_map', lambda a: a.defecting_ratio)


def plot_feature_map(m, i, folder):
    f0 = np.zeros((m.grid.height, m.grid.width), dtype=float)
    f1 = np.zeros((m.grid.height, m.grid.width), dtype=float)
    for cell in m.grid.coord_iter():
        agent, x, y = cell
        if isinstance(agent, NeuralAgent):
            f = agent.feature_vector()
            f0[y, x] = f[0]
            f1[y, x] = f[1]
            # print(f)
    plt.clf()
    plt.figure(figsize=(10, 5))
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 5]})
    ax0.imshow(f0, vmin=0, vmax=1, interpolation='nearest')
    img = ax1.imshow(f1, vmin=0, vmax=1, interpolation='nearest')
    ax0.title.set_text('Feature Map 0')
    ax1.title.set_text('Feature Map 1')
    ax0.axis('off')
    ax1.axis('off')
    plt.colorbar(img, ax=ax1)
    plt.savefig(f'{folder}/{i}_feature_map.png')
    plt.close('all')


def plot_agent_type_map(m, i, folder):
    cmap = {
        SimpleAgent: [0, 0, 255],
        TitForTatAgent: [235, 155, 52],  # Orange
        GoodAgent: [0, 255, 0],
        BadAgent: [255, 0, 0],
        NeuralAgent: [237, 14, 233],  # Pink
    }
    plot_attr_map(m, i, folder, 'agent_type_map', type, 3, int, cmap)


def plot_scores(data, folder, start=0, end=None):
    if end is None:
        end = len(data)
    plt.clf()
    plt.figure(figsize=(10, 8))
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
    plt.figure(figsize=(10, 8))
    fv = data['Mean_Feature_Vector'][start:end]  # [rounds, 2^m]
    fv = np.array([f for f in fv])
    x = np.arange(start, end)
    m = int(np.log2(fv.shape[1]))
    for i in range(fv.shape[1]):
        bin_repr = f'{i:0b}'.zfill(m)[::-1]  # 01 sequence from oldest to latest history
        # plt.plot(x, fv[:, i], '-o' if bin_repr[-1] == '0' else '-^', label=bin_repr)
        plt.plot(x, fv[:, i], '-', label=bin_repr)
    plt.plot(x, 0.5 * np.ones_like(x), 'k--')
    plt.xlabel('Generation')
    plt.ylabel('Mean Feature Vector')
    plt.legend()
    plt.savefig(f'{folder}/feat_vec.png')
