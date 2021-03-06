from datetime import datetime
from os import makedirs

from plot import *

details = True
folder = f"out/{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
print(f"Results will be saved to {folder}")
makedirs(folder)
makedirs(f"{folder}/maps")

with open(f'{folder}/config.txt', 'w') as f:
    f.write(config_to_str())


def agent_type_map(x, y):
    if ((x == 3 or x == 7) and 3 <= y <= 7) or ((y == 3 or y == 7) and 3 <= x <= 7):
        return 'tit_for_tat'
    return 'neural'


def make_callback(m, period, offset):
    def callback(i):
        j = i + offset
        if j % period == 0:
            plot_feature_map(m, j, f"{folder}/maps")

    return callback


if NUMPY_SEED is not None:
    np.random.seed(NUMPY_SEED)

m = PDModel(seed=MESA_SEED, agent_type='neural')
if not details:
    m.run(1600)
    data = m.data_collector.get_model_vars_dataframe()
    plot_scores(data, folder)
    plot_feat_vecs(data, folder)
else:
    # Fixing the seed and study the details of this code run
    m.run(1000)
    m.run(1200 - 1000, make_callback(m, 10, 1000))  # plot the map every 10 generations
    plot_defecting_ratio_map(m, 1200, folder)  # plot the defecting ratio map at the 1200th generation
    m.run(1600 - 1200, make_callback(m, 10, 1200))
    data = m.data_collector.get_model_vars_dataframe()
    # Plot more results over generations
    plot_scores(data, folder, 1000, 1600)
    plot_feat_vecs(data, folder, 1000, 1600)

m.dump(f"{folder}/model.pickle")  # save the model after 1600 generations
