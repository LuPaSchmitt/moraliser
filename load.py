from os import makedirs
from os.path import exists

from lib import model
from plot import *

in_folder = f"out/04_12_2021_23_08_18"  # TODO: fill in your result folder
assert exists(in_folder)
# out_folder = in_folder
out_folder = f"{in_folder}_detail"
makedirs(out_folder)

m = model.load_model(f'{in_folder}/model.pickle')

data = m.data_collector.get_model_vars_dataframe()
plot_scores(data, out_folder)
# plot_feat_vecs(data, out_folder)
