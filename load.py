from os.path import exists
from os import makedirs
from lib import model
from plot import *
from datetime import datetime

in_folder = f"out/15_11_2021_22:17"
assert exists(in_folder)
out_folder = f"{in_folder}_detail"
makedirs(out_folder)

m = model.load_model(f'{in_folder}/model.pickle')

data = m.data_collector.get_model_vars_dataframe()
plot_scores(data, out_folder)
plot_feat_vecs(data, out_folder)
