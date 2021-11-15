from os.path import exists
from lib import model

folder = f"out/15_11_2021_21:20"
assert exists(folder)

m = model.load_model(f'{folder}/model.pickle')

print(m.data_collector.get_model_vars_dataframe())
