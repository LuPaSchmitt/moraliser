from os.path import exists
from lib import model

folder = f"out/15_11_2021_21:39"
assert exists(folder)

m = model.load_model(f'{folder}/model.pickle')

print(m.initial_agents)
print(m.data_collector.get_model_vars_dataframe())
