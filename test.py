from lib.model import *
from matplotlib import pyplot as plt
import numpy as np


def agent_type_map(x, y):
    if x in [0, 2, 4, 6, 8]:
        return 'tit_for_tat'
    return 'neural'


m = PDModel(10, 10, seed=MESA_SEED, agent_type='mixed', agent_type_map=agent_type_map)
m.run(10000)

data = m.data_collector.get_model_vars_dataframe()

data['Mean_Score'].plot(legend=True)
data['Max_Score'].plot(legend=True)
data['Min_Score'].plot(legend=True)
plt.savefig('scores.png')

plt.clf()
vecs = data['Mean_Feature_Vector']
vec_0 = [v[0] for v in vecs]
vec_1 = [v[1] for v in vecs]
plt.plot(vec_0, label='0')
plt.plot(vec_1, label='1')
plt.legend()
plt.savefig('feat_vec.png')

plt.clf()
data['Tit_for_tat_Agents'].plot(legend=True)
data['Neural_Agents'].plot(legend=True)
plt.savefig('agents.png')
