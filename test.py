from lib.model import *
from matplotlib import pyplot as plt
import numpy as np

m = PDModel(seed=MESA_SEED, agent_type='mixed')
m.run(1000)

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
