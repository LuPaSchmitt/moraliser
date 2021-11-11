import numpy as np
from matplotlib import pyplot as plt
aver_list = np.loadtxt('readme.txt')

plt.plot(range(len(aver_list)), aver_list)
plt.ylim((0,100))
plt.show()