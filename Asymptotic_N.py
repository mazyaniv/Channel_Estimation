import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

bound_sim = 1000
n_list = np.linspace(10, 100, 12, dtype=int)
WBCRB = [weighted_BCRB(1, 1, [int(n/2),int(n/2),Matrix(int(n/2),int(n/2))], [int(n/2),int(n/2),Matrix(int(n/2),int(n/2))],bound_sim) for n in n_list]
WBCRB_quantize = [weighted_BCRB(1, 1, [0,int(n/2),Matrix(0,int(n/2))], [int(n/2),int(n/2),Matrix(int(n/2),int(n/2))],bound_sim) for n in n_list]
# WBCRB_analog = [weighted_BCRB(1, 1, [int(n/2),0,Matrix(int(n/2),0)], [int(n/2),int(n/2),Matrix(int(n/2),int(n/2))],bound_sim) for n in n_list]

plt.plot(n_list, WBCRB,label='mixed')
plt.plot(n_list, WBCRB_quantize,label='quantize')
# plt.plot(n_list, WBCRB_analog,label='analog')

ax = plt.gca()
ax.grid(which='major', alpha=1)
ax.grid(which='minor', linestyle="--", alpha=0.5)
# plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel('N')
plt.xticks()
plt.yticks()
plt.legend(loc='lower left', ncol=1)
plt.show()
