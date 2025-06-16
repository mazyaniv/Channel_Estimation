import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

bound_sim = 1000
n_list = np.linspace(10, 100, 12, dtype=int)
weights = [weights_func(1, 1, 0,n,Matrix(0,n),500) for n in n_list]
WBCRB_quantize = [weighted_BCRB(1, 1, 0,n,Matrix(0,n),bound_sim) for n in n_list]

plt.plot(n_list, weights,label='weights')
plt.plot(n_list, WBCRB_quantize,label='wbcrb')
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
