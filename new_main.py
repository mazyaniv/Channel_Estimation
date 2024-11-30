import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-1,1,10)
na = 0
nq = 10#100
matrix_const1 = Matrix(na,nq)
bound_sim = 1000
monte = 50

L_Estimator_analytic1 = [MSE_general_numerical(sigma_space[i],sigma_space[i], na,nq,matrix_const1,1000,1000) for i in range(len(sigma_space))]
######################
fig = plt.figure(figsize=(10, 6))
plt.plot(10*np.log10(1/sigma_space), L_Estimator_analytic1,linestyle='--',marker="x", label = "LMMSE")
ax = plt.gca()
ax.set_xticks(np.arange(-10,10,0.5), minor=True)
ax.grid(which='major', alpha=1)
ax.grid(which='minor',linestyle="--",alpha=0.5)
plt.grid()
plt.title(f"$n_q$={nq}")
plt.yscale('log')
plt.ylabel('MSE\BCRB')
plt.xlabel(r"$SNR_{[dB]}$")
plt.xticks()
plt.yticks()
plt.legend(loc='lower left', ncol=1)
plt.show()