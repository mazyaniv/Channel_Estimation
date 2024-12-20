import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-0.8,0.35,10)
na = 1
nq = 40
matrix_const1 = Matrix(na, nq)
monte = 50
bound_sim = 5000

L_Estimator_analytic1 = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
WBCRB =  [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq,matrix_const1, monte) for i in range(len(sigma_space))]
CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
######################
fig = plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(1 / sigma_space), L_Estimator_analytic1, linestyle='--', label="LMMSE")
plt.plot(10*np.log10(1/sigma_space), WBCRB,linestyle=':', label = "WBCRB")
plt.plot(10 * np.log10(1 / sigma_space), CRB1, color='black', label="BCRB")

ax = plt.gca()
ax.set_xticks(np.arange(-3.6, 8, 0.5), minor=True)
ax.grid(which='major', alpha=1)
ax.grid(which='minor', linestyle="--", alpha=0.5)

plt.title(f"$n_a$={na},$n_q$={nq}")
# plt.xlim(-3.6, 8) #TODO: note
plt.yscale('log')
plt.ylabel('MSE\BCRB')
plt.xlabel(r"$SNR_{[dB]}$")
plt.xticks()
plt.yticks()
plt.legend(loc='lower left', ncol=1)
plt.show()