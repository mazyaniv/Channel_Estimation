import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-0.8,0.35,10)
sigma_space2 = np.logspace(-0.8,0.35,16) #for MMSE load
na = 1
nq = 40
matrix_const1 = Matrix(na, nq)
monte = 100
bound_sim = 5000

LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
MMSE = np.delete(np.load(f'MMSE/MMSE,na=1,nq=40,sim=10000.npy'), [2, 7])
WBCRB = np.load(f'WBCRB_Mixed/WBCRB,na={na},nq={nq},sim=50.npy')
CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]

# Bhattacharyya = [Bhattacharyya_Mixed(sigma_space[i], sigma_space[i], na, nq, matrix_const1, monte) for i in range(len(sigma_space))]
# CRB1_copy = np.nan_to_num(CRB1, nan=0.001)
# probability_vec = [probability(sigma_space[i],nq,matrix_const1,monte) for i in range(len(sigma_space))]
# L_App = [(1-probability_vec[i])*CRB1_copy[i]+probability_vec[i]*LMMSE[i] for i in range(len(sigma_space))]
######################
fig = plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(1 / sigma_space), LMMSE, linestyle='--', label="LMMSE")
plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [2, 7])), MMSE, linestyle='-.',color='red', label="MMSE")
# plt.plot(10 * np.log10(1 / sigma_space), L_App, linestyle='-', color='green', label="Approximation")
plt.plot(10*np.log10(1/sigma_space), WBCRB,linestyle=':', label = "WBCRB")
plt.plot(10 * np.log10(1 / sigma_space), CRB1, color='black', label="BCRB")
# plt.plot(10 * np.log10(1 / sigma_space), Bhattacharyya, color='purple',marker="*" ,label="Bhattacharyya")

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