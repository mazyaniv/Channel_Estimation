import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-2, 1, 16)  # np.logspace(-0.8,0.35,16)
na = 1
nq = 40
matrix_const1 = Matrix(na, nq)
monte = 10000  # number of experiments
bound_sim = 5000

s = 0.5
h = 0.5
mu1 = h

L_Estimator_analytic1 = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in
                         range(len(sigma_space))]
# MMSE = np.load(f"Expected_value/E_value,na={na},nq={nq},snap=10000,monte=1000.npy")
# [E_theta_givenx_numeric(sigma_space[i],sigma_space[i], na,nq,matrix_const1,1000,1000) for i in range(len(sigma_space))]
# WBCRB =  np.load(f"WBCRB_pure_1bit/weighted_BCRB,na=0,nq=40.npy")
# [Bhattacharyya_onebit(sigma_space[i], sigma_space[i], na, nq, monte) for i in range(len(sigma_space))]
# [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, 10) for i in range(len(sigma_space))]
CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]

L_Estimator_analytic_app = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, 0) for i in
                            range(len(sigma_space))]
CRB1_copy = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
CRB1_copy = np.nan_to_num(CRB1_copy, nan=0.001)
probability_vec = [probability(sigma_space[i], nq, matrix_const1, monte) for i in range(len(sigma_space))]
L_ALTERNATIVE = [(1 - probability_vec[i]) * CRB1_copy[i] + probability_vec[i] * L_Estimator_analytic_app[i] for i in
                 range(len(sigma_space))]  # (sigma_teta**2/math.sqrt(2))
######################
fig = plt.figure(figsize=(10, 6))
plt.plot(10 * np.log10(1 / sigma_space), L_Estimator_analytic1, linestyle='--', label="LMMSE")
# plt.plot(np.concatenate((10*np.log10(1/sigma_space)[::2],np.array([10*np.log10(1/sigma_space)[-1]]))), np.concatenate((MMSE[::2],np.array([MMSE[-1]]))), label='MMSE')
plt.plot(10 * np.log10(1 / sigma_space), L_ALTERNATIVE, linestyle="--", label="Approx.")
# plt.plot(10*np.log10(1/sigma_space), WBCRB,linestyle=':', label = "WBCRB")
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