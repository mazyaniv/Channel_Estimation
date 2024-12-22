import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-0.8,0.35,10)
sigma_space2 = np.logspace(-0.8,0.35,16) #for MMSE load
na = 1
nq = 40
matrix_const1 = Matrix(na, nq)
monte = 50
bound_sim = int(1e3)
plot_dict = {'LMMSE': 1, 'MMSE': 1, 'WBCRB': 1, 'CRB': 1, 'BBZ': 1,'Bhattacharyya':1, 'Approx': 1}

fig = plt.figure(figsize=(10, 6))
if plot_dict['LMMSE'] == 1:
    LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
    plt.plot(10 * np.log10(1 / sigma_space), LMMSE, linestyle='--', label="LMMSE")
if plot_dict['MMSE'] == 1:
    MMSE = np.delete(np.load(f'MMSE/MMSE,na=1,nq=40,sim=10000.npy'), [2, 7])
    plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [2, 7])), MMSE, linestyle='-.', color='red', label="MMSE")
if plot_dict['WBCRB'] == 1:
    WBCRB = np.load(f'WBCRB_Mixed/WBCRB,na={na},nq={nq},sim=50.npy')
    plt.plot(10 * np.log10(1 / sigma_space), WBCRB, linestyle=':', label="WBCRB")
if plot_dict['CRB'] == 1:
    CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
    plt.plot(10 * np.log10(1 / sigma_space), CRB1, color='black', label="BCRB")
if plot_dict['BBZ'] == 1:
    BBZ = [BBZ_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1, monte,0.005) for i in range(len(sigma_space))]
    plt.plot(10 * np.log10(1 / sigma_space), BBZ,  label="BBZ")
if plot_dict['Bhattacharyya'] == 1:
    BBZ = [Bhattacharyya_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1, monte) for i in range(len(sigma_space))]
    plt.plot(10 * np.log10(1 / sigma_space), BBZ, label="Bhattacharyya")
if plot_dict['Approx'] == 1 and plot_dict['LMMSE'] == 1 and plot_dict['CRB'] == 1:
    CRB1_copy = np.nan_to_num(CRB1, nan=0.005)
    probability_vec = [probability(sigma_space[i],nq,matrix_const1,monte) for i in range(len(sigma_space))]
    L_App = [(1-probability_vec[i])*CRB1_copy[i]+probability_vec[i]*LMMSE[i] for i in range(len(sigma_space))]
    plt.plot(10 * np.log10(1 / sigma_space), L_App, linestyle='-', color='green', label="Approximation")

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