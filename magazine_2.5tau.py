import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

chosen_space = np.linspace(-7.5, 5, 20) #dB
sigma_space = 10**(-chosen_space/10)
plot_result = 1
save_to_mat = 0
list_output = []
na,nq = 1,100
bound_sim = 500
thresh = 2
matrix_const0 = Matrix(na, 0)
matrix_const1 = Matrix(na, nq)
plot_dict = {'LMMSE': 1, 'MMSE': 0 ,'Approx': 1, 'OPT':1,'WBCRB': 1, 'BCRB': 1}

if plot_dict['LMMSE'] == 1:
    LMMSE = np.squeeze([MSE_general_numerical(sigma_space[i], sigma_space[i], na, nq,matrix_const1,20000,thresh,thresh) for i in range(len(chosen_space))])
    list_output.append(LMMSE)
if plot_dict['MMSE'] == 1:
    MMSE = np.load('MMSE/MMSE,na=1,nq=100,thresh=2.5,snap=12000,monte=1200.npy')
    list_output.append(MMSE)
if plot_dict['Approx'] == 1:
    WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
    BCRB_a = [CRB(sigma_space[i], sigma_space[i], na, 0, matrix_const0,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
    probability_vec = [probability_new(sigma_space[i],na,nq, matrix_const1, bound_sim,20,thresh,thresh) for i in range(len(chosen_space))]
    L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(chosen_space))]
    list_output.append(L_App)
if plot_dict['OPT'] == 1:
    OPT = [weights_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
    list_output.append(OPT)
if plot_dict['WBCRB'] == 1:
    if not plot_dict['Approx']:
        WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
    list_output.append(WBCRB)

if plot_dict['BCRB'] == 1:
    BCRB = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, 10000,thresh,thresh) for i in range(len(chosen_space))]
    list_output.append(BCRB)

if plot_result:
    fig = plt.figure(figsize=(10, 6))
    plots = {}
    if plot_dict['LMMSE']: plots['LMMSE'] = ('--', "o", LMMSE)
    if plot_dict['MMSE']: plots['MMSE'] = (None, "^", MMSE)
    if plot_dict['Approx']: plots['L_App'] = ('--', "v", L_App)
    if plot_dict['OPT']: plots['OPT'] = ('--', "s", OPT)
    if plot_dict['WBCRB']: plots['WBCRB'] = (None, ".", WBCRB)
    if plot_dict['BCRB']: plots['BCRB'] = (None, None, BCRB)
    for key, (linestyle, marker, data) in plots.items():
        if key in locals():
            plt.plot(chosen_space, data, linestyle=linestyle, marker=marker, label=key)
    ax = plt.gca()
    ax.grid(which='major', alpha=1)
    ax.grid(which='minor', linestyle="--", alpha=0.5)
    plt.title(f"Threshold = {thresh}")
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel(r"$SNR_{[dB]}$")
    plt.legend(loc='lower left', ncol=1)
    plt.show()
if save_to_mat:
    key_list = [key for key, value in plot_dict.items() if value == 1]
    save_folder = r'C:\Users\Yaniv\Documents\MATLAB\tau=2'
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, 'SNR_tau2.mat')
    savemat(file_path, {"SNR_tau2": chosen_space})
    for i in range(len(key_list)):
        savemat(os.path.join(save_folder, key_list[i]+'_tau2.mat'), {key_list[i]+'_tau2': list_output[i]})


