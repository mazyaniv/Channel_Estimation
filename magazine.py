import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

plot_result = 1
save_to_mat = 0

chosen_space = np.linspace(-5, 15, 27) #dB
sigma_space = 10**(-chosen_space/10)
list_output = []
na,nq = 1,100
bound_sim = 1500
plot_dict = {'LMMSE': 0, 'MMSE': 0 ,'Approx': 0, 'OPT':1,'WBCRB': 0, 'BCRB': 0}
matrix_const0 = Matrix(na, 0)
matrix_const1 = Matrix(na, nq)

if plot_dict['LMMSE'] == 1:
    LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
    list_output.append(LMMSE)
if plot_dict['MMSE'] == 1:
    MMSE = [MMSE_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1, 5000,bound_sim) for i in range(len(sigma_space))]
    list_output.append(MMSE)
if plot_dict['Approx'] == 1:
    WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1,10*bound_sim) for i in range(len(chosen_space))]
    probability_vec = [probability(sigma_space[i],na,nq, matrix_const1, bound_sim) for i in range(len(chosen_space))]
    if na != 0:
        BCRB_a = [weighted_BCRB(sigma_space[i], sigma_space[i], na, 0, matrix_const0, bound_sim) for i in
                  range(len(chosen_space))]
        L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(chosen_space))]
    else:
        L_App = [probability_vec[i] * (1-2/math.pi) + (1 - probability_vec[i]) * WBCRB[i] for i in range(len(chosen_space))]
    list_output.append(L_App)
if plot_dict['OPT'] == 1:
    OPT = [weights_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim) for i in range(len(chosen_space))]
    list_output.append(OPT)
if plot_dict['WBCRB'] == 1:
    if plot_dict['Approx'] == 0:
        WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim) for i in range(len(chosen_space))]
    list_output.append(WBCRB)
if plot_dict['BCRB'] == 1:
    BCRB = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(chosen_space))]
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
    plt.yscale('log')
    plt.ylabel('MSE')
    plt.xlabel(r"$SNR_{[dB]}$")
    plt.legend(loc='lower left', ncol=1)
    plt.show()
if save_to_mat:
    key_list = [key for key, value in plot_dict.items() if value == 1]
    save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
    os.makedirs(save_folder, exist_ok=True)
    # file_path = os.path.join(save_folder, 'SNR.mat')
    # savemat(file_path, {"SNR": chosen_space})
    for i in range(len(key_list)):
        savemat(os.path.join(save_folder, key_list[i]+'.mat'), {key_list[i]: list_output[i]})

