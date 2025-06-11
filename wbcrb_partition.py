import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

lower_segment = np.linspace(-3.5, 2.5, 8)
upper_segment = np.linspace(2.5, 12.5, 20)
chosen_space = np.concatenate((lower_segment, upper_segment[1:])) #dB np.linspace(-5, 12, 10)
sigma_space = 10**(-chosen_space/10)
plot_result = 1
if plot_result:
    fig = plt.figure(figsize=(10, 6))
save_to_mat = 0
save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
os.makedirs(save_folder, exist_ok=True)
file_path = os.path.join(save_folder, 'SNR_thresh.mat')
savemat(file_path, {'SNR_thresh': chosen_space})
list_output = []
resource = [[1,100]]
bound_sim = 200
thresh = 0
plot_dict = {'LMMSE': 1, 'WBCRB': 1, 'WBCRB_q': 1,'WBCRB_a': 1, 'BCRB': 1}
for na,nq in resource:
    matrix_const1 = Matrix(na, nq)
    matrix_constq = Matrix(0, nq)
    matrix_consta = Matrix(na, 0)
    if plot_dict['LMMSE'] == 1:
        LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
        list_output.append(LMMSE)
    if plot_dict['WBCRB'] == 1:
        WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], [na,nq,matrix_const1], [na,nq,matrix_const1], bound_sim, thresh, thresh) for i in range(len(chosen_space))]
        list_output.append(WBCRB)
    if plot_dict['WBCRB_q'] == 1:
        WBCRB_q = [weighted_BCRB(sigma_space[i], sigma_space[i], [0,nq,matrix_constq], [na,nq,matrix_const1], bound_sim, thresh, thresh) for i in range(len(chosen_space))]
        list_output.append(WBCRB_q)
    if plot_dict['WBCRB_a'] == 1:
        WBCRB_a = [weighted_BCRB(sigma_space[i], sigma_space[i], [na,0,matrix_consta], [na,nq,matrix_const1], bound_sim, thresh, thresh) for i in range(len(chosen_space))]
        list_output.append(WBCRB_a)
    if plot_dict['BCRB'] == 1:
        CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim,thresh,thresh) for i in range(len(chosen_space))]
        list_output.append(CRB1)

    if plot_result:
        keys_with_1 = [k for k, v in plot_dict.items() if v == 1]
        for i in range(len(list_output)):
            plt.plot(chosen_space, list_output[i],marker = '.', label=f"{list(keys_with_1)[i]}")
    if save_to_mat:
        key_list = [key for key, value in plot_dict.items() if value == 1]
        save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
        os.makedirs(save_folder, exist_ok=True)
        file_path = os.path.join(save_folder, 'SNR_Thersh.mat')
        savemat(file_path, {"chosen_space": chosen_space})
        for i in range(len(key_list)):
            file_path = os.path.join(save_folder, key_list[i] + '.mat')
            savemat(file_path, {key_list[i]: list_output[i]})

ax = plt.gca()
ax.grid(which='major', alpha=1)
ax.grid(which='minor', linestyle="--", alpha=0.5)
plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel(r"$SNR_{[dB]}$")
plt.xticks()
plt.yticks()
plt.legend(loc='lower left', ncol=1)
plt.show()
