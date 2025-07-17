import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

plot_result = 1
save_to_mat = 0
thresh_space = np.linspace(0, 2.5, 20)
sigma = 0.5
bound_sim = 1000
resource = [[1,100]]
plot_dict = {'LMMSE': 0, 'MMSE': 1 ,'Approx': 0, 'OPT':0,'WBCRB': 0, 'BCRB': 0}
for na,nq in resource:
    list_output = []
    matrix_const0 = Matrix(na, 0)
    matrix_const1 = Matrix(na, nq)

    if plot_dict['LMMSE'] == 1:
        LMMSE = [MSE_general_numerical(sigma, sigma, na, nq,matrix_const1,10000,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        list_output.append(LMMSE)
    if plot_dict['MMSE'] == 1:
        MMSE = np.load('MMSE/MMSE,na='+str(na)+',nq='+str(nq)+',sigma=0.5,snap=5000,monte=1000.npy')\
        # [MMSE_func(sigma, sigma, na, nq, matrix_const1, 5000,bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        list_output.append(MMSE)
    if plot_dict['Approx'] == 1:
        WBCRB = [weighted_BCRB(sigma, sigma, na, nq, matrix_const1,bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        probability_vec = [probability(1,na,nq, matrix_const1, bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        if na != 0:
            BCRB_a = [weighted_BCRB(sigma, sigma, na, 0, matrix_const0, bound_sim,thresh_space[i],thresh_space[i]) for i in
                      range(len(thresh_space))]
            L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(thresh_space))]
        else:
            L_App = [probability_vec[i] * (1-2/math.pi) + (1 - probability_vec[i]) * WBCRB[i] for i in range(len(thresh_space))]
        list_output.append(L_App)
    if plot_dict['OPT'] == 1:
        OPT = [weights_func(sigma, sigma, na, nq, matrix_const1,bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        list_output.append(OPT)
    if plot_dict['WBCRB'] == 1:
        if plot_dict['Approx'] == 0:
            WBCRB = [weighted_BCRB(sigma, sigma, na, nq, matrix_const1,bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        list_output.append(WBCRB)
    if plot_dict['BCRB'] == 1:
        BCRB = [CRB(sigma, sigma, na, nq, matrix_const1, bound_sim,thresh_space[i],thresh_space[i]) for i in range(len(thresh_space))]
        list_output.append(BCRB)
    if plot_result:
        fig = plt.figure(figsize=(10, 6))
        plots = {}
        if plot_dict['LMMSE']: plots["LMMSE"] = ('--', "o", LMMSE)
        if plot_dict['MMSE']: plots['MMSE'] = (None, "^", MMSE)
        if plot_dict['Approx']: plots['L_App'] = ('--', "v", L_App)
        if plot_dict['OPT']: plots['OPT'] = ('--', "s", OPT)
        if plot_dict['WBCRB']: plots['WBCRB'] = (None, ".", WBCRB)
        if plot_dict['BCRB']: plots['BCRB'] = (None, None, BCRB)
        for key, (linestyle, marker, data) in plots.items():
            if key in locals():
                plt.plot(thresh_space, data, linestyle=linestyle, marker=marker, label=key+f" $n_a$={na},$n_q$={nq}")
        ax = plt.gca()
        ax.grid(which='major', alpha=1)
        ax.grid(which='minor', linestyle="--", alpha=0.5)
        plt.yscale('log')
        plt.ylabel('MSE')
        plt.xlabel(r"$\tau$")
        plt.legend(loc='lower right', ncol=2)
        plt.show()
if save_to_mat:
        key_list = [key for key, value in plot_dict.items() if value == 1]
        save_folder = r'C:\Users\Yaniv\Documents\MATLAB\thresh'
        os.makedirs(save_folder, exist_ok=True)
        file_path = os.path.join(save_folder, 'thresh.mat')
        savemat(file_path, {"thresh": thresh_space})
        for i in range(len(key_list)):
            savemat(os.path.join(save_folder, key_list[i]+'_thresh.mat'), {key_list[i]+'_thresh': list_output[i]})

