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
# index_mmse = 12
# sigma_space = np.logspace(-2,0.35,16) #np.logspace(-0.8,0.35,20)
# sigma_space_MMSE = np.linspace(sigma_space[-index_mmse:][0],sigma_space[-index_mmse:][-1],20)
# sigma_space_MMSE_2 = np.concatenate((sigma_space[:-index_mmse],sigma_space_MMSE))

resource = [[1,100]]
bound_sim = 1000
thresh = 0
plot_dict = {'LMMSE': 1, 'MMSE': 0 ,'Approx': 0 ,'WBCRB': 1, 'CRB': 1}
for na,nq in resource:
    matrix_const0 = Matrix(na, 0)
    matrix_const1 = Matrix(na, nq)
    if plot_dict['LMMSE'] == 1:
        if thresh == 0:
            LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
        else:
            LMMSE = np.squeeze([MSE_general_numerical(sigma_space[i], sigma_space[i], na, nq,matrix_const1,10000,thresh,thresh) for i in range(len(chosen_space))])
        list_output.append(LMMSE)
        # plt.plot(chosen_space, LMMSE,linestyle='--',marker="o", label=f"LMMSE")#,$n_a$={na},$n_q$={nq}")
    if plot_dict['MMSE'] == 1: #Basically I need more snap for stability
        # MMSE = [MMSE_func(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim,100,thresh,thresh) for i in range(len(sigma_space))]
        MMSE= np.load('Numeric_thresh/MMSE/E_value_thresh,na=1,nq=100,snap=5000,monte=500.npy')
        list_output.append(MMSE)
        # MMSE[-1], MMSE[-2] = LMMSE[-1], LMMSE[-2]
        # plt.plot(np.delete(10 * np.log10(1/chosen_space),[0]), np.delete(MMSE,[0]), linestyle='--', color=color, label=f"MMSE")#,$n_a$={na},$n_q$={nq}")
        #np.delete(np.load(f'MMSE/MMSE,na=1,nq=40,snap=10000,monte=1000.npy'),[5])#[MMSE_func(chosen_space[i], chosen_space[i], na, nq, matrix_const1, monte,snap) for i in range(len(chosen_space))]
        # MMSE1 = [MMSE_func(sigma_space_MMSE[i], sigma_space_MMSE[i], na, nq, matrix_const1, 700,1200) for i in range(len(sigma_space_MMSE))]
        # MMSE2 = LMMSE[:-index_mmse]
        # MMSE = np.concatenate((MMSE2, MMSE1), axis=0)
    if plot_dict['Approx'] == 1:
        CRB11 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
        BCRB_a = LMMSE#[CRB(sigma_space[i], sigma_space[i], na,0, matrix_const0, bound_sim,thresh,thresh) for i in range(len(chosen_space))]
        probability_vec = [probability(sigma_space[i],na,nq, matrix_const1, bound_sim,thresh,thresh) for i in range(len(chosen_space))]
        L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*CRB11[i] for i in range(len(chosen_space))]
        list_output.append(L_App)
        # plt.plot(chosen_space, L_App,marker='x', label=f"Approximation")#,$n_a$={na},$n_q$={nq}")

    # if plot_dict['OPT'] == 1:
    #     OPT = [ET_CRB(sigma_space[i], sigma_space[i], na, nq, bound_sim) for i in range(len(chosen_space))]
    #     OPT[0], OPT[1] = LMMSE[0], LMMSE[1]
    #     for i in range(len(chosen_space)):
    #         if math.isnan(OPT[i]):
    #             OPT[i] = WBCRB[i]
    #     list_output.append(OPT)
    #     # plt.plot(chosen_space, OPT,marker='v', label=f"Opt")#,$n_a$={na},$n_q$={nq}")
    #     # WBCRB1 = np.delete(np.load(f'Bounds_Mixed/WBCRB,na={na},nq={nq},sim=1000.npy'),[3,5,7])
    #     # plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [3,5,7])), WBCRB1,color='purple',marker="o",linestyle=':', label="WBCRB_old")
    if plot_dict['WBCRB'] == 1:
        # if plot_dict['Approx'] == 0:
        WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], [na,na], [nq,nq], matrix_const1,bound_sim,thresh,thresh) for i in range(len(chosen_space))]
            # WBCRB = np.load('WBCRB_pure_1bit/WBCRB,na=0,nq=100,thresh=2.5,sim=100000.npy')
        list_output.append(WBCRB)
        # plt.plot(chosen_space, WBCRB,marker='.', label=f"WBCRB")#,$n_a$={na},$n_q$={nq}")
        # WBCRB1 = np.delete(np.load(f'Bounds_Mixed/WBCRB,na={na},nq={nq},sim=1000.npy'),[3,5,7])
        # plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [3,5,7])), WBCRB1,color='purple',marker="o",linestyle=':', label="WBCRB_old")

    if plot_dict['CRB'] == 1:
        CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim,thresh,thresh) for i in range(len(chosen_space))]
        list_output.append(CRB1)
    if plot_result:
        plt.plot(chosen_space, LMMSE,  linestyle='--', marker="o", label=f"LMMSE") #if plot_dict['LMMSE']=1
        # plt.plot(chosen_space, MMSE, marker="^", label=f"approx_wbcrb")
        # plt.plot(chosen_space, L_App,marker="v",linestyle='--', label=f"Approximation_bcrb")
        plt.plot(chosen_space, WBCRB, marker='.', label=f"WBCRB")
        plt.plot(chosen_space, CRB1, label=f"BCRB")
    if save_to_mat:
        key_list = [key for key, value in plot_dict.items() if value == 1]
        save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
        os.makedirs(save_folder, exist_ok=True)
        file_path = os.path.join(save_folder, 'SNR_Thersh.mat')
        savemat(file_path, {"chosen_space": chosen_space})
        for i in range(len(key_list)):
            file_path = os.path.join(save_folder, key_list[i] + '.mat')
            savemat(file_path, {key_list[i]: list_output[i]})
    # if plot_dict['BBZ'] == 1:
    #     BBZ = np.load(f'Bounds_Mixed/BBZ,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(BBZ),color='purple',marker=".",  label="BBZ")
    # if plot_dict['Bhattacharyya'] == 1:
    #     Bhattacharyya = np.load(f'Bounds_Mixed/Bhattacharyya,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(Bhattacharyya), color='grey', marker="x", label="Bhattacharyya")

ax = plt.gca()
# ax.set_xticks(np.arange(-4, 12.5, 0.5), minor=True)
ax.grid(which='major', alpha=1)
ax.grid(which='minor', linestyle="--", alpha=0.5)
# plt.title(f"Threshold = {thresh}")
# plt.xlim(-3.6, 8)
# plt.ylim([np.min(list_output) * 0.8, np.max(list_output) * 1.2])  # Set y-axis limits
plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel(r"$SNR_{[dB]}$")
plt.xticks()
plt.yticks()
plt.legend(loc='lower left', ncol=1)
plt.show()
