import numpy as np
from matplotlib import pyplot as plt
from function_main import *

sigma_space = np.logspace(-2,0.35,20) #np.logspace(-0.8,0.35,16)
sigma_space_MMSE = np.linspace(sigma_space[-10:][0],sigma_space[-10:][-1],20)
sigma_space_MMSE_2 = np.concatenate((sigma_space[:-10],sigma_space_MMSE))
resource = [[1,40]]#[[2,40,'red'],[1,100,'black']]
monte = 500
bound_sim = 1000
plot_dict = {'LMMSE': 1, 'MMSE': 1,'Approx': 1, 'WBCRB': 1, 'CRB': 1}

fig = plt.figure(figsize=(10, 6))
for na,nq in resource:
    matrix_const0 = Matrix(na, 0)
    matrix_const1 = Matrix(na, nq)
    if plot_dict['LMMSE'] == 1:
        LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
        plt.plot(10 * np.log10(1 / sigma_space), LMMSE, linestyle='--', marker="x", label=f"LMMSE,$n_a$={na},$n_q$={nq}")
    if plot_dict['MMSE'] == 1:
        MMSE1 = [MMSE_func(sigma_space_MMSE[i], sigma_space_MMSE[i], na, nq, matrix_const1, 500,200) for i in range(len(sigma_space_MMSE))]#np.delete(np.load(f'MMSE/MMSE,na=1,nq=40,sim=10000.npy'), [2, 7])
        MMSE2 = LMMSE[:-10]
        MMSE = np.concatenate((MMSE2, MMSE1), axis=0)
        plt.plot(10 * np.log10(1 / sigma_space_MMSE_2), MMSE, linestyle='--', marker="x", label=f"MMSE,$n_a$={na},$n_q$={nq}")
        # plt.plot(10 * np.log10(1 / np.delete(sigma_space, [2, 7])), MMSE, linestyle='--', color='red', label="MMSE")
    if plot_dict['Approx'] == 1:
        WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
        BCRB_a = [CRB(sigma_space[i], sigma_space[i], na, 0, matrix_const0, bound_sim) for i in range(len(sigma_space))]
        probability_vec = [probability(sigma_space[i],na,nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
        L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(sigma_space))]
        # L_App = [probability_vec[i]*LMMSE0[i]+(1-probability_vec[i])*CRB1_copy[i] for i in range(len(sigma_space))]
        plt.plot(10 * np.log10(1 / sigma_space), L_App, linestyle='-.', label=f"Approximation,$n_a$={na},$n_q$={nq}")

    if plot_dict['WBCRB'] == 1:
        if plot_dict['Approx'] == 0:
            WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(sigma_space))]
        plt.plot(10 * np.log10(1 / sigma_space), WBCRB,marker="o",linestyle=':', label=f"WBCRB,$n_a$={na},$n_q$={nq}")
        # WBCRB1 = np.delete(np.load(f'Bounds_Mixed/WBCRB,na={na},nq={nq},sim=1000.npy'),[3,5,7])
        # plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [3,5,7])), WBCRB1,color='purple',marker="o",linestyle=':', label="WBCRB_old")
    if plot_dict['CRB'] == 1:
        CRB1 = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix_const1, 10*bound_sim) for i in range(len(sigma_space))]
        plt.plot(10 * np.log10(1 / sigma_space), CRB1, label=f"BCRB,$n_a$={na},$n_q$={nq}")

    # if plot_dict['BBZ'] == 1:
    #     BBZ = np.load(f'Bounds_Mixed/BBZ,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(BBZ),color='purple',marker=".",  label="BBZ")
    # if plot_dict['Bhattacharyya'] == 1:
    #     Bhattacharyya = np.load(f'Bounds_Mixed/Bhattacharyya,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(Bhattacharyya), color='grey', marker="x", label="Bhattacharyya")

    ax = plt.gca()
    ax.set_xticks(np.arange(-3.6, 20, 0.5), minor=True)
    ax.grid(which='major', alpha=1)
    ax.grid(which='minor', linestyle="--", alpha=0.5)
    # plt.title(f"$n_a$={na},$n_q$={nq}")
    # plt.xlim(-3.6, 8) #TODO: note
    plt.yscale('log')
    # plt.ylabel('MSE\BCRB')
    plt.xlabel(r"$SNR_{[dB]}$")
    plt.xticks()
    plt.yticks()
    plt.legend(loc='lower left', ncol=1)
plt.show()