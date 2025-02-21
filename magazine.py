from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os

lower_segment = np.linspace(-3.5, 2.5, 8)
upper_segment = np.linspace(2.5, 12.5, 20)
db_range = np.concatenate((lower_segment, upper_segment[1:]))
sigma_space = 10 ** (-db_range / 10)
save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
save_to_mat = True
os.makedirs(save_folder, exist_ok=True)
file_path = os.path.join(save_folder, 'SNR.mat')
savemat(file_path, {'SNR': db_range})
chosen_space = sigma_space
# index_mmse = 12
# sigma_space = np.logspace(-2,0.35,16) #np.logspace(-0.8,0.35,20)
# sigma_space_MMSE = np.linspace(sigma_space[-index_mmse:][0],sigma_space[-index_mmse:][-1],20)
# sigma_space_MMSE_2 = np.concatenate((sigma_space[:-index_mmse],sigma_space_MMSE))

resource = [[2,40,'black']]#,[1,100,'black']]#[[2,40,'red'],[1,100,'black']]
monte,snap = 100,5000
bound_sim = int(1e3)
plot_dict = {'LMMSE': 1, 'MMSE': 0 ,'Approx': 1, 'WBCRB': 1,'OPT': 1, 'CRB': 1}
fig = plt.figure(figsize=(10, 6))

for na,nq,color in resource:
    matrix_const0 = Matrix(na, 0)
    matrix_const1 = Matrix(na, nq)
    if plot_dict['LMMSE'] == 1:
        LMMSE = [MSE_zertothresh_analytic(chosen_space[i], chosen_space[i], na, nq) for i in range(len(chosen_space))]
        plt.plot(10 * np.log10(1 / chosen_space), LMMSE,linestyle='--',marker="o", label=f"LMMSE,$n_a$={na},$n_q$={nq}")
    if plot_dict['MMSE'] == 1: #Basically I need more snap for stability
        MMSE= np.load(f'MMSE/MMSE,na={na},nq={nq},snap=10000,monte=1000.npy')
        plt.plot(np.delete(10 * np.log10(1/chosen_space),[0]), np.delete(MMSE,[0]), linestyle='--', color=color, label=f"MMSE,$n_a$={na},$n_q$={nq}")
        #np.delete(np.load(f'MMSE/MMSE,na=1,nq=40,snap=10000,monte=1000.npy'),[5])#[MMSE_func(chosen_space[i], chosen_space[i], na, nq, matrix_const1, monte,snap) for i in range(len(chosen_space))]
        # MMSE1 = [MMSE_func(sigma_space_MMSE[i], sigma_space_MMSE[i], na, nq, matrix_const1, 700,1200) for i in range(len(sigma_space_MMSE))]
        # MMSE2 = LMMSE[:-index_mmse]
        # MMSE = np.concatenate((MMSE2, MMSE1), axis=0)
    if plot_dict['Approx'] == 1:
        WBCRB = [weighted_BCRB(chosen_space[i], chosen_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(chosen_space))]
        BCRB_a = [CRB(chosen_space[i], chosen_space[i], na,0, matrix_const0, bound_sim) for i in range(len(chosen_space))]
        probability_vec = [probability(chosen_space[i],na,nq, matrix_const1, bound_sim) for i in range(len(chosen_space))]
        L_App = [probability_vec[i]*BCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(chosen_space))]
        plt.plot(10 * np.log10(1 / chosen_space), L_App,marker='x', label=f"Approximation,$n_a$={na},$n_q$={nq}")

    if plot_dict['OPT'] == 1:
        OPT = [ET_CRB(chosen_space[i], chosen_space[i], na, nq, bound_sim) for i in range(len(chosen_space))]
        OPT[0] = LMMSE[0]
        plt.plot(10 * np.log10(1 / chosen_space), OPT,marker='v', label=f"OPT,$n_a$={na},$n_q$={nq}")
        # WBCRB1 = np.delete(np.load(f'Bounds_Mixed/WBCRB,na={na},nq={nq},sim=1000.npy'),[3,5,7])
        # plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [3,5,7])), WBCRB1,color='purple',marker="o",linestyle=':', label="WBCRB_old")
    if plot_dict['WBCRB'] == 1:
        if plot_dict['Approx'] == 0:
            WBCRB = [weighted_BCRB(chosen_space[i], chosen_space[i], na, nq, matrix_const1,bound_sim) for i in range(len(chosen_space))]
        plt.plot(10 * np.log10(1 / chosen_space), WBCRB,marker='.', label=f"WBCRB,$n_a$={na},$n_q$={nq}")
        # WBCRB1 = np.delete(np.load(f'Bounds_Mixed/WBCRB,na={na},nq={nq},sim=1000.npy'),[3,5,7])
        # plt.plot(10 * np.log10(1 / np.delete(sigma_space2, [3,5,7])), WBCRB1,color='purple',marker="o",linestyle=':', label="WBCRB_old")

    if plot_dict['CRB'] == 1:
        CRB1 = [CRB(chosen_space[i], chosen_space[i], na, nq, matrix_const1, bound_sim) for i in range(len(chosen_space))]
        plt.plot(10 * np.log10(1 / chosen_space), CRB1,color='black', label=f"BCRB,$n_a$={na},$n_q$={nq}")

    # if plot_dict['BBZ'] == 1:
    #     BBZ = np.load(f'Bounds_Mixed/BBZ,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(BBZ),color='purple',marker=".",  label="BBZ")
    # if plot_dict['Bhattacharyya'] == 1:
    #     Bhattacharyya = np.load(f'Bounds_Mixed/Bhattacharyya,na={na},nq={nq},sim=1000.npy')
    #     plt.plot(10 * np.log10(1 / sigma_space), np.real(Bhattacharyya), color='grey', marker="x", label="Bhattacharyya")
if save_to_mat:
    save_folder = r'C:\Users\Yaniv\Documents\MATLAB'
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, 'LMMSE.mat')
    savemat(file_path, {'LMMSE': LMMSE})

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
