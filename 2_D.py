import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

plot_result = 1
save_to_mat = 0
grid_size = 20
sigma,thresh = np.logspace(-1, 1, grid_size), np.linspace(-2, 2, grid_size)
X, Y = np.meshgrid(sigma,thresh)
list_output = []
na,nq = 2,100
bound_sim = 100
plot_dict = {'LMMSE': 0, 'MMSE': 0 ,'Approx': 0, 'OPT':0,'WBCRB': 1, 'BCRB': 1}
matrix_const0 = Matrix(na, 0)
matrix_const1 = Matrix(na, nq)

if plot_dict['LMMSE'] == 1:
    LMMSE = np.array([[MSE_general_numerical(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    list_output.append(LMMSE)
if plot_dict['MMSE'] == 1:
    MMSE = np.array([[MMSE_func(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    list_output.append(MMSE)
if plot_dict['Approx'] == 1:
    WBCRB = np.array([[weighted_BCRB(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    probability_vec = np.array([[probability(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    if na != 0:
        BCRB_a = np.array([[weighted_BCRB(sigma[j], sigma[j], na, 0, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
        L_App = np.array([[probability_vec[i,j]*BCRB_a[i,j]+(1-probability_vec[i,j])*WBCRB[i,j] for j in
                   range(grid_size)] for i in range(grid_size)])
    else:
        L_App = np.array([[probability_vec[i, j] * (1-2/math.pi) + (1 - probability_vec[i, j]) * WBCRB[i, j] for j in
                           range(grid_size)] for i in range(grid_size)])
    list_output.append(L_App)
if plot_dict['OPT'] == 1:
    OPT = np.array([[weighted_fun(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    list_output.append(OPT)
if plot_dict['WBCRB'] == 1:
    if plot_dict['Approx'] == 0:
        WBCRB = np.array([[weighted_BCRB(sigma[j], sigma[j], na, nq, matrix_const1, bound_sim, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    list_output.append(WBCRB)
if plot_dict['BCRB'] == 1:
    BCRB = np.array([[CRB(sigma[j], sigma[j], na, nq, matrix_const1, 100, thresh[i], thresh[i]) for j in range(grid_size)] for i in range(grid_size)])
    list_output.append(BCRB)
if plot_result:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    plots = {}
    if plot_dict['LMMSE']: plots['LMMSE'] = ('blue',0.8, LMMSE)
    if plot_dict['MMSE']: plots['MMSE'] = (None,0.5,MMSE)
    if plot_dict['Approx']: plots['L_App'] = ('green',0.5, L_App)
    if plot_dict['OPT']: plots['OPT'] = (None,0.5,OPT)
    if plot_dict['WBCRB']: plots['WBCRB'] = ('red',1, WBCRB)
    if plot_dict['BCRB']: plots['BCRB'] = ('black',0.5, BCRB)
    for key, (color,alpha, data) in plots.items():
        if key in locals():
            ax.plot_surface(10 * np.log10(1 / X), Y, data,alpha=alpha, color=color, label=key)
    ax.set_xlabel(r'$10\log_{10}(1/\sigma^2)$')
    ax.set_ylabel(r'$\tau$')
    # ax.set_zlabel('BCRB')
    # ax.set_title('Bound')
    plt.show()
