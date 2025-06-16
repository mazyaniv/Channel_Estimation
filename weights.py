import numpy as np
from matplotlib import pyplot as plt
from function_main import *
from scipy.io import savemat
import os
import math

chosen_space = np.linspace(-5, 20, 50) #dB
sigma_space = 10**(-chosen_space/10)
na,nq = 1,100
bound_sim = 1000

matrix = Matrix(na, nq)
matrix_q = Matrix(0, nq)
matrix_a = Matrix(na, 0)


# LMMSE_a = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, 0) for i in range(len(sigma_space))]
# WBCRB_q = [weighted_new(sigma_space[i], sigma_space[i], na, nq, matrix,bound_sim) for i in range(len(chosen_space))]
# WBCRB_a = [weighted_analog(sigma_space[i], sigma_space[i], na, 0, matrix_a,bound_sim) for i in range(len(chosen_space))]
WBCRB = [weighted_BCRB(sigma_space[i], sigma_space[i], na, nq, matrix, bound_sim) for i in range(len(chosen_space))]
# LMMSE = [MSE_zertothresh_analytic(sigma_space[i], sigma_space[i], na, nq) for i in range(len(sigma_space))]
weights = [weights_func(sigma_space[i], sigma_space[i], na, nq, matrix,500) for i in range(len(sigma_space))]
BCRB = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix, 10000) for i in range(len(chosen_space))]
# probability_vec = [probability(sigma_space[i],na,nq, matrix, bound_sim) for i in range(len(chosen_space))]
# L_App = [probability_vec[i]*WBCRB_a[i]+(1-probability_vec[i])*WBCRB[i] for i in range(len(chosen_space))]

plt.figure(figsize=(10, 6))
# plt.plot(chosen_space, LMMSE_a, label ='LMMSE_a', linestyle='-', marker='.')
# plt.plot(chosen_space, WBCRB_a, label ='WBCRB_a', linestyle='-', marker='.')
plt.plot(chosen_space, weights, label ='weights', linestyle='-', marker='.')
# plt.plot(chosen_space, LMMSE, label ='LMMSE', linestyle='-', marker='.')
# plt.plot(chosen_space, L_App, label ='L_App', linestyle='-', marker='.')
plt.plot(chosen_space, WBCRB, label ='WBCRB', linestyle='-', marker='.')
plt.plot(chosen_space, BCRB, label ='BCRB', linestyle='-', marker='.')

plt.yscale('log')
plt.legend(loc='lower left', ncol=1)
plt.show()
