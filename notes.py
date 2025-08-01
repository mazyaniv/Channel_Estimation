import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
from function_intro import samp_teta, Matrix
from function_main import *

import numpy as np
from scipy.stats import norm
# def J_q(sigma1, sigma2, n_a, n_q, matrix, observ=sim, thresh_real=0, thresh_im=0):  #BCRB
#     teta_samp = samp_teta(observ)
#     matrix = list(matrix)
#     matrix[1] = np.ones((n_q * M, M))
#     g_teta = matrix[1] @ teta_samp
#     G_normal = matrix[1] / math.sqrt(n_q * rho_q)
#     zeta_real = ((math.sqrt(2) / sigma2) * (g_teta.real - thresh_real))
#     zeta_im = ((math.sqrt(2) / sigma2) * (g_teta.imag - thresh_im))
#     pdf_real = norm.pdf(zeta_real)
#     pdf_im = norm.pdf(zeta_im)
#     d_vec_real = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real))))
#     d_vec_im = np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
#     d_vec = d_vec_real + d_vec_im
#     d = np.nanmean(d_vec, axis=1)
#     my_vector = [(n_q * rho_q * d[i]) * G_normal[i].reshape(M, 1).conjugate() * G_normal[i].reshape(M, 1).transpose()
#                  for i in range(len(d))]
#     J2 = np.sum(my_vector, axis=0) * (1 / (2 * pow(sigma2, 2)))
#     J1 = (1 + (rho_a * n_a / pow(sigma1, 2))) * np.identity(M)
#     J = J1 + J2
#     return LA.norm((LA.inv(J)).real, "fro")  # np.squeeze(J2.real)

chosen_space = np.linspace(-5, 25, 25) #dB
sigma_space = 10**(-chosen_space/10)

# theta_range = np.linspace(-3,3, 100)
na, nq = 0,100
matrix = Matrix(0, nq)
# Jq = [J_q(sigma_space[i], sigma_space[i], na, nq, matrix, 50) for i in range(len(chosen_space))]
# BCRB = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix, 1000) for i in range(len(chosen_space))]
# P = [P_xq(theta_range[i], -(np.ones(nq)+1j*np.ones(nq))/math.sqrt(2),0.05) for i in range(len(theta_range))]
probability_vec = [probability(sigma_space[i],na,nq, matrix, 500) for i in range(len(chosen_space))]
probability_vec2 = [probability_new(sigma_space[i],na,nq, matrix, 500,20) for i in range(len(chosen_space))]
# area = simps(P, theta_range)
# print(f"Area under the curve: {area}")

plt.figure(figsize=(10, 6))
# plt.plot(theta_range, P, label='P_xq', linestyle='-', marker='.')
plt.plot(chosen_space, probability_vec, linestyle='-', marker='.', label='prob_original')
plt.plot(chosen_space, probability_vec2, linestyle='--', marker='x', label='prob_new')
# plt.plot(chosen_space, BCRB, linestyle='-', marker='.', label='BCRB')
# plt.xlabel('Theta')
# plt.ylabel('P_xq')
plt.legend()
plt.grid()
plt.show()
