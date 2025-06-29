import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
from function_intro import samp_teta, Matrix
from function_main import rho_q, P_xq, probability, CRB


def d2(x):
    zeta_real = (1/x)
    zeta_im = (1/x)
    pdf_real = norm.pdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    return np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
            np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
def d(sigma, matrix, observ=1000, thresh_real=0, thresh_im=0):
    teta_samp = samp_teta(observ)
    g_teta = matrix[1] @ teta_samp
    zeta_real = ((math.sqrt(2) / sigma) * (g_teta.real - thresh_real))
    zeta_im = ((math.sqrt(2) / sigma) * (g_teta.imag - thresh_im))
    pdf_real = norm.pdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    d_vec = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
            np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
    return np.nanmean(d_vec, axis=1)*(1/(2*sigma**2))

chosen_space = np.linspace(40, 100, 25) #dB
sigma_space = 10**(-chosen_space/10)

# theta_range = np.linspace(-3,3, 100)
na, nq = 0,10
matrix = Matrix(0, nq)
BCRB = [CRB(sigma_space[i], sigma_space[i], na, nq, matrix, 1000) for i in range(len(chosen_space))]
print(BCRB)
# P = [P_xq(theta_range[i], -(np.ones(nq)+1j*np.ones(nq))/math.sqrt(2),0.05) for i in range(len(theta_range))]
# probability_vec = [probability(sigma_space[i],na,nq, matrix, 3000) for i in range(len(chosen_space))]
# area = simps(P, theta_range)
# print(f"Area under the curve: {area}")

plt.figure(figsize=(10, 6))
# plt.plot(theta_range, P, label='P_xq', linestyle='-', marker='.')
plt.plot(chosen_space, BCRB, linestyle='-', marker='.')
# plt.xlabel('Theta')
# plt.ylabel('P_xq')
plt.title('vdsvsdvs')
# plt.legend()
plt.grid()
plt.show()