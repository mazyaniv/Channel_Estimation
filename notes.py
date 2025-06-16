import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
from function_intro import samp_teta, Matrix
from function_main import rho_q, P_xq


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

theta_range = np.linspace(10, 14.5, 100)
nq = 1
matrix = Matrix(0, nq)
P = [P_xq(theta_range[i], (np.ones(nq)+1j*np.ones(nq))/math.sqrt(2),0.05) for i in range(len(theta_range))]

# area = simps(P, theta_range)
# print(f"Area under the curve: {area}")

plt.figure(figsize=(10, 6))
plt.plot(theta_range, P, label='P_xq', linestyle='-', marker='.')
plt.xlabel('Theta')
plt.ylabel('P_xq')
plt.title('P_xq vs Theta')
plt.legend()
plt.grid()
plt.show()