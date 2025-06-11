import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from function_intro import samp_teta, Matrix
from function_main import rho_q

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
    return np.nanmean(d_vec, axis=1)

chosen_space = np.linspace(-10, 15, 30)
sigma_space = 10**(-chosen_space/10)
nq = 10
matrixco = Matrix(0, nq)
plt.figure(figsize=(10, 6))
plt.ylabel(r"$d$")
plt.xlabel("SNR (dB)")
# plt.plot(chosen_space, [d(sigma_space[i], matrixco) for i in range(len(chosen_space))], linestyle='--', marker="o", label="d")
plt.plot(chosen_space, [norm.pdf(1/sigma_space[i]) for i in range(len(chosen_space))], linestyle='--', marker="o", label=r"$\phi$")
plt.plot(chosen_space, [d2(sigma_space[i]) for i in range(len(chosen_space))], linestyle='--', marker="o", label="d")
plt.legend()
plt.show()
print(d2(0))