import numpy as np
import math
from scipy.stats import unitary_group
import random

mu = 0
sim = 100
sigma_teta = (1/math.sqrt(2))
M=1

def CDF_app(x):
    a = 0.339
    b = 5.510
    step_plus = np.heaviside(x, 1)
    step_minus = np.heaviside(-x, 1)
    Q_app_plus = (1 / ((1 - a) * x + a * np.sqrt(x**2 + b))) * ((np.exp(-0.5 * x**2)) / math.sqrt(2 * math.pi))
    Q_app_minus = (1 / ((1 - a) * (-x) + a * np.sqrt(x**2 + b))) * ((np.exp(-0.5 * x**2)) / math.sqrt(2 * math.pi))
    result = step_minus * (1 - Q_app_minus) + step_plus * Q_app_plus
    return 1-result

def Matrix(na,nq,rho_a=1):
    rho_q = rho_a
    H_mat = np.zeros((na*M,M), complex)
    G_mat = np.zeros((nq*M,M), complex)
    x1 = random.random()
    y1 = math.sqrt(1 - pow(x1, 2))
    x2 = random.random()
    y2 = np.sqrt(1 - np.power(x2, 2))
    if M>1:
        H_1 = math.sqrt(rho_a)*unitary_group.rvs(M)
        G_1 = math.sqrt(rho_q)*unitary_group.rvs(M)
        for i in range(0,na*M,M):
            H_mat[i:M+i,:] = H_1
        for i in range(0,nq*M,M):
            G_mat[i:M+i,:] = G_1
    else: #M=1
        for i in range(0,na*M,M):
            H_mat[i:M + i, :] = math.sqrt(rho_a)*(x1+1j*y1)
        for i in range(0,nq):
            G_mat[i:1+ i, :] = math.sqrt(rho_q)*(x2 + 1j * y2)
    return H_mat, G_mat

def thresh_G(n_q, Mat):
    if M>1:
        G_teta=Mat@((mu+1j*mu)*np.ones(M))
    else:
        G_teta=Mat*((mu+1j*mu)*np.ones(M))
    return G_teta.real.reshape(M*n_q, 1), G_teta.imag.reshape(M*n_q, 1)

def x(sigma1,sigma2, n_a,n_q, matrix,theta,thresh_real=0,thresh_im=0): #the observations- function of teta
    theta = theta.reshape(M, 1)
    sigma_w_a = sigma1 * (1 / math.sqrt(2))
    real_w_a = np.random.normal(mu, sigma_w_a, M*n_a)
    im_w_a = np.random.normal(mu, sigma_w_a, M*n_a)
    w_a = real_w_a + 1j * im_w_a
    w_a = w_a.reshape(M*n_a, 1)

    sigma_w_q = sigma2 * (1 / math.sqrt(2))
    real_w_q = np.random.normal(mu, sigma_w_q, M*n_q)
    im_w_q = np.random.normal(mu, sigma_w_q, M*n_q)
    w_q = real_w_q + 1j * im_w_q
    w_q = w_q.reshape(M*n_q, 1)

    if M>1:
        x_a = matrix[0]@theta+w_a
        y = matrix[1]@theta+w_q
    else:
        x_a = matrix[0]*theta+w_a
        y = matrix[1]*theta+w_q

    x_q = (1/math.sqrt(2))*(np.sign(y.real-(thresh_real))+1j*np.sign(y.imag-((thresh_im))))
    return x_a.reshape(M*n_a,), x_q.reshape(M*n_q,)

def samp(sigma1,sigma2, n_a,n_q, matrix, observ,thresh_real=0,thresh_im=0): #samples
    real_teta_samp = np.random.normal(mu, sigma_teta, (M,observ))
    im_teta_samp = np.random.normal(mu, sigma_teta, (M,observ))
    teta_samp = (real_teta_samp + 1j*im_teta_samp)

    sigma_w_a_samp = sigma1 * (1 / math.sqrt(2))
    real_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (M*n_a,observ))
    im_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (M*n_a,observ))
    w_a_samp = (real_w_a_samp + 1j * im_w_a_samp)

    sigma_w_q_samp = sigma2* (1 / math.sqrt(2))
    real_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(M*n_q,observ))
    im_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(M*n_q,observ))
    w_q_samp = (real_w_q_samp + 1j * im_w_q_samp)

    x_a_samp = (matrix[0]@teta_samp)+w_a_samp
    y_samp = (matrix[1]@teta_samp) + w_q_samp

    x_q_samp = (1/math.sqrt(2))*(np.sign(y_samp.real-(thresh_real))+(1j*(np.sign(y_samp.imag-((thresh_im))))))
    return x_a_samp, x_q_samp, teta_samp

def samp_teta(observ):
    real_teta_samp = np.random.normal(mu, sigma_teta, (M, observ))
    im_teta_samp = np.random.normal(mu, sigma_teta, (M, observ))
    teta_samp = real_teta_samp + 1j * im_teta_samp
    return  teta_samp.reshape(M,observ)

def covariance(v1,v2):
    normv1 = np.mean(v1,1)
    normv2 = np.mean(v2,1)
    v = v1-normv1.reshape(np.shape(v1)[0],1)
    u = v2 -normv2.reshape(np.shape(v2)[0],1)
    result = [v[:,i].reshape(np.shape(v)[0], 1)@u[:,i].conjugate().transpose().reshape(1, np.shape(u)[0]) for i in range(np.shape(v)[1])]
    return np.sum(result,0)/(np.shape(v)[1]-1)