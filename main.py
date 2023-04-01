import numpy as np
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.stats import unitary_group
import random
############################################################# Variables
sigma = np.logspace(-1,1,100)
thresh_space = np.linspace(-5,5,100)
M =5
mu = 1
sigma_teta = math.sqrt(1)*(1/math.sqrt(2))
rho_q = 1
rho_a = 1
n_a = [1,1,2,2]
n_q = [10,40,10,40]
sim = pow(10,2)
########Teta
real_teta = np.random.normal(mu, sigma_teta,M)
im_teta = np.random.normal(mu, sigma_teta,M)
teta = real_teta + 1j*im_teta
teta = teta.reshape(M,1)
############################################################# Matrices
def Matrix(na,nq): #new model
    H_mat = np.zeros((na*M,M), complex)
    G_mat = np.zeros((nq*M,M), complex)
    for i in range(0,na*M,M):
        if M>1:
            H_mat[i:M+i,:] = math.sqrt(rho_a)*unitary_group.rvs(M)
        else:
            x1 = random.random()
            y1 = math.sqrt(1 - pow(x1, 2))
            H_mat[i:M + i, :] = math.sqrt(rho_a)*(x1+1j*y1)
    for i in range(0,nq*M,M):
        if M > 1:
            G_mat[i:M+i,:] = math.sqrt(rho_q)*unitary_group.rvs(M)
        else:
            x2 = random.random()
            y2 = np.sqrt(1 - np.power(x2, 2))
            G_mat[i:M + i, :] = math.sqrt(rho_a)*(x2 + 1j * y2)
    return H_mat, G_mat
############################################################# Threshold
def thresh_random(n_q,Mat, observ): #random
    N_q = M*n_q
    real = np.random.normal(mu, sigma_teta, (M,observ))
    im = np.random.normal(mu, sigma_teta, (M,observ))
    teta_hat = real + 1j * im
    G_teta = Mat@teta_hat #G_teta = G_mat @ teta_hat
    r_hold = G_teta.real
    im_hold = G_teta.imag
    thresh_re = np.mean(r_hold,axis = 1)
    thresh_im = np.mean(im_hold,axis = 1)
    threshold_real = thresh_re*np.ones(N_q)
    threshold_im = thresh_im*np.ones(N_q)
    return threshold_real.reshape(N_q,1), threshold_im.reshape(N_q,1)

############################################################# Samples
def x(sigma, n_a,n_q, matrix,observ,naive): #the observations- function of teta
    sigma_w_a = sigma * (1 / math.sqrt(2))
    real_w_a = np.random.normal(mu, sigma_w_a, M*n_a)
    im_w_a = np.random.normal(mu, sigma_w_a, M*n_a)
    w_a = real_w_a + 1j * im_w_a
    w_a = w_a.reshape(M*n_a, 1)

    sigma_w_q = sigma * (1 / math.sqrt(2))
    real_w_q = np.random.normal(mu, sigma_w_q, M*n_q)
    im_w_q = np.random.normal(mu, sigma_w_q, M*n_q)
    w_q = real_w_q + 1j * im_w_q
    w_q = w_q.reshape(M*n_q, 1)

    x_a = matrix[0]@teta + w_a
    y = matrix[1]@teta + w_q
    if naive == 0:
        x_q = (1 / math.sqrt(2)) * (np.sign(y.real - (thresh_random(n_q,M,matrix[1], observ)[0]) +
                                1j * np.sign(y.imag - ((thresh_random(n_q,M,matrix[1], observ)[1])))))
    if naive == 1:
        x_q = (1 / math.sqrt(2)) * (np.sign(y.real)+1j * np.sign(y.imag))
    return x_a.reshape(M*n_a, ), x_q.reshape(M*n_q, )

def samp(sigma, n_a,n_q, matrix, observ,naive): #samples
    sigma_teta_samp = (1/math.sqrt(2))
    real_teta_samp = np.random.normal(mu, sigma_teta_samp, (M,observ))
    im_teta_samp = np.random.normal(mu, sigma_teta_samp, (M,observ))
    teta_samp = real_teta_samp + 1j*im_teta_samp

    sigma_w_a_samp = sigma * (1 / math.sqrt(2))
    real_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (M*n_a,observ))
    im_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (M*n_a,observ))
    w_a_samp = real_w_a_samp + 1j * im_w_a_samp

    sigma_w_q_samp = sigma * (1 / math.sqrt(2))
    real_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(M*n_q,observ))
    im_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(M*n_q,observ))
    w_q_samp = real_w_q_samp + 1j * im_w_q_samp

    x_a_samp = (matrix[0]@teta_samp)+w_a_samp
    y_samp = (matrix[1]@teta_samp) + w_q_samp
    if naive == 0:
        x_q_samp = (1 / math.sqrt(2)) * (np.sign(y_samp.real - (thresh_random(n_q, M, matrix[1], observ)[0]) +
                                            1j * np.sign(y_samp.imag - ((thresh_random(n_q, M, matrix[1], observ)[1])))))
    if naive == 1:
        x_q_samp = (1 / math.sqrt(2)) * (np.sign(y_samp.real)+1j * np.sign(y_samp.imag))
    return x_a_samp, x_q_samp, teta_samp.reshape(M,observ)

def samp_teta(observ): #samples- for CRB function
    sigma_teta_samp = (1 / math.sqrt(2))
    real_teta_samp = np.random.normal(mu, sigma_teta_samp, (M, observ))
    im_teta_samp = np.random.normal(mu, sigma_teta_samp, (M, observ))
    teta_samp = real_teta_samp + 1j * im_teta_samp
    return  teta_samp.reshape(M,observ)

def covariance(v1,v2):  #TODO GPT?
    normv1 = np.mean(v1,1)
    normv2 = np.mean(v2,1)
    v = v1-normv1.reshape(np.shape(v1)[0],1)
    u = v2 -normv2.reshape(np.shape(v2)[0],1)
    result = [v[:,i].reshape(np.shape(v)[0], 1)@u[:,i].transpose().reshape(1, np.shape(u)[0]) for i in range(np.shape(v)[1])]
    return np.mean(result,0)
############################################################# MSE/Frobenius
def MSE_zertothresh_analytic(sigma, n_a,n_q):
    alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma, 2)))
    beta = (1-alpha)/rho_q
    first = (rho_a*n_a)/(rho_a*n_a+pow(sigma, 2))
    second = (2*rho_q*n_q*pow(sigma,4))/(math.pi*(rho_q+pow(sigma, 2))*(alpha+beta*rho_q*n_q)*pow(rho_a*n_a+pow(sigma, 2),2))
    return math.sqrt(M)*(1-first-second) #Frobenius  norm

def MSE_general_numerical(sigma, n_a,n_q, matrix, observ,naive=1): #TODO: simulations, check it
    cov = np.zeros((observ, M, M))
    for i in range(observ):
        x_a_vec, x_q_vec, teta_vec = samp(sigma, n_a,n_q, matrix, observ,naive)
        mat1 = np.concatenate((covariance(x_a_vec,x_a_vec), covariance(x_a_vec,x_q_vec)),axis=1)
        mat2 = np.concatenate((covariance(x_q_vec,x_a_vec), covariance(x_q_vec,x_q_vec)),axis=1)
        cov_x = np.concatenate((mat1, mat2),axis=0)
        cov_teta_x = np.concatenate((covariance(teta_vec,x_a_vec), covariance(teta_vec,x_q_vec)),axis=1)
        cov_x_inv =LA.pinv(cov_x) #TODO check it

        mu_xq = np.mean(x_q_vec, 1)
        x_a, x_q = x(sigma, n_a, n_q, matrix, observ, naive) #the actually observations
        x_a_vec_norm = x_a - matrix[0]@((mu+1j*mu)*np.ones(M))
        x_q_vec_norm = x_q - mu_xq #TODO - analytic expression for mu_xq
        x_vec_norm = np.concatenate((x_a_vec_norm, x_q_vec_norm), axis=0)
        teta_hat = mu*np.ones(M) + (cov_teta_x@cov_x_inv@x_vec_norm)
        cov[i,:,:] = ((teta_hat-teta)@((teta_hat-teta).conjugate().T))
    cov_matrix = np.mean(cov,0)
    return LA.norm(cov_matrix, "fro")

# def MSE_zertothresh_numerical(sigma, n_a, n_q, matrix, observ, naive=1):  # TODO
#     MSE = np.zeros((observ, M, M))
#     for i in range(observ):
#         x_a, x_q = x(sigma, n_a, n_q, matrix, observ, naive)  # function of teta
#         alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma, 2)))
#         C = (2 * rho_q * rho_a * n_a) / (math.pi * (rho_q + pow(sigma, 2)) * (rho_a * n_a + pow(sigma, 2)))
#         w_q = (pow(sigma, 2) / (pow(sigma, 2) + (rho_a * n_a))) * (
#                     (alpha + (1 - alpha) * n_q) / (alpha + ((1 - alpha - C) * n_q)))
#         w_a = 1 - ((2 * rho_q * n_q * w_q) / (math.pi * (rho_q + pow(sigma, 2)) * (alpha + (1 - alpha) * n_q)))
#         teta_a_hat = (1 / ((rho_a * n_a) + pow(sigma, 2))) * ((matrix[0].conjugate().T) @ (x_a))
#         teta_q_hat = math.sqrt(2 / (math.pi * (rho_q + pow(sigma, 2)))) * (1 / (alpha + ((1 - alpha) * n_q))) * (
#                     (matrix[1].conjugate().T) @ (x_q))
#         teta_hat = (w_q * teta_q_hat) + (w_a * teta_a_hat)
#         MSE[i, :, :] = ((teta_hat - teta) @ ((teta_hat - teta).conjugate().T))
#     cov_matrix = np.mean(MSE, 0)
#     return LA.norm(cov_matrix, "fro")
############################################################# Bounds
def thresh_G(Mat,mu):
    N_q = M * n_q
    G_teta = Mat@(mu*np.ones(M))
    return G_teta.real.reshape(N_q, 1), G_teta.imag.reshape(N_q, 1)

def CRB(sigma, n_a, n_q,matrix, observ, thresh, naive=1):
    teta_samp = samp_teta(observ)
    g_teta = matrix[1] @ teta_samp

    if naive==1:
        zeta_real = (math.sqrt(2) / sigma * (g_teta.real-thresh*np.ones((M*n_q,1))))
        zeta_im = (math.sqrt(2) / sigma * (g_teta.imag-thresh*np.ones((M*n_q,1))))
    else:
        G_teta=matrix[1]@((mu+1j*mu)*np.ones(M))
        zeta_real = (math.sqrt(2) / sigma * (g_teta.real - G_teta.real.reshape(M*n_q, 1)))
        zeta_im = (math.sqrt(2) / sigma * (g_teta.imag - G_teta.imag.reshape(M*n_q, 1)))

    pdf_real = norm.pdf(zeta_real)
    cdf_real = norm.cdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    cdf_im = norm.cdf(zeta_im)

    d_vec = np.divide(np.power(pdf_real, 2), np.multiply(cdf_real, (norm.cdf(-zeta_real)))) + \
            np.divide(np.power(pdf_im, 2), np.multiply(cdf_im, (norm.cdf(-zeta_im))))
    d = np.mean(d_vec, axis=1)

    J =(1 + (rho_a * n_a / pow(sigma, 2))) * np.identity(M)+\
       (1 /(2 * pow(sigma, 2)))*(matrix[1].transpose().conjugate() @ np.diag(d) @ matrix[1])
    return LA.norm((LA.inv(J)).real,"fro")
############################################################# Plot
matrix_const1 = Matrix(n_a[0],n_q[0])
matrix_const2 = Matrix(n_a[0],n_q[1])
matrix_const3 = Matrix(n_a[1],n_q[0])
matrix_const4 = Matrix(n_a[1],n_q[1])

#L_Estimator_numerical1 = [MSE_general_numerical(sigma[i], n_a[0],n_q[0], matrix_const1, sim) for i in range(len(sigma))]
# L_Estimator_numerical2 = [MSE_general_numerical(sigma[i], n_a[0],n_q[1], matrix_const2, sim) for i in range(len(sigma))]
# L_Estimator_numerical3 = [MSE_general_numerical(sigma[i], n_a[1],n_q[0], matrix_const3, sim) for i in range(len(sigma))]
# L_Estimator_numerical4 = [MSE_general_numerical(sigma[i], n_a[1],n_q[1], matrix_const4, sim) for i in range(len(sigma))]

L_Estimator_analytic1 = [MSE_zertothresh_analytic(sigma[i], n_a[0],n_q[0]) for i in range(len(sigma))]
L_Estimator_analytic2 = [MSE_zertothresh_analytic(sigma[i], n_a[0],n_q[1]) for i in range(len(sigma))]
L_Estimator_analytic3 = [MSE_zertothresh_analytic(sigma[i], n_a[1],n_q[0]) for i in range(len(sigma))]
L_Estimator_analytic4 = [MSE_zertothresh_analytic(sigma[i], n_a[1],n_q[1]) for i in range(len(sigma))]

CRB2 = [CRB(sigma[i],n_a[0],n_q[0],matrix_const1,sim,mu,1) for i in range(len(thresh_space))]
CRB4 = [CRB(sigma[i],n_a[0],n_q[1],matrix_const2,sim,mu,1) for i in range(len(thresh_space))]
CRB6 = [CRB(sigma[i],n_a[0],n_q[0],matrix_const3,sim,mu,0) for i in range(len(thresh_space))]
CRB8 = [CRB(sigma[i],n_a[0],n_q[1],matrix_const4,sim,mu,0) for i in range(len(thresh_space))]
######################
list_of_colors = ['red','blue','pink','black']

list_of_functions = [CRB2,CRB4,CRB6,CRB8]
for i in range(len(list_of_functions)):
    plt.plot(10*np.log10(1/sigma), list_of_functions[i],"*", color=list_of_colors[i], label='n_a={}, n_q = {}, CRB'.format(n_a[i], n_q[i]))

list_of_functions1 = [L_Estimator_analytic1 ,L_Estimator_analytic2,L_Estimator_analytic3,L_Estimator_analytic4]
for i in range(len(list_of_functions1)):
    plt.plot(10*np.log10(1/sigma), list_of_functions1[i], color=list_of_colors[i], label='n_a={}, n_q = {}, numerical'.format(n_a[i], n_q[i]))

# list_of_functions2 = [L_Estimator_analytic1,L_Estimator_analytic2,L_Estimator_analytic3,L_Estimator_analytic4]
# for i in range(len(list_of_functions2)):
#     plt.plot(10*np.log10(1/sigma), list_of_functions2[i] ,color=list_of_colors[i], label='n_a={}, n_q = {}, analytic'.format(n_a[i], n_q[i]))

plt.title("M={}, mu={}, variance ={}".format(M,mu,2*pow(sigma_teta,2)))
plt.yscale('log')
plt.xlabel("SNR [dB]")
plt.legend()
plt.legend()
plt.show()