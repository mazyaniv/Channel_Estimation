import numpy as np
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.stats import unitary_group
import random
from mpl_toolkits import mplot3d
import scipy.integrate as spi

sim = 1000
sigma_space = np.logspace(-1,1,20)
M =1
mu = 0
rho_q = 1
rho_a = 1
########Teta
sigma_teta = (1/math.sqrt(2))
real_teta = np.random.normal(mu, sigma_teta,M)
im_teta = np.random.normal(mu, sigma_teta,M)
teta = real_teta + 1j*im_teta
teta = teta.reshape(M,1)

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

def thresh_G(n_q, Mat):
    if M>1:
        G_teta=Mat@((mu+1j*mu)*np.ones(M))
    else:
        G_teta=Mat*((mu+1j*mu)*np.ones(M))
    return G_teta.real.reshape(M*n_q, 1), G_teta.imag.reshape(M*n_q, 1)

def x(sigma1,sigma2, n_a,n_q, matrix,thresh_real=0,thresh_im=0): #the observations- function of teta
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
        x_a = matrix[0]@teta+w_a
        y = matrix[1]@teta+w_q
    else:
        x_a = matrix[0]*teta+w_a
        y = matrix[1]*teta+w_q

    x_q = (1/math.sqrt(2))*(np.sign(y.real-(thresh_real))+1j*np.sign(y.imag-((thresh_im))))
    return x_a.reshape(M*n_a,), x_q.reshape(M*n_q,)

def samp(sigma1,sigma2, n_a,n_q, matrix, observ,thresh_real,thresh_im): #samples
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

    x_q_samp = (1/math.sqrt(2))*(np.sign(y_samp.real-(thresh_real)+1j * np.sign(y_samp.imag-((thresh_im)))))
    return x_a_samp, x_q_samp, teta_samp

def samp_teta(observ): #samples-for CRB function (d_k)
    real_teta_samp = np.random.normal(mu, sigma_teta, (M, observ))
    im_teta_samp = np.random.normal(mu, sigma_teta, (M, observ))
    teta_samp = real_teta_samp + 1j * im_teta_samp
    return  teta_samp.reshape(M,observ)

def covariance(v1,v2):
    normv1 = np.mean(v1,1)
    normv2 = np.mean(v2,1)
    v = v1-normv1.reshape(np.shape(v1)[0],1)
    u = v2 -normv2.reshape(np.shape(v2)[0],1)
    result = [v[:,i].reshape(np.shape(v)[0], 1)@u[:,i].transpose().reshape(1, np.shape(u)[0]) for i in range(np.shape(v)[1])]
    return np.sum(result,0)/(np.shape(v)[1]-1)

def MSE_general_numerical(sigma1,sigma2, n_a,n_q, matrix, observ, epsilon, thresh_real = 0,thresh_im = 0):
    cov = np.zeros((observ, M, M))
    for i in range(observ):
        x_a_vec, x_q_vec, teta_vec = samp(sigma1,sigma2, n_a,n_q, matrix, observ,thresh_real,thresh_im)
        inv_A = (np.identity(M*n_a)-((1/(rho_a*n_a+pow(sigma1,2)))*(matrix[0]@matrix[0].transpose().conjugate())))\
                /pow(sigma1,2) #Na X Na
        B = covariance(x_a_vec,x_q_vec) #Na X Nq
        C = B.conjugate().transpose() #Nq X Na, B.conjugate().transpose()
        D = covariance(x_q_vec,x_q_vec) #Nq X Nq
        K = D-C@inv_A@B

        U,S,V = np.linalg.svd(K)
        S[S<epsilon] = epsilon
        K = U@np.diag(S)@V #K+epsilon*np.identity(K.shape[0])

        inv_K = LA.inv(K)
        cov_x_inv_up =np.concatenate((inv_A+(inv_A@B@inv_K@C@inv_A), -1*(inv_A@B@inv_K)),axis=1)
        cov_x_inv_down = np.concatenate((-1*(inv_K@C@inv_A), inv_K),axis=1)
        cov_x_inv = np.concatenate((cov_x_inv_up,cov_x_inv_down),axis=0)

        mu_tilda_real = mu*(np.sum(matrix[1].real,axis=1)-np.sum(matrix[1].imag,axis=1))
        mu_tilda_imag = mu*(np.sum(matrix[1].real,axis=1)+np.sum(matrix[1].imag,axis=1))
        sigma_tilda = 1+(np.sum(np.power(matrix[1].real,2),axis=1)+np.sum(np.power(matrix[1].imag,2),axis=1))
        p1 = norm.cdf(np.divide(np.subtract(thresh_real,mu_tilda_real.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))
        p2 = norm.cdf(np.divide(np.subtract(thresh_im,mu_tilda_imag.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))

        cov_teta_x = np.concatenate((matrix[0].transpose().conjugate(), covariance(teta_vec,x_q_vec)),axis=1)
        x_a, x_q = x(sigma1,sigma2, n_a, n_q, matrix,thresh_real,thresh_im) #the actually observations
        x_a_vec_norm = x_a-matrix[0]@((mu+1j*mu)*np.ones(M))#np.mean(x_a)*np.ones((np.shape(x_a)))
        x_q_vec_norm = x_q.reshape(M*n_q,1)-math.sqrt(2)*((1-2*p1)+1j*(1-2*p2))#np.mean(x_q)*np.ones((M*n_q,1))
        x_vec_norm = np.concatenate((x_a_vec_norm, x_q_vec_norm.reshape(M*n_q,)), axis=0)
        teta_hat = (cov_teta_x@cov_x_inv@x_vec_norm) +(0+1j*0)*np.ones(M)
        cov[i,:,:] = ((teta_hat-teta)@((teta_hat-teta).conjugate().T)).real #m>1, real number
        #cov[i,:,:] = ((teta_hat-teta)*(teta_hat-teta).conjugate()).real #M=1, real number
    cov_matrix = np.sum(cov,0)/(np.shape(cov)[0])
    return LA.norm(cov_matrix, "fro") #M>1 #np.squeeze(cov_matrix) #M=1 ,

def MSE_zertothresh_analytic(sigma1,sigma2, n_a,n_q):
    alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma2, 2)))
    beta = ((1-alpha)/rho_q)-((2*rho_a*n_a)/(math.pi*(rho_q+pow(sigma2,2))*(rho_a*n_a+pow(sigma1,2))))
    first = (rho_a*n_a)/(rho_a*n_a+pow(sigma1, 2))
    second = (2*rho_q*n_q*pow(sigma1,4))/(math.pi*(rho_q+pow(sigma2, 2))*(alpha+beta*rho_q*n_q)*pow(rho_a*n_a+pow(sigma1, 2),2))
    return math.sqrt(M)*(1-first-second) #Frobenius  norm- not MSE, but relevant for CRB



def CRB(sigma1,sigma2, n_a,n_q,matrix, observ=sim,thresh_real=0,thresh_im=0,quantize=1):
    teta_samp = samp_teta(observ)
    g_teta = matrix[1] @ teta_samp
    G_normal = matrix[1]/math.sqrt(n_q*rho_q)
    zeta_real = ((math.sqrt(2)/sigma2)*(g_teta.real-thresh_real))
    zeta_im = ((math.sqrt(2)/sigma2)*(g_teta.imag-thresh_im))
    pdf_real = norm.pdf(zeta_real)
    cdf_real = norm.cdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    cdf_im = norm.cdf(zeta_im)
    # d_vec = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
    #         np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
    d_vec = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
            np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
    d = np.mean(d_vec, axis=1) #converges to 0.95 aprox.
    if quantize == 0:
        J2 = (rho_q * n_q / pow(sigma2, 2)) * np.identity(M)
    else:
        my_vector = [(n_q*rho_q*d[i])*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
        J2 = np.sum(my_vector,axis=0)*(1/(2*pow(sigma2, 2)))
    J1 = (1 + (rho_a * n_a / pow(sigma1, 2))) * np.identity(M)
    J = J1 + J2
    return LA.norm(J, "fro")

sim = 5
na = 0 #only quantize
nq = 40
thresh_space = np.linspace(-5,5,20)
matrix_const1 = Matrix(na,nq)
thresh_real = thresh_G(nq,matrix_const1[1])[0]
thresh_imag = thresh_G(nq,matrix_const1[1])[1]
thresh_const = np.ones((M*nq, 1))
epsilon = 0.001

L_Estimator_numerical = [MSE_general_numerical(1,1, na,nq, matrix_const1, sim,epsilon,thresh_space[i]*thresh_const,thresh_space[i]*thresh_const) for i in range(len(thresh_space))]
CRB = [CRB(1,1,na,nq,matrix_const1,sim,thresh_space[i]*thresh_const,thresh_space[i]*thresh_const,1) for i in range(len(thresh_space))]


# np.save("L_Estimator_numerical", L_Estimator_numerical)
# np.save("CRB", CRB)
