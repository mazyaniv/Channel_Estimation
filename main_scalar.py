import numpy as np
import math
from scipy.stats import norm
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from numpy import linalg
import scipy.integrate
############################################################# Variables
sigma = np.logspace(-1,1,100) #irrelevant for lower sigma
#thresh = 0 #np.linspace(-3,3,200)
M = 1
mu = 0
sigma_teta = math.sqrt(1)*(1/math.sqrt(2))
rho_q = 1
rho_a = 1
n_a = [1,1,2,2]
n_q = [40,100,40,100]
sim = pow(10,2)

x = random.random()
y = math.sqrt(1-pow(x,2))
G1_normal = x+1j*y
G_mat = math.sqrt(rho_q)*G1_normal
##M>1
#G1_normal = unitary_group.rvs(M)
#G1 = math.sqrt(rho_q)*G1_normal
# ones = np.ones(n_q)
# G = np.kron(ones, G1).transpose()
x1 = random.random()
y1 = math.sqrt(1-pow(x1,2))
H_mat = math.sqrt(rho_a)*(x1+1j*y1)

real_teta = np.random.normal(mu, sigma_teta)
im_teta = np.random.normal(mu, sigma_teta)
teta = real_teta + 1j*im_teta
############################################################# Matrices
def Matrix(n_a,n_q,H,G):
    H_Mat = np.zeros((n_a,1), complex)
    H_Mat[0,0] = H
    x2 = np.random.rand(n_a-1,1)
    y2 = np.sqrt(1-np.power(x2,2))
    H_mat2 = math.sqrt(rho_a) * (x2 + 1j * y2)
    H_Mat[1:,] = H_mat2.reshape(n_a-1,1)

    ones_vec = np.ones(n_q)
    G_Mat = np.multiply(ones_vec, G)
    G_Mat = G_Mat.reshape(n_q, 1)
    return H_Mat, G_Mat
############################################################# Threshold
def new_thresh_naive(n_q, observ):
    real = np.random.normal(mu, sigma_teta, observ)
    im = np.random.normal(mu, sigma_teta, observ)
    teta_hat = real + 1j * im
    g_teta = G_mat * teta_hat
    r_hold = g_teta.real
    im_hold = g_teta.imag
    thresh_re = np.mean(r_hold)
    thresh_im = np.mean(im_hold)
    N_q = M * n_q
    threshold_real = thresh_re*np.ones(N_q)
    threshold_im = thresh_im*np.ones(N_q)
    return threshold_real.reshape(n_q,1), threshold_im.reshape(n_q,1)

def new_thresh(n_q,observ): #random
    real = np.random.normal(mu, sigma_teta,(n_q,observ))
    im = np.random.normal(mu, sigma_teta,(n_q,observ))
    teta_hat = real + 1j*im
    g_teta = G_mat * teta_hat
    r_hold = g_teta.real
    im_hold = g_teta.imag
    return np.mean(r_hold,1).reshape(n_q,1), np.mean(im_hold,1).reshape(n_q,1)

yaniv = new_thresh(3,10)
############################################################# Estimation
def x(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive):
    sigma_w_a = sigma * (1 / math.sqrt(2))
    real_w_a = np.random.normal(mu, sigma_w_a, n_a)
    im_w_a = np.random.normal(mu, sigma_w_a, n_a)
    w_a = real_w_a + 1j * im_w_a
    w_a = w_a.reshape(n_a, 1)

    sigma_w_q = sigma * (1 / math.sqrt(2))
    real_w_q = np.random.normal(mu, sigma_w_q, n_q)
    im_w_q = np.random.normal(mu, sigma_w_q, n_q)
    w_q = real_w_q + 1j * im_w_q
    w_q = w_q.reshape(n_q, 1)

    x_a = matrix[0] * teta + w_a
    y = matrix[1] * teta + w_q

    if naive == 1:
        x_q = (1 / math.sqrt(2)) * (np.sign(y.real - (thresh_re*np.ones(n_q)).reshape(n_q, 1)) +
                                    1j * np.sign(y.imag - (thresh_im*np.ones(n_q)).reshape(n_q, 1)))
    elif naive ==0:
        x_q = (1 / math.sqrt(2)) * (np.sign(y.real - new_thresh(n_q, observ)[0]) + 1j * np.sign(y.imag - new_thresh(n_q, observ)[1]))
    else:
        x_q = (1 / math.sqrt(2)) * (np.sign(y.real - new_thresh_naive(n_q, observ)[0]) + 1j * np.sign(y.imag - new_thresh_naive(n_q, observ)[1]))

    return x_a.reshape(n_a, ), x_q.reshape(n_q, )


def samp(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive): #samples
    sigma_teta_samp = (1/math.sqrt(2))
    real_teta_samp = np.random.normal(mu, sigma_teta_samp, observ)
    im_teta_samp = np.random.normal(mu, sigma_teta_samp, observ)
    teta_samp = real_teta_samp + 1j*im_teta_samp

    sigma_w_a_samp = sigma * (1 / math.sqrt(2))
    real_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (n_a,observ))
    im_w_a_samp = np.random.normal(mu, sigma_w_a_samp, (n_a,observ))
    w_a_samp = real_w_a_samp + 1j * im_w_a_samp

    sigma_w_q_samp = sigma * (1 / math.sqrt(2))
    real_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(n_q,observ))
    im_w_q_samp = np.random.normal(mu, sigma_w_q_samp,(n_q,observ))
    w_q_samp = real_w_q_samp + 1j * im_w_q_samp

    x_a_samp = (matrix[0]*teta_samp) + w_a_samp
    y_samp = (matrix[1]*teta_samp) + w_q_samp

    if naive == 1:
        x_q_samp = (1 / math.sqrt(2)) * (np.sign(y_samp.real - (thresh_re * np.ones(n_q)).reshape(n_q, 1)) +
                                    1j * np.sign(y_samp.imag - (thresh_im * np.ones(n_q)).reshape(n_q, 1)))
    elif naive == 0:
        x_q_samp = (1 / math.sqrt(2)) * (
                    np.sign(y_samp.real - new_thresh(n_q, observ)[0]) + 1j * np.sign(y_samp.imag - new_thresh(n_q, observ)[1]))
    else:
        x_q_samp = (1 / math.sqrt(2)) * (np.sign(y_samp.real - new_thresh_naive(n_q, observ)[0]) + 1j * np.sign(
            y_samp.imag - new_thresh_naive(n_q, observ)[1]))

    return x_a_samp, x_q_samp, teta_samp.reshape(1,observ)

def samp_teta(observ): #samples
    sigma_teta_samp = (1/math.sqrt(2))
    real_teta_samp = np.random.normal(mu, sigma_teta_samp, observ)
    im_teta_samp = np.random.normal(mu, sigma_teta_samp, observ)
    teta_samp = real_teta_samp + 1j*im_teta_samp
    return  teta_samp.reshape(1,observ)

def covariance(v1,v2):
    normv1 = np.mean(v1,1)
    normv2 = np.mean(v2,1)
    v = v1- normv1.reshape(np.shape(v1)[0],1)
    u = v2 - normv2.reshape(np.shape(v2)[0],1)

    result = [np.matmul(np.array(v.transpose()[i]).reshape(np.shape(v)[0], 1),
                        np.array(u.transpose()[i]).reshape(1, np.shape(u)[0])) for i in range(np.shape(v)[1])]
    return np.mean(result,0)
######################
def L_Estimator_0thresh(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive=2): #0 threshold
    MSE = np.zeros(sim)
    for i in range(sim):
        x_a, x_q = x(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive)
        alpha = (2/math.pi)*math.acos(rho_q/(rho_q+pow(sigma, 2)))
        C = ((2*rho_q*rho_a*n_a)/(math.pi*(rho_q+pow(sigma, 2))*(rho_a*n_a+pow(sigma, 2))))
        w_q = (pow(sigma, 2)/(pow(sigma, 2)+(rho_a*n_a)))*((alpha+(1-alpha)*n_q)/(alpha+((1-alpha-C)*n_q)))
        w_a = 1-((2*rho_q*n_q*w_q)/(math.pi*(rho_q+pow(sigma, 2))*(alpha+(1-alpha)*n_q)))
        teta_a_hat = (1/((rho_a*n_a)+pow(sigma, 2)))*(matrix[0].conjugate().T.dot(x_a))
        teta_q_hat = math.sqrt(2/(math.pi*(rho_q+pow(sigma, 2))))*(1/(alpha+((1-alpha)*n_q)))*(matrix[1].conjugate().T.dot(x_q))
        teta_hat = (w_q*teta_q_hat)+(w_a*teta_a_hat)
        MSE[i] = pow(teta.real-teta_hat.real, 2)+pow(teta.imag-teta_hat.imag, 2)
    return np.mean(MSE)
######################
def L_Estimator(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive=2):
    MSE = np.zeros(observ)
    for i in range(observ):
        x_a, x_q = x(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive)
        x_a_vec, x_q_vec, teta_vec = samp(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive)
        mat1 = np.concatenate((covariance(x_a_vec,x_a_vec), covariance(x_a_vec,x_q_vec)),axis=1)
        mat2 = np.concatenate((covariance(x_q_vec,x_a_vec), covariance(x_q_vec,x_q_vec)),axis=1)
        cov_x = np.concatenate((mat1, mat2),axis=0)
        cov_teta_x = np.concatenate((covariance(teta_vec,x_a_vec), covariance(teta_vec,x_q_vec)),axis=1)
        cov_x_inv =linalg.pinv(cov_x)

        mu_teta = np.mean(teta_vec, 1)
        mu_xa = np.mean(x_a_vec, 1)
        mu_xq = np.mean(x_q_vec, 1)
        x_a_vec_norm = x_a #- H_mat*mu_teta  # H*teta?
        x_q_vec_norm = x_q #- mu_xq  # p(X_q>thresh)-(p(X_q<thresh))
        x_vec_norm = np.concatenate((x_a_vec_norm, x_q_vec_norm), axis=0)
        teta_hat = mu + cov_teta_x.dot(cov_x_inv.dot(x_vec_norm.transpose()))
        MSE[i] = pow(teta.real-teta_hat.real, 2)+pow(teta.imag-teta_hat.imag, 2)
    return np.mean(MSE)
############################################################# Bounds
def CRB(sigma, n_a, n_q, matrix, thresh_re,thresh_im,observ, naive=2): #M=1
     teta_samp = samp_teta(observ)
     teta_samp = teta_samp.transpose()
     g_teta = np.matmul(matrix[1], teta_samp.transpose())

     if naive == 1:
         zeta_real = (math.sqrt(2) * (g_teta.real - (thresh_re*np.ones(n_q)).reshape(n_q, 1))) / sigma  # same thresh_re )
         zeta_im = (math.sqrt(2) * (g_teta.imag - (thresh_im*np.ones(n_q)).reshape(n_q, 1))) / sigma  # same thresh_im
     elif naive == 0:
          zeta_real = (math.sqrt(2) * (g_teta.real - new_thresh(n_q,observ)[0])) / sigma
          zeta_im = (math.sqrt(2) * (g_teta.imag - new_thresh(n_q,observ)[1])) / sigma
     else:
         zeta_real = (math.sqrt(2) * (g_teta.real - (new_thresh_naive(n_q, observ)[0]).reshape(n_q, 1))) / sigma  # same thresh_re )
         zeta_im = (math.sqrt(2) * (g_teta.imag - (new_thresh_naive(n_q, observ)[1]).reshape(n_q, 1))) / sigma  # same thresh_im

     pdf_real = norm.pdf(zeta_real)
     cdf_real = norm.cdf(zeta_real)
     pdf_im = norm.pdf(zeta_im)
     cdf_im = norm.cdf(zeta_im)

     d_vec = np.divide(pow(pdf_real, 2),np.multiply(cdf_real,(norm.cdf(-zeta_real)))) + np.divide(pow(pdf_im, 2),np.multiply(cdf_im,(norm.cdf(-zeta_im))))
     d = np.mean(d_vec)
     # #Integral:
     # def p(real_teta,im_teta):
     #      return (1/(np.pi**M * 1)) * np.exp(-np.dot(np.dot(np.conj(real_teta + 1j*im_teta-mu).T, 1), (real_teta + 1j*im_teta-mu))) #covariance matrix = 1
     # d[k] = scipy.integrate.nquad(lambda real_teta, im_teta: p(real_teta,im_teta)*argu(real_teta,im_teta), [[-5, 5], [-5, 5]], full_output=True)[0]
     nume = (2*pow(sigma,2)*pow(sigma,2))*(G1_normal.conjugate()*G1_normal)
     dev = (2*pow(sigma,2)*rho_q*n_a)+(2*pow(sigma,2)*pow(sigma,2))+(rho_q*n_q*d*pow(sigma,2)) #d[k],M=1
     return (nume/dev).real #Im=0, if M>1, sum
######################
def tight(sigma, n_a, n_q, matrix, thresh_re,thresh_im, observ,naive=2):
    xa_samp , xq_samp, teta_samp = samp(sigma, n_a,n_q, matrix, thresh_re,thresh_im, observ,naive)
    teta_samp = teta_samp.transpose()
    xq_samp = xq_samp.transpose()
    xa_samp = xa_samp.transpose()
    expected = np.zeros(observ)
    for j in range(observ):
        l_teta_vec = (-1*(teta_samp-(mu*np.ones(observ)).reshape(observ,1))).conjugate() #itay- no .conjugate()
        l_teta_vec = l_teta_vec.reshape(1,observ)

        l_xa_vec = ((1/pow(sigma,2))*(xa_samp[j].reshape(n_a,1)-np.matmul(matrix[0],teta_samp.transpose())).conjugate()).transpose() @ matrix[0]
        l_xa_vec = l_xa_vec.reshape(1,observ)

        g_teta = np.matmul(matrix[1],teta_samp.transpose())
        if naive == 1:
            zeta_real = (math.sqrt(2) * (g_teta.real - (thresh_re * np.ones(n_q)).reshape(n_q, 1))) / sigma  # same thresh_re )
            zeta_im = (math.sqrt(2) * (g_teta.imag - (thresh_im * np.ones(n_q)).reshape(n_q, 1))) / sigma  # same thresh_im
        elif naive == 0:
            zeta_real = (math.sqrt(2) * (g_teta.real - new_thresh(n_q, observ)[0])) / sigma
            zeta_im = (math.sqrt(2) * (g_teta.imag - new_thresh(n_q, observ)[1])) / sigma
        else:
            zeta_real = (math.sqrt(2) * (g_teta.real - (new_thresh_naive(n_q, observ)[0]).reshape(n_q, 1))) / sigma  # same thresh_re )
            zeta_im = (math.sqrt(2) * (g_teta.imag - (new_thresh_naive(n_q, observ)[1]).reshape(n_q, 1))) / sigma  # same thresh_im

        # zeta_real = (math.sqrt(2)*(g_teta.real-(threshold(thresh_re, n_q)).reshape(n_q,M)))/sigma
        # zeta_im = (math.sqrt(2)*(g_teta.imag-(threshold(thresh_im, n_q)).reshape(n_q,M)))/sigma

        pdf_real = norm.pdf(zeta_real)
        cdf_real = norm.cdf(zeta_real)
        pdf_im = norm.pdf(zeta_im)
        cdf_im = norm.cdf(zeta_im)

        a_real = (2*pdf_real-1-math.sqrt(2)*((xq_samp[0]).real).reshape(n_q,1))
        b_real = np.divide(pdf_real,np.multiply(cdf_real,1-cdf_real))
        real_part = np.multiply(a_real,b_real)

        a_im = (2 * pdf_im - 1 - math.sqrt(2) * ((xq_samp[0]).imag).reshape(n_q, 1))
        b_im = np.divide(pdf_im, np.multiply(cdf_im, 1 - cdf_im))
        im_part = np.multiply(a_im, b_im)
        l_xq1 = real_part - 1j * im_part
        l_xq_vec = np.sum(l_xq1,0)
        l_xq_vec = l_xq_vec.reshape(1,observ)

        expected_1 = pow(np.absolute(l_teta_vec+l_xa_vec+l_xq_vec),2) #elementwise
        expected1_final = np.mean(expected_1)
        expected[j] = 1/expected1_final
    return np.mean(expected)
############################################################# Plot
matrix_const1 = Matrix(n_a[0],n_q[0],H_mat,G_mat)
matrix_const2 = Matrix(n_a[0],n_q[1],H_mat,G_mat)
matrix_const3 = Matrix(n_a[1],n_q[0],H_mat,G_mat)
matrix_const4 = Matrix(n_a[1],n_q[1],H_mat,G_mat)

L_Estimator_zerothresh1 = [L_Estimator_0thresh(sigma[i], n_a[0],n_q[0], matrix_const1, 0,0, sim) for i in range(len(sigma))]
L_Estimator_zerothresh2 = [L_Estimator_0thresh(sigma[i], n_a[0], n_q[1], matrix_const2,0,0, sim) for i in range(len(sigma))]
L_Estimator_zerothresh3 = [L_Estimator_0thresh(sigma[i], n_a[1], n_q[0], matrix_const3,0,0, sim) for i in range(len(sigma))]
L_Estimator_zerothresh4 = [L_Estimator_0thresh(sigma[i], n_a[1], n_q[1], matrix_const4,0,0, sim) for i in range(len(sigma))]

#L_Estimator = [L_Estimator(sigma[i], n_a[0],n_q[0], matrix_const1, G_mat*mu,G_mat*mu, sim,0) for i in range(len(sigma))]

#CRB1 = [CRB(sigma[i], n_a[0], n_q[0],matrix_const1,G_mat.real*mu,G_mat.imag*mu,sim,1) for i in range(len(sigma))]
CRB2 = [CRB(sigma[i],n_a[0],n_q[0],matrix_const1,0,0,sim,0) for i in range(len(sigma))]

#CRB3 = [CRB(sigma[i], n_a[0], n_q[1],matrix_const2,mu,mu,sim,1) for i in range(len(sigma))]
CRB4 = [CRB(sigma[i],n_a[0],n_q[1],matrix_const2,0,0,sim) for i in range(len(sigma))]

#CRB5 = [CRB(sigma[i], n_a[1], n_q[0],matrix_const3,mu,mu,sim,1) for i in range(len(sigma))]
CRB6 = [CRB(sigma[i],n_a[1],n_q[0],matrix_const3,0,0,sim) for i in range(len(sigma))]

#CRB7 = [CRB(sigma[i], n_a[1], n_q[1],matrix_const4,mu,mu,sim,1) for i in range(len(sigma))]
CRB8 = [CRB(sigma[i],n_a[1],n_q[1],matrix_const4,0,0,sim) for i in range(len(sigma))]

#tight1 = [tight(sigma[i], n_a[0],n_q[0], matrix_const1, mu,mu, sim,0) for i in range(len(sigma))]
######################
list_of_functions = [CRB2,CRB4,CRB6,CRB8]
list_of_colors = ['red','blue','pink','black']
for i in range(len(list_of_functions)):
    plt.plot(10*np.log10(1/sigma), list_of_functions[i], color=list_of_colors[i], label='n_a={}, n_q = {}, CRB'.format(n_a[i], n_q[i]))

list_of_functions = [L_Estimator_zerothresh1,L_Estimator_zerothresh2,L_Estimator_zerothresh3,L_Estimator_zerothresh4]
for i in range(len(list_of_functions)):
    plt.plot(10*np.log10(1/sigma), list_of_functions[i], color=list_of_colors[i], label='n_a={}, n_q = {}, LMMSE_zero'.format(n_a[i], n_q[i]))

# plt.plot(10*np.log10(1/sigma), L_Estimator_zerothresh3, color='pink', label='n_a={}, n_q = {}, LMMSE'.format(n_a[1], n_q[0]))
# plt.plot(10*np.log10(1/sigma), L_Estimator_zerothresh4, color='black', label='n_a={}, n_q = {}, LMMSE'.format(n_a[1], n_q[1]))

#plt.plot(10*np.log10(1/sigma), L_Estimator, color='blue', label='n_a={}, n_q = {}, LMMSE_numerical_not naive'.format(n_a[0], n_q[0]))

#plt.plot(10*np.log10(1/sigma), CRB1, color='red',marker = ",", label="'n_a={}, n_q = {}, CRB naive".format(n_a[0], n_q[0]))
#plt.plot(10*np.log10(1/sigma), CRB2, color='red',marker = "v", label="'n_a={}, n_q = {}, CRB random".format(n_a[0], n_q[0]))
#plt.plot(10*np.log10(1/sigma), CRB12, color='red',marker = "_", label="'n_a={}, n_q = {}, random-const thresh".format(n_a[0], n_q[0]))

# plt.plot(10*np.log10(1/sigma), CRB3, color='blue',marker = ",", label="'n_a={}, n_q = {}, hold=naive".format(n_a[0], n_q[1]))
# plt.plot(10*np.log10(1/sigma), CRB4, color='blue',marker = "v", label="'n_a={}, n_q = {}, hold=random".format(n_a[0], n_q[1]))
#
# plt.plot(10*np.log10(1/sigma), CRB5, color='pink',marker = ",", label="'n_a={}, n_q = {}, hold=naive".format(n_a[1], n_q[0]))
# plt.plot(10*np.log10(1/sigma), CRB6, color='pink',marker = "v", label="'n_a={}, n_q = {}, hold=random".format(n_a[1], n_q[0]))
#
# plt.plot(10*np.log10(1/sigma), CRB7, color='black',marker = ",", label="'n_a={}, n_q = {}, hold=naive".format(n_a[1], n_q[1]))
# plt.plot(10*np.log10(1/sigma), CRB8, color='black',marker = "v", label="'n_a={}, n_q = {}, hold=random".format(n_a[1], n_q[1]))

#plt.plot(10*np.log10(1/sigma), tight1, color='black',marker = ".", label="'n_a={}, n_q = {}, tight".format(n_a[0], n_q[0]))

plt.title("CRB with threshold- mu={}, variance ={}".format(mu,2*pow(sigma_teta,2)))
plt.yscale('log')
plt.xlabel("SNR [dB]")
plt.ylabel("BCRB VS L-MSE")
plt.legend()
plt.legend()
plt.show()

# list_of_outputs = [L_Estimator_zerothresh1,CRB2,L_Estimator,tight1]
# names_of_outputs = ['L_Estimator_zerothresh1','CRB2','L_Estimator','tight1']
# for i in range(len(list_of_outputs)):
#   np.save("{}.npy".format(names_of_outputs[i]), list_of_outputs[i])