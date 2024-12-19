from scipy.stats import norm
from numpy import linalg as LA
import scipy.integrate as spi
from scipy.integrate import quad
from function_intro import *
rho_a=rho_q=1 #note

############################################################################################################ Basic
def MSE_zertothresh_analytic(sigma1,sigma2, n_a,n_q): #Itay
    alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma2, 2)))
    beta = ((1-alpha)/rho_q)-((2*rho_a*n_a)/(math.pi*(rho_q+pow(sigma2,2))*(rho_a*n_a+pow(sigma1,2))))
    first = (rho_a*n_a)/(rho_a*n_a+pow(sigma1, 2))
    second = (2*rho_q*n_q*pow(sigma1,4))/(math.pi*(rho_q+pow(sigma2, 2))*(alpha+beta*rho_q*n_q)*pow(rho_a*n_a+pow(sigma1, 2),2))
    return M*(1-first-second)

def MSE_general_numerical(sigma1, sigma2, n_a, n_q, matrix, observ, snap=1000, thresh_real=0, thresh_im=0): #Threshold
    cov = np.zeros((observ, M, M),dtype=complex)
    cov_teta_xa = matrix[0].transpose().conjugate()
    x_a_vec, x_q_vec, teta_vec = samp(sigma1, sigma2, n_a, n_q, matrix, snap, thresh_real, thresh_im)
    cov_teta_xq = covariance(teta_vec, x_q_vec)

    A = matrix[0] @ matrix[0].T.conjugate() + sigma1 ** 2 * np.identity(M * n_a)
    inv_A = (np.identity(M * n_a) - (
                (1 / (rho_a * n_a + pow(sigma1, 2))) * (matrix[0] @ matrix[0].transpose().conjugate()))) / pow(sigma1,
                                                                                                               2)  # Na X Na
    B = covariance(x_a_vec, x_q_vec)  # Na X Nq
    C = B.conjugate().transpose()  # Nq X Na, B.conjugate().transpose()
    D = np.cov(x_q_vec)  # covariance(x_q_vec,x_q_vec) #np.cov(x_q_vec) #Nq X Nq
    K = D - C @ inv_A @ B

    # if LA.cond(K) > 1e32: #K is non inverse - big threhold
    U, S, V = np.linalg.svd(K)
    epsilon = np.std(np.diag(S))  # random.uniform(0.01, 15)
    K = K + epsilon * np.identity(K.shape[0])  # U@np.diag(S)@V
    inv_K = LA.inv(K)
    # U,S,V = np.linalg.svd(D)
    # epsilon = np.std(np.diag(S))#random.uniform(0.01, 15)
    # D = D+epsilon*np.identity(D.shape[0])# U@np.diag(S)@V
    # inv_K = LA.inv(D)+LA.inv(D)@C@LA.inv(A-B@LA.inv(D)@C)@B@LA.inv(D)
    # else:
    # while True:
    # U,S,V = np.linalg.svd(K)
    # epsilon = np.std(np.diag(S))#random.uniform(0.01, 1)#np.std(np.diag(S))
    # K = K+epsilon*np.identity(K.shape[0])
    # S = S+epsilon*np.ones(S.shape[0])
    # K = U@np.diag(S)@V#K+epsilon*np.identity(K.shape[0])
    # if float(LA.cond(K)) < float(1.08):
    #     break

    cov_x_inv_up = np.concatenate((inv_A + (inv_A @ B @ inv_K @ C @ inv_A), -1 * (inv_A @ B @ inv_K)), axis=1)
    cov_x_inv_down = np.concatenate((-1 * (inv_K @ C @ inv_A), inv_K), axis=1)
    cov_x_inv = np.concatenate((cov_x_inv_up, cov_x_inv_down), axis=0)
    cov_teta_x = np.concatenate((cov_teta_xa, cov_teta_xq), axis=1)
    for i in range(observ):
        real_teta = np.random.normal(mu, sigma_teta, M)
        im_teta = np.random.normal(mu, sigma_teta, M)
        teta = real_teta + 1j * im_teta
        # mu_tilda_real = mu*(np.sum(matrix[1].real,axis=1)-np.sum(matrix[1].imag,axis=1))
        # mu_tilda_imag = mu*(np.sum(matrix[1].real,axis=1)+np.sum(matrix[1].imag,axis=1))
        # sigma_tilda = 1+(np.sum(np.power(matrix[1].real,2),axis=1)+np.sum(np.power(matrix[1].imag,2),axis=1))
        # p1 = norm.cdf(np.divide(np.subtract(thresh_real,mu_tilda_real.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))
        # p2 = norm.cdf(np.divide(np.subtract(thresh_im,mu_tilda_imag.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))
        p = norm.cdf(thresh_real / (np.sqrt(0.5 * (sigma1 ** 2 + rho_q)))) * np.ones((n_q * M, 1))
        x_e = (1 / math.sqrt(2) )* (1 - 2 * p) * (1 + 1j)

        x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, teta, thresh_real, thresh_im)  # the actually observations
        x_a_vec_norm = x_a - matrix[0]@((mu+1j*mu)*np.ones(M))#np.mean(x_a_vec)
        x_q_vec_norm = x_q.reshape(M * n_q, 1)-x_e*np.ones((n_q*M,1))#np.mean(x_q_vec)#np.mean(x_q_vec)
        x_vec_norm = np.concatenate((x_a_vec_norm, x_q_vec_norm.reshape(M * n_q, )), axis=0)

        teta_hat = (cov_teta_x @ cov_x_inv @ x_vec_norm) + (mu + 1j * mu) * np.ones(M)
        epsilon = (teta_hat - teta).reshape(M, 1)
        cov[i, :, :] = (epsilon @ (epsilon.conjugate().T))
    cov_matrix = np.sum(cov, 0) / (np.shape(cov)[0])
    return LA.norm(cov_matrix, "fro")/math.sqrt(M)
def E_theta_givenx_numeric(sigma1, sigma2, n_a, n_q, matrix, observ, monte=sim, thresh_real=0, thresh_im=0): #MMSE
    MSE = np.zeros((monte, M, M))
    # print("SNR=", 10 * np.log10(1 / sigma1))
    flag = 1
    for j in range(monte):
        x_a, x_q, theta_org = samp(sigma1, sigma2, n_a, n_q, matrix, 1, thresh_real,
                                   thresh_im)  # observations made by original theta
        theta_org = theta_org[0]
        # theta_vec = theta_org #observ=1
        theta_vec = samp_teta(observ)[0]
        result2 = np.zeros((observ), complex)
        result1 = np.zeros((observ), complex)

        for i in range(len(theta_vec)):
            f_xa = (1 / (pow(math.pi, n_a) * pow(sigma1, 2 * n_a))) * math.exp(np.real((-(1 / pow(sigma1, 2)) * (
                np.subtract(x_a, matrix[0] * theta_vec[i])).transpose().conjugate() @ (np.subtract(x_a, matrix[0] *
                                                                                                   theta_vec[i])))))

            zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta_vec[i]).real - thresh_real)
            zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta_vec[i]).imag - thresh_im)

            p_xq = np.prod(np.power(norm.cdf(zeta_real), (0.5 + x_q.real / math.sqrt(2)).reshape(-1, 1))) \
                   * np.prod(np.power(norm.cdf(zeta_im), (0.5 + x_q.imag / math.sqrt(2)).reshape(-1, 1))) * np.prod(
                np.power(norm.cdf(-zeta_real), (0.5 - x_q.real / math.sqrt(2)).reshape(-1, 1))) * np.prod(
                np.power(norm.cdf(-zeta_im), (0.5 - x_q.imag / math.sqrt(2)).reshape(-1, 1)))

            result2[i] = theta_vec[i] * (f_xa * p_xq)
            result1[i] = (f_xa * p_xq)

        teta_hat = np.mean(result2) / np.mean(result1)
        MSE[j, :, :] = (((teta_hat - theta_org) @ ((teta_hat - theta_org).conjugate().T)).real)
        # print("Finish simulation number:", flag)
        flag += 1
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")

def CRB(sigma1,sigma2, n_a,n_q,matrix,observ=sim,quantize=1,thresh_real=0,thresh_im=0): #BCRB
    teta_samp = samp_teta(observ)
    g_teta = matrix[1] @ teta_samp
    G_normal = matrix[1]/math.sqrt(n_q*rho_q)
    zeta_real = ((math.sqrt(2)/sigma2)*(g_teta.real-thresh_real))
    zeta_im = ((math.sqrt(2)/sigma2)*(g_teta.imag-thresh_im))
    pdf_real = norm.pdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    # d_vec = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
    #         np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
    d_vec = np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
            np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im))))
    d = np.mean(d_vec, axis=1) #converges to 0.95 aprox.
    # print(d)
    if quantize == 0:
        J2 = (2/math.pi)*(rho_q * n_q / pow(sigma2, 2))*np.identity(M)
    else:
        my_vector = [(n_q*rho_q*d[i])*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
        J2 = np.sum(my_vector,axis=0)*(1/(2*pow(sigma2, 2)))
    J1 = (1 + (rho_a * n_a / pow(sigma1, 2))) * np.identity(M)
    J = J1+J2
    # my_vector = [((2*pow(sigma1,2)*pow(sigma2,2))+(2*rho_a*n_a*pow(sigma2,2))+n_q*rho_q*d[i]*pow(sigma1,2))*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
    # J_sum = np.sum(my_vector,axis=0)*(1/(2*pow(sigma2, 2)*pow(sigma1,2)))
    # my_vector = [(((2 * pow(sigma2, 2))) /((2*pow(sigma1,2)*pow(sigma2,2))+(2*rho_a*n_a*pow(sigma2,2))+n_q*rho_q*d[i]*pow(sigma1,2)))*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
    # Bound_sum = np.sum(my_vector,axis=0) #Harder to compute
    return LA.norm((LA.inv(J)).real,"fro")#LA.norm((LA.inv(J_sum)).real,"fro"), , LA.norm((LA.inv(J1)).real,"fro"), LA.norm((LA.inv(J2+np.identity(M))).real,"fro")

############################################################################################################ WWB FOR REAL THETA!
def inner(theta,sigma,s,h,thresh_real):
    return pow(norm.cdf((1/sigma)*(theta+h-thresh_real)),s)*\
    pow(norm.cdf((1/sigma)*(theta-thresh_real)),1-s)+\
    pow(norm.cdf(-(1/sigma)*(theta+h-thresh_real)),s)*\
    pow(norm.cdf(-(1/sigma)*(theta-thresh_real)),1-s)
def ratio_func(theta,mu,sigma2,s,h,thresh_real):
    return pow(1/(math.sqrt(2*math.pi))*math.exp(-0.5*((theta-mu+h)**2)),s)*pow(1/(math.sqrt(2*math.pi))*math.exp(-0.5*((theta-mu)**2)),1-s)*inner(theta,sigma2,s,h,thresh_real)
def etha(mu,sigma2,s,h,thresh_real):
    expected_value, _ = quad(ratio_func,-15, 15, args=(mu,sigma2,s,h,thresh_real))
    return math.log(expected_value)
def WWS(mu,sigma2,s,h,thresh_real=0):
    return (h**2)*(math.e**(2*etha(mu,sigma2,s,h,thresh_real)))/(math.e**(etha(mu,sigma2,2*s,h,thresh_real))+math.e**(etha(mu,sigma2,2-2*s,-h,thresh_real))-2*math.e**(etha(mu,sigma2,s,2*h,thresh_real)))
    #equation 231 in VAN TREESE
############################################################################################################ BBZ FOR REAL THETA!
def gaussian_distri(mu,argu):
    return (1/(math.sqrt(2*math.pi)))*math.exp(-0.5*((argu-mu)**2)) #TODO- I removed the h
def inner2(theta,sigma,h,thresh_real):
    return (norm.cdf((1/sigma)*(theta+h-thresh_real))**2/norm.cdf((1/sigma)*(theta-thresh_real)))+\
    (norm.cdf(-(1/sigma)*(theta+h-thresh_real))**2/norm.cdf(-(1/sigma)*(theta-thresh_real)))
def ratio_func2(theta,mu,sigma,h,thresh_real):
    return gaussian_distri(mu,theta+h)**2/gaussian_distri(mu,theta)*inner2(theta,sigma,h,thresh_real)-\
    2*gaussian_distri(mu,theta+h)*(norm.cdf((1/sigma)*(theta+h-thresh_real))+norm.cdf(-(1/sigma)*(theta+h-thresh_real)))+\
    gaussian_distri(mu,theta)*(norm.cdf((1/sigma)*(theta-thresh_real))+norm.cdf(-(1/sigma)*(theta-thresh_real)))
def integ(mu,sigma2,h,thresh_real):
    expected_value, _ = quad(ratio_func2,-15, 15, args=(mu,sigma2,h,thresh_real))
    return math.log(expected_value)
def BBZ(mu,sigma,h,thresh_real=0):
    return (h**2)/(integ(mu,sigma,h,thresh_real))
############################################################################################################ Approximation
def probability(sigma, nq, matrix, monte): #for approximation
    prob_vec = np.zeros((monte))
    for i in range(monte):
        real_teta = np.random.normal(mu, sigma_teta, M)
        im_teta = np.random.normal(mu, sigma_teta, M)
        teta = real_teta + 1j * im_teta
        teta = teta.reshape(M, 1)
        x_observ = x(sigma, sigma, 0, nq, matrix, teta, 0, 0)[
            1]  # (sigma1, sigma2, n_a, n_q, matrix, teta, thresh_real, thresh_im)
        if np.all(x_observ == x_observ[0]):
            prob_vec[i] = 1
    return np.mean(prob_vec)
def LMMSE_numerical_ONEBIT(sigma1, sigma2, n_a, n_q, matrix, observ, snap=1000, thresh_real=0, thresh_im=0):
    cov = np.zeros((observ, M, M))
    _, x_q_vec, teta_vec = samp(sigma1, sigma2, n_a, n_q, matrix, snap, thresh_real, thresh_im)
    cov_teta_x = covariance(teta_vec, x_q_vec)
    K = np.cov(x_q_vec)  # covariance(x_q_vec,x_q_vec) #np.cov(x_q_vec) #Nq X Nq
    U, S, V = np.linalg.svd(K)
    epsilon = np.std(np.diag(S))
    K = K + epsilon * np.identity(K.shape[0])
    cov_x_inv = LA.inv(K)

    for i in range(observ):
        real_teta = np.random.normal(mu, sigma_teta, M)
        im_teta = np.random.normal(mu, sigma_teta, M)
        teta = real_teta + 1j * im_teta
        teta = teta.reshape(M, 1)

        p = norm.cdf(thresh_real / (np.sqrt(0.5 * (sigma1 ** 2 + rho_q)))) * np.ones((n_q * M, 1))
        x_e = (1 / math.sqrt(2)) * (1 - 2 * p) * (1 + 1j)

        _, x_q = x(sigma1, sigma2, n_a, n_q, matrix, teta, thresh_real, thresh_im)  # the actually observations
        x_q_vec_norm = x_q.reshape(M * n_q, 1) - x_e * np.ones((n_q * M, 1))  # np.mean(x_q_vec)#np.mean(x_q_vec)

        teta_hat = (cov_teta_x @ cov_x_inv @ x_q_vec_norm) + (mu + 1j * mu) * np.ones(M)
        cov[i, :, :] = ((teta_hat - teta) @ ((teta_hat - teta).conjugate().T)).real  # m>1, real number
    cov_matrix = np.sum(cov, 0) / (np.shape(cov)[0])
    return LA.norm(cov_matrix, "fro")  # M>1 np.squeeze(cov_matrix)
############################################################################################################ WBCRB
def ET_CRB(sigma1, sigma2, n_a, n_q, observ=sim):  # M=1 ! its the jensen imquallity usage- not the ATBCRB
    teta_samp = samp_teta(observ)
    G = np.ones((n_q * M, M))
    g_teta = G @ teta_samp
    zeta_real = ((math.sqrt(2) / sigma2) * (g_teta.real))
    zeta_im = ((math.sqrt(2) / sigma2) * (g_teta.imag))
    pdf_real = norm.pdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    d_vec = ((n_q * rho_q) / (2 * pow(sigma2, 2))) * (
                np.divide(np.power(pdf_real, 2), np.multiply(norm.cdf(zeta_real), (norm.cdf(-zeta_real)))) + \
                np.divide(np.power(pdf_im, 2), np.multiply(norm.cdf(zeta_im), (norm.cdf(-zeta_im)))))

    return np.mean(1 / (1 + (rho_a * n_a / pow(sigma1, 2)) + d_vec[:1, :] * np.abs(G[0, :]) ** 2))  # M=1 !
def weighted_fun(theta_real, theta_imag, sigma1, sigma2, na, nq):
    zeta_real = (math.sqrt(2) / sigma2) * (theta_real)
    zeta_im = (math.sqrt(2) / sigma2) * (theta_imag)
    d = norm.pdf(zeta_real) ** 2 / (norm.cdf(zeta_real) * (norm.cdf(-zeta_real))) + norm.pdf(zeta_im) ** 2 / (
                norm.cdf(zeta_im) * (norm.cdf(-zeta_im)))
    return 1/(1+(nq*d)/(2*sigma2**2))

def weighted_BCRB(sigma1, sigma2, n_a, n_q, monte, thresh_real=0, thresh_im=0):
    monte2 = 10
    delta = 1e-5
    result = np.zeros((monte))
    weighted_vec = np.zeros((monte))
    theta_org = samp_teta(monte)[0]
    for j in range(monte):
        matrix = [np.ones((n_a * M, M)), np.ones((n_q * M, M))]
        theta_real, theta_imag = theta_org[j].real, theta_org[j].imag
        theta = theta_real + 1j * theta_imag
        weighted = weighted_fun(theta_real, theta_imag, sigma1, sigma2, n_a, n_q)
        weighted_vec[j] = weighted
        result2 = np.zeros((monte2))
        for i in range(monte2):
            x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, theta)
            zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
            zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
            logP_x_q_der = np.sum(((((norm.pdf(zeta_real) / (norm.cdf(zeta_real) * (-norm.cdf(-zeta_real)))) *
                                 (norm.cdf(zeta_real) - 0.5 - x_q.real.reshape(n_q, M) / math.sqrt(2))))
                               - 1j * (((norm.pdf(zeta_im) / (norm.cdf(zeta_im) * (-norm.cdf(-zeta_im)))) *
                                        (norm.cdf(zeta_im) - 0.5 - x_q.imag.reshape(n_q, M) / math.sqrt(2))))) * (
                                          1 / (sigma2 * math.sqrt(2))), 0)
            logf_der = -theta.conjugate() + matrix[0].transpose() @ (
                        x_a.reshape(n_a, M) - matrix[0] * theta).conjugate() / (sigma1 ** 2) + logP_x_q_der
            divv = 0.5 * (((weighted_fun(theta_real + delta, theta_imag, sigma1, sigma2, n_a,
                                         n_q) - weighted) / delta) - 1j * ((weighted_fun(theta_real, theta_imag + delta,sigma1, sigma2, n_a,n_q) - weighted) / delta))  # div_weighted(theta,sigma1,sigma2,n_a,n_q)
            result2[i] = np.abs(divv + weighted * logf_der) ** 2
        result[j] = np.mean(result2)
    return np.abs(np.mean(weighted_vec)) ** 2 / np.mean(result)

############################################################################################################ Bhattacharyya - not working well
def logP_x_q_der(theta,x_q,sigma2, matrix, n_q, thresh_real=0, thresh_im=0):
    zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
    zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
    return np.sum(((((norm.pdf(zeta_real) / (norm.cdf(zeta_real) * (-norm.cdf(-zeta_real)))) *
                                 (norm.cdf(zeta_real) - 0.5 - x_q.real.reshape(n_q, M) / math.sqrt(2))))
                               - 1j * (((norm.pdf(zeta_im) / (norm.cdf(zeta_im) * (-norm.cdf(-zeta_im)))) *
                                        (norm.cdf(zeta_im) - 0.5 - x_q.imag.reshape(n_q, M) / math.sqrt(2))))) * (
                                          1 / (sigma2 * math.sqrt(2))), 0)

def logP_x_q_der2(theta,x_q,sigma2, matrix, n_q, thresh_real=0, thresh_im=0):
    zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
    zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
    return np.sum(((((norm.pdf(zeta_real)**2 / (norm.cdf(zeta_real)**2 * (-norm.cdf(-zeta_real))**2)) *
                                 ((0.5+x_q.real.reshape(n_q, M)/math.sqrt(2))*(-zeta_real*norm.cdf(zeta_real)-norm.pdf(zeta_real))*(norm.cdf(zeta_real)-np.ones(((n_q, M))))**2
                                  +(0.5-x_q.real.reshape(n_q, M)/math.sqrt(2))*(norm.cdf(zeta_real)-zeta_real-norm.pdf(zeta_real))*(norm.cdf(zeta_real))**2)))
                               - 1j * (((norm.pdf(zeta_im)**2 / (norm.cdf(zeta_im)**2 * (-norm.cdf(-zeta_im))**2)) *
                                 ((0.5+x_q.imag.reshape(n_q, M)/math.sqrt(2))*(-zeta_im*norm.cdf(zeta_im)-norm.pdf(zeta_im))*(norm.cdf(zeta_im)-np.ones(((n_q, M))))**2
                                  +(0.5-x_q.imag.reshape(n_q, M)/math.sqrt(2))*(norm.cdf(zeta_im)-zeta_im-norm.pdf(zeta_im))*(norm.cdf(zeta_im))**2)))) * (
                                          1 / (sigma2**2 * 2)), 0)

def Bhattacharyya_onebit(sigma1, sigma2, n_a, n_q, monte):
    monte2 = 50
    delta = 1e-5
    G_11 = np.zeros((monte),dtype=complex)
    G_22 = np.zeros((monte),dtype=complex)
    G_12 = np.zeros((monte),dtype=complex)
    theta_org = samp_teta(monte)[0]
    for j in range(monte):
        matrix = [np.ones((n_a * M, M)), np.ones((n_q * M, M))]
        theta_real, theta_imag = theta_org[j].real, theta_org[j].imag
        theta = theta_real + 1j * theta_imag
        G_11_argu = np.zeros((monte2),dtype=complex)
        G_22_argu = np.zeros((monte2),dtype=complex)
        G_12_argu = np.zeros((monte2),dtype=complex)
        for i in range(monte2):
            x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, theta)
            logP_deff = logP_x_q_der(theta,x_q,sigma2, matrix, n_q)
            logP_deff2 = logP_x_q_der2(theta,x_q,sigma2, matrix, n_q)
            logf_der = -theta.conjugate()+ logP_deff
            logf_der_2 = (-1+logP_deff2)
                          #0.5*(((logP_x_q_der(theta+delta,x_q, sigma2,matrix, n_q) - logP_deff) / delta)-1j*((logP_x_q_der(theta+1j*delta,x_q, sigma2,matrix, n_q) - logP_deff) / delta)))
            G_11_argu[i] = np.abs(logf_der)**2
            G_22_argu[i] = np.abs(logf_der_2)**2
            G_12_argu[i] = logf_der*logf_der_2
        G_11[j] = np.mean(G_11_argu)
        G_22[j] = np.mean(G_22_argu)
        G_12[j] = np.mean(G_12_argu)
    return 1/np.mean(G_11)+np.abs(np.mean(G_12))**2/np.mean(G_11)*(np.mean(G_11)*np.mean(G_22)-np.abs(np.mean(G_12))**2)

############################################################################################################ Stein
def CRB_pp(sigma1,sigma2, n_a,n_q,matrix, observ=sim,thresh_real=0,thresh_im=0): #Mistake
    teta_samp = samp_teta(observ)
    g_teta = matrix[1] @ teta_samp
    zeta_real = ((math.sqrt(2)/sigma2)*(g_teta.real-thresh_real))
    zeta_im = ((math.sqrt(2)/sigma2)*(g_teta.imag-thresh_im))
    pdf_real = norm.pdf(zeta_real)
    pdf_im = norm.pdf(zeta_im)
    J2_vec = np.mean(((pdf_real+pdf_im)*matrix[1]),1)
    J2 = (1/(pow(sigma2,2)))*J2_vec.transpose().conjugate()@J2_vec
    J1 = (1 + (rho_a * n_a / pow(sigma1, 2))) * np.identity(M)
    J = J1 + J2
    return LA.norm((LA.inv(J)).real,"fro")

# def CRB_pp_new(sigma1,sigma2, n_a,n_q,matrix, observ=sim,thresh_real=0,thresh_im=0): #The theoretical Numeric_thresh - not sure what is it
#     alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma2, 2)))
#     x_a_vec, x_q_vec, teta_vec = samp(sigma1,sigma2, n_a,n_q, matrix,int(5e4),thresh_real,thresh_im)
#     teta_samp = teta_vec.transpose()
#     J2_vec = np.zeros((observ,1), dtype=complex)
#     K = alpha*np.identity(M*n_q)+((1-alpha)/rho_q)*matrix[1]@matrix[1].transpose().conjugate()#covariance(x_q_vec,x_q_vec)
#     # if np.linalg.det(K) == 0: #K is singular
#     #     # print("Singular")
#     #     U,S,V = np.linalg.svd(K)
#     #     # S[S<0.1] = 0.1
#     #     epsilon = np.std(np.diag(S))#random.uniform(0.01, 15)
#     #     K = K+epsilon*np.identity(K.shape[0])# U@np.diag(S)@V
#     # else:
#     #     # print("Non singular")
#     #     while True:
#     #         U,S,V = np.linalg.svd(K)
#     #         epsilon = np.std(np.diag(S))#random.uniform(0.01, 15)
#     #         # S[S<epsilon] = epsilon
#     #         K = K+epsilon*np.identity(K.shape[0])#U@np.diag(S)@V
#     #         if float(LA.cond(K)) < float(1.2):
#     #             # print(epsilon)
#     #             break
#     A = LA.inv(K)
#     for i in range(observ):
#         g_teta = matrix[1]*teta_samp[i]
#         zeta_real = ((math.sqrt(2)/sigma2)*(g_teta.real-thresh_real))
#         zeta_im = ((math.sqrt(2)/sigma2)*(g_teta.imag-thresh_im))
#         pdf_real = norm.pdf(zeta_real)
#         pdf_im = norm.pdf(zeta_im)
#         J2_vec[i] = ((pdf_real+pdf_im)*matrix[1]).transpose().conjugate()@A@((pdf_real+pdf_im)*matrix[1])
#     J2 = (1/(2*pow(sigma2,2)))*np.mean(J2_vec,0)
#     J1 = (1 + (rho_a * n_a / pow(sigma1, 2)))*np.identity(M)
#     J = J1 + J2
#     return LA.norm((LA.inv(J)).real,"fro")
#
# def CRB2(sigma1, sigma2, n_a, n_q, matrix, observ=sim, thresh_real=0, thresh_im=0): #not sure what is it
#     teta_samp = samp_teta(observ)
#     g_teta = matrix[1] @ teta_samp
#     G_normal = matrix[1] / math.sqrt(n_q * rho_q)
#     zeta_real = ((math.sqrt(2) / sigma2) * (g_teta.real - thresh_real))
#     zeta_im = ((math.sqrt(2) / sigma2) * (g_teta.imag - thresh_im))
#     pdf_real = norm.pdf(zeta_real)
#     pdf_im = norm.pdf(zeta_im)
#
#     d_vec = np.divide(np.power(np.divide(pdf_real, np.power(norm.cdf(-zeta_real), 2)) + \
#                                np.divide(pdf_real, math.sqrt(2) * norm.cdf(zeta_real)), 2) + \
#                       np.power(np.divide(pdf_im, np.power(norm.cdf(-zeta_im), 2)) + \
#                                np.divide(pdf_im, math.sqrt(2) * norm.cdf(zeta_im)), 2),
#                       np.multiply(norm.cdf(-zeta_real), norm.cdf(-zeta_im)))
#
#     d_vec2 = np.power(np.multiply(norm.cdf(-zeta_real), norm.cdf(-zeta_im)), -0.5)
#
#     my_vector = [
#         (n_q * rho_q * d_vec[i, :]) * G_normal[i].reshape(M, 1).conjugate() * G_normal[i].reshape(M, 1).transpose() for
#         i in range(np.shape(d_vec)[0])]
#     my_vector = np.array(my_vector)[:, 0, :]
#     J2 = np.sum(my_vector, axis=0) * (1 / (8 * pow(sigma2, 2)))
#     bound = ((np.mean(d_vec2).real) ** 2) / (np.mean(J2).real)
#     return bound
#
#
#
#
#
