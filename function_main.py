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

# from scipy import integrate
# from scipy.stats import norm
# def integrand_mmse(theta_real,theta_imag,x_a,x_q,sigma1, sigma2, n_a, n_q, matrix,flag=1,thresh_real=0, thresh_im=0):
#     theta = theta_real+1j*theta_imag
#     f_xa = (1 / (pow(math.pi, n_a) * pow(sigma1, 2 * n_a))) * math.exp(np.real((-(1 / pow(sigma1, 2)) * (
#         np.subtract(x_a, matrix[0] * theta)).transpose().conjugate() @ (np.subtract(x_a, matrix[0] *theta)))))
#     zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
#     zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
#
#     p_xq = np.prod(np.power(norm.cdf(zeta_real), (0.5 + x_q.real / math.sqrt(2)))).reshape(-1, 1) \
#            * np.prod(np.power(norm.cdf(zeta_im), (0.5 + x_q.imag / math.sqrt(2)))).reshape(-1, 1)* np.prod(
#         np.power(norm.cdf(-zeta_real), (0.5 - x_q.real / math.sqrt(2)))).reshape(-1, 1) * np.prod(
#         np.power(norm.cdf(-zeta_im), (0.5 - x_q.imag / math.sqrt(2)))).reshape(-1, 1)
#     if flag==2:
#         return (theta*f_xa*p_xq).real
#     else:
#         return (f_xa*p_xq).real
# def MMSE_integ(sigma1, sigma2, n_a, n_q, matrix, monte=sim, thresh_real=0, thresh_im=0):
#     lim = 5
#     MSE = np.zeros((monte))
#     for j in range(monte):
#         x_a, x_q, theta_org = samp(sigma1, sigma2, n_a, n_q, matrix, 1, thresh_real,thresh_im)  # observations made by original theta
#         theta_org = theta_org[0]
#         result2, _ = integrate.dblquad(integrand_mmse, -lim, lim, -lim, lim,args=(x_a, x_q,sigma1, sigma2, n_a, n_q, matrix,2))
#         result1, _ = integrate.dblquad(integrand_mmse, -lim, lim, -lim, lim,args=(x_a, x_q,sigma1, sigma2, n_a, n_q, matrix))
#         teta_hat = result2/result1
#         MSE[j] = (((teta_hat - theta_org)*((teta_hat - theta_org).conjugate())).real)
#     return np.mean(MSE)
def MMSE_func(sigma1, sigma2, n_a, n_q, matrix, snap,monte, thresh_real=0, thresh_im=0):
    MSE = np.zeros((monte, M, M))
    for j in range(monte):
        x_a, x_q, theta_org = samp(sigma1, sigma2, n_a, n_q, matrix, 1, thresh_real,
                                   thresh_im)  # observations made by original theta
        theta_org = theta_org[0]
        # theta_vec = theta_org #observ=1
        theta_vec = samp_teta(snap)[0]
        result2 = np.zeros((snap), complex)
        result1 = np.zeros((snap), complex)

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

        teta_hat = np.mean(result2)/np.nanmean(result1)
        MSE[j, :, :] = (((teta_hat - theta_org) @ ((teta_hat - theta_org).conjugate().T)).real)
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")
def CRB(sigma1,sigma2, n_a,n_q,matrix,observ=sim,quantize=1,thresh_real=0,thresh_im=0): #BCRB
    if sigma1 <= 0.03:
        observ = 10*observ
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

    d = np.nanmean(d_vec, axis=1) #converges to 0.95 aprox.
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
    return LA.norm((LA.inv(J)).real,"fro") #LA.norm((LA.inv(J_sum)).real,"fro"), , LA.norm((LA.inv(J1)).real,"fro"), LA.norm((LA.inv(J2+np.identity(M))).real,"fro")
############################################################################################################
def BBZ_func(sigma1, sigma2, n_a, n_q,matrix, monte,h=0.0001, thresh_real=0, thresh_im=0):
    monte2 = int(monte)
    result = np.zeros((monte),dtype=complex)
    theta_org = samp_teta(monte)[0]
    prior = lambda theta: (1 /(math.pi))*math.exp(-np.abs(theta) ** 2)
    f_xa = lambda theta,x_a: (1/(pow(math.pi*(sigma1**2), n_a)))*np.exp(-(1/pow(sigma1, 2))*((x_a.reshape(n_a,1)-matrix[0]*theta).conj().T@(x_a.reshape(n_a,1)-matrix[0]*theta)))
    zeta_real = lambda theta: (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
    zeta_im = lambda theta: (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
    p_xq = lambda theta,x_q: np.prod(np.power(norm.cdf(zeta_real(theta)), (0.5 + x_q.real / math.sqrt(2)).reshape(-1, 1))) \
                         * np.prod(
        np.power(norm.cdf(zeta_im(theta)), (0.5 + x_q.imag / math.sqrt(2)).reshape(-1, 1))) * np.prod(
        np.power(norm.cdf(-zeta_real(theta)), (0.5 - x_q.real / math.sqrt(2)).reshape(-1, 1))) * np.prod(
        np.power(norm.cdf(-zeta_im(theta)), (0.5 - x_q.imag / math.sqrt(2)).reshape(-1, 1)))
    for j in range(monte): #run over theta
        theta = theta_org[j]
        result2 = np.zeros((monte2),dtype=complex)
        for i in range(monte2): #run over x
            x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, theta)
            result2[i] = np.abs(((f_xa(theta+h,x_a) * p_xq(theta+h,x_q)*prior(theta+h))/
                                (f_xa(theta,x_a)*p_xq(theta,x_q)*prior(theta)))-1)**2
        result[j] = np.mean(result2)
    return 2*h**2/np.mean(result)
############################################################################################################ Bhattacharyya
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

def Bhattacharyya_func(sigma1, sigma2, n_a, n_q,matrix, monte):
    monte2 = 100
    delta = 1e-5
    G_11 = np.zeros((monte),dtype=complex)
    G_22 = np.zeros((monte),dtype=complex)
    G_12 = np.zeros((monte),dtype=complex)
    theta_org = samp_teta(monte)[0]
    for j in range(monte):
        theta_real, theta_imag = theta_org[j].real, theta_org[j].imag
        theta = theta_real + 1j * theta_imag
        G_11_argu = np.zeros((monte2),dtype=complex)
        G_22_argu = np.zeros((monte2),dtype=complex)
        G_12_argu = np.zeros((monte2),dtype=complex)
        for i in range(monte2):
            x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, theta)
            logP_deff = logP_x_q_der(theta,x_q,sigma2, matrix, n_q)
            logP_deff2 = logP_x_q_der2(theta,x_q,sigma2, matrix, n_q)
            logf_der = -theta.conjugate()+logP_deff+matrix[0].transpose()@(x_a.reshape(n_a, M)-matrix[0] * theta).conjugate()/(sigma1 ** 2)
            logf_der_2 = (-1+logP_deff2-(n_a/sigma1**2)) #TODO
                          #0.5*(((logP_x_q_der(theta+delta,x_q, sigma2,matrix, n_q) - logP_deff) / delta)-1j*((logP_x_q_der(theta+1j*delta,x_q, sigma2,matrix, n_q) - logP_deff) / delta)))
            G_11_argu[i] = np.abs(logf_der)**2
            G_22_argu[i] = np.abs(logf_der_2)**2
            G_12_argu[i] = logf_der*logf_der_2
        G_11[j] = np.mean(G_11_argu)
        G_22[j] = np.mean(G_22_argu)
        G_12[j] = np.mean(G_12_argu)
    return 1/np.mean(G_11)+np.abs(np.mean(G_12))**2/(np.mean(G_11)*(np.mean(G_11)*np.mean(G_22)-np.abs(np.mean(G_12))**2))

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
############################################################################################################ Approximation
def probability(sigma, na,nq, matrix, monte): #for approximation
    prob_vec = np.zeros((monte))
    for i in range(monte):
        real_teta = np.random.normal(mu, sigma_teta, M)
        im_teta = np.random.normal(mu, sigma_teta, M)
        teta = real_teta + 1j * im_teta
        teta = teta.reshape(M, 1)
        x_observ = x(sigma, sigma, na, nq, matrix, teta, 0, 0)[1]  # (sigma1, sigma2, n_a, n_q, matrix, teta, thresh_real, thresh_im)
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

def weighted_fun_div(theta, sigma1, sigma2, na, nq,matrix):
    G = matrix[1][0] #block matrix
    zeta_real = (math.sqrt(2) / sigma2)*((G * theta).real)
    zeta_im = (math.sqrt(2) / sigma2)*((G * theta).imag)
    d = norm.pdf(zeta_real) ** 2 / (norm.cdf(zeta_real) * (norm.cdf(-zeta_real))) + norm.pdf(zeta_im) ** 2 / (
            norm.cdf(zeta_im) * (norm.cdf(-zeta_im)))
    f_divv = lambda x: ((-2 * x * (norm.pdf(x)**2) * norm.cdf(x) * (1 - norm.cdf(x))-(norm.pdf(x)**3) * (1 - 2 * norm.cdf(x))) /
    (norm.cdf(x) * (1 - norm.cdf(x)))**2)
    divv_d_x = (math.sqrt(2)/sigma2)*G.real*f_divv(zeta_real)+(math.sqrt(2)/sigma2)*G.imag*f_divv(zeta_im)
    divv_d_y = -(math.sqrt(2)/sigma2)*G.imag*f_divv(zeta_real)+(math.sqrt(2)/sigma2)*G.real*f_divv(zeta_im)
    div_x = -(2*theta.real+((nq*divv_d_x)/(2*sigma2**2)))/(abs(theta)**2+(na/sigma1**2)+((nq*d)/(2*sigma2**2)))**2
    div_y = -(2*theta.imag+((nq*divv_d_y)/(2*sigma2**2)))/(abs(theta)**2+(na/sigma1**2)+((nq*d)/(2*sigma2**2)))**2
    return 0.5*(div_x-1j*div_y)
def weighted_fun(theta, sigma1, sigma2, na, nq,matrix):
    zeta_real = (math.sqrt(2) / sigma2)*((matrix[1] * theta).real)
    zeta_im = (math.sqrt(2) / sigma2)*((matrix[1] * theta).imag)
    d = norm.pdf(zeta_real) ** 2 / (norm.cdf(zeta_real) * (norm.cdf(-zeta_real))) + norm.pdf(zeta_im) ** 2 / (
                norm.cdf(zeta_im) * (norm.cdf(-zeta_im)))
    return 1/(abs(theta)**2+(na/sigma1**2)+((nq*d[0])/(2*sigma2**2))) #d[0] since G is a block matrix

def weighted_BCRB(sigma1, sigma2, n_a, n_q,matrix, monte, thresh_real=0, thresh_im=0):
    weighted_vec = np.zeros((monte),dtype=complex)
    argu = np.zeros((monte),dtype=complex)
    theta_org = samp_teta(monte)[0]
    for j in range(monte): #run over theta
        theta = theta_org[j]
        weighted_vec[j] = weighted_fun(theta, sigma1, sigma2, n_a, n_q,matrix)
        weighted_vec_divv = weighted_fun_div(theta, sigma1, sigma2, n_a, n_q, matrix)
        argu[j] = (weighted_vec[j]**2)*np.abs(weighted_vec_divv)**2
    argu = argu[~np.isnan(argu)]#np.nan_to_num(argu, nan=1e-13)
    weighted_vec = weighted_vec[~np.isnan(weighted_vec)]#np.nan_to_num(weighted_vec, nan=1e-5)
    return (np.abs(np.mean(weighted_vec))**2/(np.mean(weighted_vec)+np.mean(argu))).real
def weighted_BCRB_old(sigma1, sigma2, n_a, n_q,matrix, monte, thresh_real=0, thresh_im=0):
    delta = 1e-5
    monte2 = int(monte)
    result = np.zeros((monte),dtype=complex)
    weighted_vec = np.zeros((monte),dtype=complex)
    theta_org = samp_teta(monte)[0]
    for j in range(monte): #run over theta
        theta = theta_org[j]
        zeta_real = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).real - thresh_real)
        zeta_im = (math.sqrt(2) / sigma2) * ((matrix[1] * theta).imag - thresh_im)
        weighted = weighted_fun(theta, sigma1, sigma2, n_a, n_q,matrix)
        weighted_vec[j] = weighted#[0][0]
        result2 = np.zeros((monte2),dtype=complex)
        for i in range(monte2): #run over x
            x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, theta)
            logP_x_q_der = np.sum(((((norm.pdf(zeta_real) / (norm.cdf(zeta_real) * (-norm.cdf(-zeta_real)))) *
                                 (norm.cdf(zeta_real) - 0.5 - x_q.real.reshape(n_q, M) / math.sqrt(2))))
                               - 1j * (((norm.pdf(zeta_im) / (norm.cdf(zeta_im) * (-norm.cdf(-zeta_im)))) *
                                        (norm.cdf(zeta_im) - 0.5 - x_q.imag.reshape(n_q, M) / math.sqrt(2))))) * (
                                          1 / (sigma2 * math.sqrt(2))), 0)
            logf_der = (-theta.conjugate()
                        +matrix[0].transpose()@(x_a.reshape(n_a, M)-matrix[0] * theta).conjugate()/(sigma1 ** 2) #Kay
                        +logP_x_q_der)
            divv = 0.5 * (((weighted_fun(theta+delta, sigma1, sigma2, n_a,
                                         n_q,matrix) - weighted) / delta) - 1j * ((weighted_fun(theta+1j*delta,sigma1, sigma2, n_a,n_q,matrix) - weighted) / delta))  # div_weighted(theta,sigma1,sigma2,n_a,n_q)
            result2[i] = np.abs(divv + weighted * logf_der) ** 2
        result[j] = np.mean(result2)
    return (np.abs(np.mean(weighted_vec)) ** 2 / np.mean(result)).real
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
