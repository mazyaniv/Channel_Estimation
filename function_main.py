from scipy.stats import norm
from numpy import linalg as LA
import scipy.integrate as spi
from function_intro import *
rho_a=rho_q=1 #note
def MSE_zertothresh_analytic(sigma1,sigma2, n_a,n_q):
    alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma2, 2)))
    beta = ((1-alpha)/rho_q)-((2*rho_a*n_a)/(math.pi*(rho_q+pow(sigma2,2))*(rho_a*n_a+pow(sigma1,2))))
    first = (rho_a*n_a)/(rho_a*n_a+pow(sigma1, 2))
    second = (2*rho_q*n_q*pow(sigma1,4))/(math.pi*(rho_q+pow(sigma2, 2))*(alpha+beta*rho_q*n_q)*pow(rho_a*n_a+pow(sigma1, 2),2))
    return M*(1-first-second)

def MSE_biglaw(sigma1,sigma2, n_a, n_q, matrix, observ=sim,thresh_real=0,thresh_im=0):
    MSE = np.zeros((observ, M, M))
    for i in range(observ):
        real_teta = np.random.normal(mu, sigma_teta,M)
        im_teta = np.random.normal(mu, sigma_teta,M)
        teta = real_teta + 1j*im_teta
        teta = teta.reshape(M,1)
        x_a, x_q = x(sigma1,sigma2, n_a, n_q, matrix,teta,thresh_real,thresh_im)  # function of teta
        alpha = (2 / math.pi) * math.acos(rho_q / (rho_q + pow(sigma2, 2)))
        C = (2 * rho_q * rho_a * n_a) / (math.pi * (rho_q + pow(sigma2, 2)) * (rho_a * n_a + pow(sigma1, 2)))
        w_q = (pow(sigma1, 2) / (pow(sigma1, 2) + (rho_a * n_a))) * (
                    (alpha + (1 - alpha) * n_q) / (alpha + ((1 - alpha - C) * n_q)))
        w_a = 1 - ((2 * rho_q * n_q * w_q) / (math.pi * (rho_q + pow(sigma2, 2)) * (alpha + (1 - alpha) * n_q)))
        teta_a_hat = (1 / ((rho_a * n_a) + pow(sigma1, 2))) * ((matrix[0].conjugate().T) @ (x_a))
        teta_q_hat = math.sqrt(2 / (math.pi * (rho_q + pow(sigma2, 2)))) * (1 / (alpha + ((1 - alpha) * n_q)))*((matrix[1].conjugate().T) @ (x_q))
        teta_hat = (w_a * teta_a_hat)+(w_q * teta_q_hat)
        MSE[i, :, :] = ((teta_hat - teta)@((teta_hat - teta).conjugate().T)).real
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")

def LS(sigma1,sigma2, n_a, n_q, matrix, observ=sim,thresh_real=0,thresh_im=0):
    MSE = np.zeros((observ, M, M))
    for i in range(observ):
        real_teta = np.random.normal(mu, sigma_teta,M)
        im_teta = np.random.normal(mu, sigma_teta,M)
        teta = real_teta + 1j*im_teta
        teta = teta.reshape(M,1)
        x_a, x_q = x(sigma1,sigma2, n_a, n_q, matrix,teta,thresh_real,thresh_im)  # function of teta
        H = matrix[0]
        teta_hat = (1/(n_a*rho_a))*H.conjugate().T@x_a
        MSE[i, :, :] = ((teta_hat - teta)@((teta_hat - teta).conjugate().T)).real
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")

def E_theta_givenx_integral(sigma1,sigma2, n_a, n_q,matrix,observ=sim,thresh_real=0,thresh_im=0): #M=1
    def integrand1(u,v):
            p_theta =  (1/math.pi)*math.exp((-((u+1j*v)-mu).conjugate()*((u+1j*v)-mu)).real)
            f_xa = (1/(pow(math.pi,n_a)*pow(sigma1,2*n_a)))*math.exp((-(1/pow(sigma1,2))*(np.subtract(x_a.reshape(n_a,1),matrix[0]*(u+1j*v))).transpose().conjugate()@(np.subtract(x_a.reshape(n_a,1),matrix[0]*(u+1j*v)))).real)
            zeta_real = (math.sqrt(2) / sigma2 * (matrix[1]*(u+1j*v).real - thresh_real))
            zeta_im = (math.sqrt(2) / sigma2 * (matrix[1]*(u+1j*v).imag - thresh_im))
            p_xq = np.prod(np.power(norm.cdf(zeta_real),(0.5+x_q.real/math.sqrt(2)).reshape(-1,1)))\
            *np.prod(np.power(norm.cdf(zeta_im),(0.5+x_q.imag/math.sqrt(2)).reshape(-1,1)))*np.prod(np.power(norm.cdf(-zeta_real),(0.5-x_q.real/math.sqrt(2)).reshape(-1,1)))*np.prod(np.power(norm.cdf(-zeta_im),(0.5-x_q.imag/math.sqrt(2)).reshape(-1,1)))
            return p_theta*f_xa*p_xq
    def integrand2(u,v):
        pro = integrand1(u,v)
        return u*(pro)+1j*v*(pro)

    def integrand1_real(u,v):
        return integrand1(u,v).real
    def integrand1_imag(u,v):
        return integrand1(u,v).imag
    def integrand2_real(u,v):
        return integrand2(u,v).real
    def integrand2_imag(u,v):
        return integrand2(u,v).imag
    monte = observ
    MSE = np.zeros((monte, M, M))
    for j in range(monte):
        x_a, x_q,theta_org =samp(sigma1,sigma2, n_a,n_q, matrix, 1,thresh_real,thresh_im)
        lim_inf, lim_sup = [-3,3]
        result1_real = spi.dblquad(integrand1_real, lim_inf, lim_sup, lim_inf, lim_sup)[0]
        result1_imag = spi.dblquad(integrand1_imag, lim_inf, lim_sup, lim_inf, lim_sup)[0]
        result2_real = spi.dblquad(integrand2_real, lim_inf, lim_sup,  lim_inf, lim_sup)[0]
        result2_imag = spi.dblquad(integrand2_imag, lim_inf, lim_sup,  lim_inf, lim_sup)[0]
        teta_hat = (result2_real+1j*result2_imag)/(result1_real+1j*result1_imag)
        MSE[j, :, :]= (((teta_hat-theta_org)@((teta_hat-theta_org).conjugate().T)).real)
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")

def E_theta_givenx_Reiman(sigma1, sigma2, n_a, n_q, matrix, observ=sim, thresh_real=0, thresh_im=0): # M=1
    def integrand1(u, v):
        p_theta = (1/math.pi) * math.exp((-((u+1j*v)-mu).conjugate()*((u+1j*v)-mu)).real)
        f_xa = (1/(pow(math.pi, n_a)*pow(sigma1, 2*n_a))) * math.exp((-(1/pow(sigma1, 2)) * (np.subtract(x_a.reshape(n_a, 1), matrix[0]*(u+1j*v))).transpose().conjugate() @ (np.subtract(x_a.reshape(n_a, 1), matrix[0]*(u+1j*v)))).real)
        zeta_real = (math.sqrt(2) / sigma2 * (matrix[1]*(u+1j*v).real - thresh_real))
        zeta_im = (math.sqrt(2) / sigma2 * (matrix[1]*(u+1j*v).imag - thresh_im))
        p_xq = np.prod(np.power(norm.cdf(zeta_real), (0.5+x_q.real/math.sqrt(2)).reshape(-1, 1))) \
               * np.prod(np.power(norm.cdf(zeta_im), (0.5+x_q.imag/math.sqrt(2)).reshape(-1, 1))) \
               * np.prod(np.power(norm.cdf(-zeta_real), (0.5-x_q.real/math.sqrt(2)).reshape(-1, 1))) \
               * np.prod(np.power(norm.cdf(-zeta_im), (0.5-x_q.imag/math.sqrt(2)).reshape(-1, 1)))
        return p_theta * f_xa * p_xq

    def integrand2(u, v):
        pro = integrand1(u, v)
        return u * (pro) + 1j*v * (pro)

    def integrand1_real(u, v):
        return integrand1(u, v).real

    def integrand1_imag(u, v):
        return integrand1(u, v).imag

    def integrand2_real(u, v):
        return integrand2(u, v).real

    def integrand2_imag(u, v):
        return integrand2(u, v).imag

    monte = observ
    M = 1  # Assuming M is defined somewhere else in your code
    MSE = np.zeros((monte, M, M))
    for j in range(monte):
        x_a, x_q, theta_org = samp(sigma1, sigma2, n_a, n_q, matrix, 1, thresh_real, thresh_im)
        lim_inf, lim_sup = [-3, 3]
        N = 100
        du = (lim_sup - lim_inf) / N
        dv = (lim_sup - lim_inf) / N
        sum_real = 0
        sum_imag = 0
        for u in np.arange(lim_inf, lim_sup, du):
            for v in np.arange(lim_inf, lim_sup, dv):
                sum_real += integrand1_real(u, v) * du * dv
                sum_imag += integrand1_imag(u, v) * du * dv
        result1_real = sum_real
        result1_imag = sum_imag
        sum_real = 0
        sum_imag = 0
        for u in np.arange(lim_inf, lim_sup, du):
            for v in np.arange(lim_inf, lim_sup, dv):
                sum_real += integrand2_real(u, v) * du * dv
                sum_imag += integrand2_imag(u, v) * du * dv
        result2_real = sum_real
        result2_imag = sum_imag
        teta_hat = (result2_real + 1j * result2_imag) / (result1_real + 1j * result1_imag)
        MSE[j, :, :] = (((teta_hat - theta_org) @ ((teta_hat - theta_org).conjugate().T)).real)
    cov_matrix = np.mean(MSE, 0)
    return LA.norm(cov_matrix, "fro")

def E_theta_givenx_numeric(sigma1, sigma2, n_a, n_q, matrix, observ, monte=sim, thresh_real=0, thresh_im=0):
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


def MSE_general_numerical(sigma1, sigma2, n_a, n_q, matrix, observ, snap=1000, thresh_real=0, thresh_im=0):
    cov = np.zeros((observ, M, M))
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
        teta = teta.reshape(M, 1)

        # mu_tilda_real = mu*(np.sum(matrix[1].real,axis=1)-np.sum(matrix[1].imag,axis=1))
        # mu_tilda_imag = mu*(np.sum(matrix[1].real,axis=1)+np.sum(matrix[1].imag,axis=1))
        # sigma_tilda = 1+(np.sum(np.power(matrix[1].real,2),axis=1)+np.sum(np.power(matrix[1].imag,2),axis=1))
        # p1 = norm.cdf(np.divide(np.subtract(thresh_real,mu_tilda_real.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))
        # p2 = norm.cdf(np.divide(np.subtract(thresh_im,mu_tilda_imag.reshape(n_q*M,1)),sigma_tilda.reshape(n_q*M,1)))
        p = norm.cdf(thresh_real / (np.sqrt(0.5 * (sigma1 ** 2 + rho_q)))) * np.ones((n_q * M, 1))
        x_e = (1 / math.sqrt(2) )* (1 - 2 * p) * (1 + 1j)

        x_a, x_q = x(sigma1, sigma2, n_a, n_q, matrix, teta, thresh_real, thresh_im)  # the actually observations
        x_a_vec_norm = x_a - np.mean(x_a_vec)  # matrix[0]@((mu+1j*mu)*np.ones(M))
        x_q_vec_norm = x_q.reshape(M * n_q, 1)-x_e*np.ones((n_q*M,1))#np.mean(x_q_vec)#np.mean(x_q_vec)
        x_vec_norm = np.concatenate((x_a_vec_norm, x_q_vec_norm.reshape(M * n_q, )), axis=0)

        teta_hat = (cov_teta_x @ cov_x_inv @ x_vec_norm) + (mu + 1j * mu) * np.ones(M)
        cov[i, :, :] = ((teta_hat - teta) @ ((teta_hat - teta).conjugate().T)).real  # m>1, real number
        # cov[i,:,:] = ((teta_hat-teta)*(teta_hat-teta).conjugate()).real #M=1, real number
    cov_matrix = np.sum(cov, 0) / (np.shape(cov)[0])
    return LA.norm(cov_matrix, "fro")  # M>1 np.squeeze(cov_matrix)
    # print("Error:", LA.norm(cov_matrix, "fro"))#M>1 np.squeeze(cov_matrix)
    # print("========")

def CRB(sigma1,sigma2, n_a,n_q,matrix, quantize=1,observ=sim,thresh_real=0,thresh_im=0):
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
    # print(d)
    if quantize == 0:
        J2 = (2/math.pi)*(rho_q * n_q / pow(sigma2, 2))*np.identity(M)
    else:
        my_vector = [(n_q*rho_q*d[i])*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
        J2 = np.sum(my_vector,axis=0)*(1/(2*pow(sigma2, 2)))
    J1 = (1 + (rho_a * n_a / pow(sigma1, 2))) * np.identity(M)
    J = J1 + J2
    # my_vector = [((2*pow(sigma1,2)*pow(sigma2,2))+(2*rho_a*n_a*pow(sigma2,2))+n_q*rho_q*d[i]*pow(sigma1,2))*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
    # J_sum = np.sum(my_vector,axis=0)*(1/(2*pow(sigma2, 2)*pow(sigma1,2)))
    # my_vector = [(((2 * pow(sigma2, 2))) /((2*pow(sigma1,2)*pow(sigma2,2))+(2*rho_a*n_a*pow(sigma2,2))+n_q*rho_q*d[i]*pow(sigma1,2)))*G_normal[i].reshape(M,1).conjugate()*G_normal[i].reshape(M,1).transpose() for  i in range(len(d))]
    # Bound_sum = np.sum(my_vector,axis=0) #Harder to compute
    return LA.norm((LA.inv(J)).real,"fro")#LA.norm((LA.inv(J_sum)).real,"fro"), , LA.norm((LA.inv(J1)).real,"fro"), LA.norm((LA.inv(J2+np.identity(M))).real,"fro")

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

# def CRB_pp_new(sigma1,sigma2, n_a,n_q,matrix, observ=sim,thresh_real=0,thresh_im=0): #The theoretical numeric_thresh
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

# def CRB2(sigma1, sigma2, n_a, n_q, matrix, observ=sim, thresh_real=0, thresh_im=0):
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