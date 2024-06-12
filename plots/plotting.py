import numpy as np
import math
from scipy.stats import norm
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from numpy import linalg
import scipy.integrate
############################################################# Variables
# sigma = np.logspace(-1,1,20) #irrelevant for lower sigma
# #thresh = 0 #np.linspace(-3,3,200)
# M = 1
# mu = 0 zero mean
# sigma_teta = math.sqrt(1)*(1/math.sqrt(2))
# rho_q = 1
# rho_a = 1
# n_a = [1,1,2,2]
# n_q = [40,100,40,100]
# sim = pow(10,4)
#
# x = random.random()
# y = math.sqrt(1-pow(x,2))
# G1_normal = x+1j*y
# G_mat = math.sqrt(rho_q)*G1_normal
# ##M>1
# #G1_normal = unitary_group.rvs(M)
# #G1 = math.sqrt(rho_q)*G1_normal
# # ones = np.ones(n_q)
# # G = np.kron(ones, G1).transpose()
# x1 = random.random()
# y1 = math.sqrt(1-pow(x1,2))
# H_mat = math.sqrt(rho_a)*(x1+1j*y1)
#
# real_teta = np.random.normal(mu, sigma_teta)
# im_teta = np.random.normal(mu, sigma_teta)
# teta = real_teta + 1j*im_teta
####################################################
names_of_outputs = ['CRB','LMMSE']
out1 = np.load('C:/Users/Yaniv/PycharmProjects/pythonProject/plots/CRB1.npy')
out2 = np.load('C:/Users/Yaniv/PycharmProjects/pythonProject/plots/L_Estimator_numerical.npy')
list_of_outputs = [out1,out2]
list_of_colors = ['red',"black"]
fig = plt.figure(figsize=(10, 6))

thresh_space = np.linspace(-8,8,20)
for i in range(len(names_of_outputs)):
    plt.plot(thresh_space, list_of_outputs[i], color=list_of_colors[i], label=names_of_outputs[i])

plt.title("CRB & LMMSE vs Threshold")
plt.yscale('log')
plt.xlabel("Threshold")
plt.ylabel("BCRB & LMMSE")
plt.legend()
plt.legend()
plt.show()
