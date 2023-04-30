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
sim = pow(10,4)

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
####################################################
names_of_outputs = ['CRB_1_40_10^3_thresh.npy','L_Estimator_numerical1_40_10^3_thresh.npy']

Lower_bound = np.load('C:/Users/Yaniv/PycharmProjects/pythonProject/CRB_1_40_10^3_thresh.npy')
L_Estimator = np.load('C:/Users/Yaniv/PycharmProjects/pythonProject/L_Estimator_numerical1_40_10^3_thresh.npy')

list_of_outputs = [Lower_bound,L_Estimator]
list_of_colors = ['red',"black"]
fig = plt.figure(figsize=(10, 6))

for i in range(len(names_of_outputs)):
    plt.plot(10*np.log10(1/sigma), list_of_outputs[i], color=list_of_colors[i], label=names_of_outputs[i])

plt.title("CRB with threshold- mu={}, variance ={}".format(mu,round(2*pow(sigma_teta,2))))
plt.yscale('log')
plt.xlabel("SNR [dB]")
plt.ylabel("BCRB VS L-MSE")
plt.legend()
plt.legend()
plt.show()