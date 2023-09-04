from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0.01, 20, 10)
fig, axes = plt.subplots(4, 2, figsize=(12, 4))
axes[0,0].plot(x, norm.pdf(1/x))
axes[0,0].set_title('f')

axes[0,1].plot(x, pow(norm.pdf(1/x),2))
axes[0,1].set_title('f^2')

axes[1,0].plot(x, norm.cdf(1/x))
axes[1,0].set_title('F(x)')

axes[1,1].plot(x, norm.cdf(-1/x))
axes[1,1].set_title('F(-x)')

axes[2,0].plot(x, norm.cdf(1/x)*norm.cdf(-1/x))
axes[2,0].set_title('F(x)*F(-x)')

axes[2,1].plot(x, pow(norm.pdf(0/x),2)*(norm.cdf(0/x)*norm.cdf(-0/x)))
axes[2,1].set_title('(f^2)/(F*F(-x))')

axes[3,0].plot(x, pow(1/x,2)*pow(norm.pdf(0/x),2)*(norm.cdf(0/x)*norm.cdf(-0/x)))
axes[3,0].set_title('(x^2)*(f^2)/(F*F(-x))')

fig.subplots_adjust(wspace=0.4, hspace=1)
plt.show()



