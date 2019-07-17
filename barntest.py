import numpy as np
from math import pi, sin, cos, exp
import matplotlib.pyplot as plt


def RadialR(n, alpha):
    
    if n%2 != 0:
        return alpha*(sin(n*alpha*pi/2))/(n*alpha*pi/2)

    if n%2 == 0:
        return 0

def RadialT(n, alpha):
    return 0


gap = .001
r_s = .051
n_m = 20
n_s = 24
ts = 2*pi/n_s #slot to slot angle 
tt = .6*pi/10 #tooth width angle
mu_r = 1.05
t_mag = .003
l_st = .008
k_st = 1
w_tb = .003
r_m = r_s - gap
b_r = 1.3
alpha = 1/30 #magnetic fraction, fraction of rotor circumference occupied by one magnet
n_p = n_m/2 #number of pole pairs
r_r = r_m - t_mag

N = 21
i = np.zeros(2*N+1, dtype = complex)
for n in range(-N, N+1):
    beta = n*n_p
    k_rn = RadialR(n, alpha)
        
    k_tn = RadialT(n, alpha)

    k_mc = (k_rn + 1j*beta*k_tn)/(1 - beta**2)
    
    del_r = (mu_r + 1)*((r_r/r_m)**(2*beta) - (r_s/r_m)**(2*beta)) + (mu_r - 1)*(1 - ((r_r/r_m)**(2*beta))*((r_s/r_m)**(2*beta)))
    
    #print(del_r)
    if n != 0:
        k_a = (k_mc*k_rn/del_r)*((1-(r_r/r_m)**(2*beta))*((beta + 1)*((r_r/r_m)**(2*beta)) - (2*beta)*((r_r/r_m)**(beta+1)) + beta - 1))
    else:
        k_a = 0

    
    #k_a = k_a.real
    #print(n)
    Bgn = -b_r*k_a*(((r_m/r_m)**(beta-1)) + ((r_s/r_m)**(2*beta))*((r_m/r_m)**(beta+1)))
    print(Bgn)
    i[n+N] = Bgn

plt.plot(i)
plt.show()
