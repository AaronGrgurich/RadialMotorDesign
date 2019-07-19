
import numpy as np
import matplotlib.pyplot as plt

# Da, Em, Dm
# Ea is precomputed
def RadSinR(n, alpha):
    if n == 1 or n == -1:
        return .5
    else:
        return 0

def RadSinT(n, alpha):
    return 0

def RadialR(n, alpha):
    
    if n%2 != 0:
        return alpha*np.sin(n*alpha*np.pi/2)/(n*alpha*np.pi/2)
    
    if n%2 == 0:
        return 0

def RadialT(n, alpha):
    return 0

n_p = 8
r_s = .051
r_m = .05
r_r = .045
b_r = 1.3
mu_r = 1.05
mu_0 = 0.001 #value doesn't matter
alpha = .85

def B_fourier_terms(n):
    bmat = np.zeros((3, 3))
    rhs = np.zeros(3)

    beta = n*n_p
    
    k_rn = RadialR(n, alpha)
    k_tn = RadialT(n, alpha)
    #Cm = b_r*(k_rn + 1j*beta*k_tn)/((1-beta**2)*mu_r*mu_0)
    Cm = b_r*(k_rn)/((1-beta**2)*mu_r*mu_0)
    
    #Hatheta boundary condition
    # bmat[0, 0] = (r_s**(-beta-1))
    # rhs[0] = -r_s**(beta-1)
    Ea = -(r_s)**(2*beta)
    
    #Hmtheta boundary condition
    bmat[0, 1] = (r_r**(-beta-1))
    bmat[0, 2] = (r_r**(beta-1))
    rhs[0] = -Cm
    
    #Htheta boundary condition
    bmat[1, 0] = -Ea*(r_m**(-beta-1)) - r_m**(beta - 1)
    bmat[1, 1] = r_m**(-beta-1) 
    bmat[1, 2] = r_m**(beta-1)
    rhs[1] = -Cm
    
    #Hr boundary condition
    bmat[2, 0] = (beta/mu_r)*(Ea*(r_m**(-beta-1)) - r_m**(beta - 1))
    bmat[2, 1] = -beta*(r_m**(-beta-1))
    bmat[2, 2] = beta*(r_m**(beta-1))
    rhs[2] = (b_r*k_rn/(mu_r*mu_0)) - Cm 
    
    #print(bmat)
    #print(rhs)
    
    bmat_inv = np.linalg.inv(bmat)
    
    #print(bmat_inv)
    
    x = bmat_inv.dot(rhs)
    return x
    
def B_arn(b_fourier, n, r):
    Da = -b_fourier[0]
    #print(Da)
    beta = n*n_p
    Ea = -(r_s)**(2*beta)
    b_arn = mu_0*beta*Da*(Ea*(r**(-beta-1)) - r**(beta-1))
    return b_arn
    
if __name__ == '__main__':
    
    N = 17
    B_r = np.zeros(N)
    
    for n in range(1, N):
        b_f = B_fourier_terms(n)
        B_r[n] = B_arn(b_f, n, r_s)
        print(B_r[n])
        
    i = np.fft.irfft(B_r)
    #print(i)
    plt.plot(i)
    plt.show()
        
