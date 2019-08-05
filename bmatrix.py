'''
Compute the coefficients of the solution to the air gap and magnet region flux density

Also compute the Fourier coefficients specifically for the radial component of the air gap

Based on Appendix B of 'Brushless Permanent Magnet Motor Design' by Duane P. Hanselman
'''

import numpy as np
import matplotlib.pyplot as plt

# Definitions of magnetic field profiles, which determine k_rn and k_tn. Use as appropriate

# Radial Sinusoidal Profile
def RadSinR(n, alpha):
    if n == 1 or n == -1:
        return .5
    else:
        return 0

def RadSinT(n, alpha):
    return 0

# Radial Profile
def RadialR(n, alpha):

    if n%2 != 0:
        return alpha*np.sin(n*alpha*np.pi/2)/(n*alpha*np.pi/2)

    if n%2 == 0:
        return 0

def RadialT(n, alpha):
    return 0


# Permitivity of Free Space. Cancels out in computation, so the value doesn't matter
mu_0 = 0.001


# Compute the coefficients of the assumed solution using the boundary conditions
# Solves a size 3 linear system for every nth Fourier series term
def B_fourier_terms(n, params):
    n_p = params['n_p'] #number of pole pairs
    r_s = params['r_s'] #stator inner radius
    r_m = params['r_m'] #rotor outer radius (INCLUDING MAGNETS)
    r_r = params['r_r'] #rotor outer radius
    b_r = params['b_r'] #magnetic remanence of stator material
    mu_r = params['mu_r'] #magnetic relative recoil permitivity
    alpha = params['alpha'] #magnet fraction in rotor

    bmat = np.zeros((3, 3))
    rhs = np.zeros(3)

    beta = n*n_p

    # magnetic profile coefficients
    k_rn = RadialR(n, alpha)
    k_tn = RadialT(n, alpha)

    # assume output is real, ignore imaginary part (Equation B.18)
    Cm = b_r*(k_rn)/((1-beta**2)*mu_r*mu_0)

    #Hatheta boundary condition (excluded from linear system)
    Ea = -(r_s)**(2*beta)
    print(r_s)
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

    print(bmat)
    #invert matrix
    bmat_inv = np.linalg.inv(bmat)

    #solve system using inverse of system matrix
    x = bmat_inv.dot(rhs)
    print(x)
    return x

# Compute Fourier Coefficients of radial component air gap magnetic flux Density
def B_arn(b_fourier, n, params, r):
    r_s = params['r_s'] #stator inner radius
    n_p = params['n_p'] #number of pole pairs

    Da = -b_fourier[0]
    beta = n*n_p
    Ea = -(r_s)**(2*beta)
    b_arn = mu_0*beta*Da*(Ea*(r**(-beta-1)) - r**(beta-1)) #Equation B.13
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
