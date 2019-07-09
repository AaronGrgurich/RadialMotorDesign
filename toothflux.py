'''
This function computes magnetic flux in the stator teeth of a motor, described using a Fourier series with respect to angular position theta

Taken from Brushless Permanent Magnet Motor Design, Second Edition by Dr. Duane Hanselman, Chapter 7
Magnetic Flux Density Coefficient formula taken from Appendix B, pg 349
'''
import numpy as np
from math import pi, sin, cos, exp


def toothflux(theta, params, phi):
    n_m = params['n_m'] #number of poles
    n_s = params['n_s'] #number of Slots
    l_st = params['l_st'] #axial motor length
    r_s = params['sta_ir'] #stator inner radius
    t_mag = params['t_mag'] #magnet thickness
    r_m = params['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
    gap = params['gap'] #air gaps
    
    r_r = r_m - gap
    
    b_r = 1.3 #magnetic remanence
    
    nterms = 5 #flux fourier series terms
    mterms = 5 #slot correction fourier series terms
    
    n_p = n_m/2 #number of pole pairs

    Kslm = np.zeros(mterms + 1)
    Bgn = np.zeros(nterms + 1)
    
    mu_r = .001 #magnet relative recoil permeability, aka relative permitivity, mu/mu_0
    #k_rn = 1
    #k_tn = 1
    
    phi = 0
    #flux term layer
    for n in range(-nterms, nterms)
        beta = n*n_p
        
        Ksl = 0
        #slot correction term layer
        for m in range(-mterms, mterms)
            
            Kslm[m + mterms] = 
            
            Ksl += Kslm[m + mterms]*sin(pi*(m + n*(n_m/(2*n_s))))/(pi*(m + n*(n_m/(2*n_s))))
            
        k_mc = (k_rn + 1j*beta*k_tn)/(1 - beta**2)
        
        del_r = (mu_r + 1)*((r_r/r_m)**(2*beta) - (r_s/r_m)**(2*beta)) + (mu_r - 1)*(1 -((r_r/r_m)**(2*beta))*((r_s/r_m)**(2*beta)))
        
        k_a = (k_mc*k_rn/del_r)*((1-(r_r/r_m)**(2*beta))*((beta + 1)*(r_r/r_m)**(2*beta) - (2*beta)*(r_r/r_m)**(beta+1) + beta - 1))
        
        Bgn[n+nterms] = -b_r*k_a*(((r_s/r_m)**(beta-1)) + ((r_s/r_m)**(2*beta))*((r_m/r_s)**(beta+1)))
            
        
        phi += Bgn[n+nterms]*(2*pi*l_st*r_s/n_s)*Ksl*exp(1j*n*theta)
        
        
            
    
    
            
