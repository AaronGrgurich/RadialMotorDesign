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
    sta_ir = params['sta_ir'] #stator inner radius
    t_mag = params['t_mag'] #magnet thickness
    rot_or = params['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
    gap = params['gap'] #air gaps
    
    nterms = 5 #flux fourier series terms
    mterms = 5 #slot correction fourier series terms
    
    n_p = n_m/2 #number of pole pairs

    Kslm = np.zeros(mterms + 1)
    Bgn = np.zeros(nterms + 1)
    
    phi = 0
    #flux term layer
    for n in range(-nterms, nterms)
        beta = n*n_p
        
        Ksl = 0
        #slot correction term layer
        for m in range(-mterms, mterms)
            
            Kslm[m + mterms] = 
            
            Ksl += Kslm[m + mterms]*sin(pi*(m + n*(n_m/(2*n_s))))/(pi*(m + n*(n_m/(2*n_s))))
            
        
        
        Bgn[n+nterms] = -br*ka*(((sta_ir/rot_or)**(beta-1)) + ((sta_ir/rot_or)**(2*beta))*((rot_or/sta_ir)**(beta+1)))
            
        
        phi += Bgn[n+nterms]*(2*pi*l_st*sta_ir/n_s)*Ksl*exp(1j*n*theta)
        
        
            
    
    
            
