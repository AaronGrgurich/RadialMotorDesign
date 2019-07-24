'''
Compute core loss for either stator teeth or stator yoke

Depends on 4 constants characteristic of the lamination material, in this case Hiperco 50
Will need to determine them, either through data sheets or experimentation

From section 9.4 of Hanselman book
'''

import numpy as np

def core_loss(B_n, params):
    f = params['f'] # motor speed in rpm
    
    # convert motor speed to rad/s
    omega = f*(2*np.pi/60)
    
    # peak magnetic flux density is the largest data point from an inverse fourier transform
    B_pk = max(B_n)
    
    N = len(B_n)
    #assume B_n is real, compute mean-square value of dB/dt
    B_msq = 0
    for t in range(1, N):
        B_msq += 2*((t*omega*B_n[t])**2)
        
    # k_h, k_e, n, m are material constants and need to be fitted to real material data
    n = 2
    m = 0.01
    k_h = 0.5
    k_e = 0.5
        
    # compute core loss
    L_core = k_h*f*(B_pk**(n + m*B_pk)) + B_msq*k_e/(2*(np.pi**2))
    
    return L_core
