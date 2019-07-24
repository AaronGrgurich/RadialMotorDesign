''' 
Helps to compute the Fourier coefficients of the stator tooth flux density by computing a correction factor for the slots

Based on the radial air gap solution as solved in bmatrix.py

Based on section 7.3 of 'Brushless Permanent Magnet Motor Design' by Duane P. Hanselman
'''


import numpy as np
import matplotlib.pyplot as plt

# Compute slot correction Fourier terms using model presented in Chapter 7
def slot_correction(M, params):
    n_p = params['n_p'] #number of pole pairs
    k_st = params['k_st'] #lamination stacking factor
    w_tb = params['w_tb'] #tooth body width
    r_s = params['r_s'] #stator inner radius
    r_m = params['r_m'] #rotor outer radius (INCLUDING MAGNETS)
    t_mag = params['t_mag'] #magnet thickness
    tt = params['tt'] #max angle occupied by stator tooth
    ts = params['ts'] #angle of separation between stator teeth
    mu_r = params['mu_r'] #magnetic relative recoil permeability
    
    gap = r_s - r_m #air gap thickness
    n_m = n_p*2 #number of poles
    T = ts #period of solution, equivalent to the angle of separation between teeth
    
    
    Kslm = FourierCoeffsNumpy(kfunc, T, M, gap, r_s, n_m, tt, ts, mu_r, t_mag)
        
        
    return Kslm

# Compute Fouerier coefficients of a real valued function, arrange in linear order
def FourierCoeffsNumpy(func, T, N, gap, r_s, n_m, tt, ts, mu_r, t_mag):
    
    fsample = 2*N 
    t = np.linspace(-T/2, T/2, fsample + 2, endpoint = False)
    y = np.fft.rfft(func(t, gap, r_s, n_m, tt, ts, mu_r, t_mag))/fsample
    
    c = y
    for k in range(1, N + 1):
        c = np.insert(c, 0, np.conj(y[k]))
    
    c = c.real
    return c

# Equation 7.9
def kfunc(theta, gap, r_s, n_m, tt, ts, mu_r, t_mag):
    
    gratio = gfunc(theta, gap, r_s, n_m, tt, ts)
    
    Ksl = (1 + t_mag/(gap*mu_r))/(gratio + t_mag/(gap*mu_r))
    
    return Ksl

    
# Equation 7.11
def gfunc(theta, gap, r_s, n_m, tt, ts):
    
    gratio = np.zeros(len(theta))
    for x in range(0, len(theta)):
        if abs(theta[x]) <= tt/2:
            gratio[x] = 1
        
        if theta[x] <= ts/2 and theta[x] >= tt/2:
            gratio[x] = 1 + np.pi*r_s*(theta[x]-tt/2)/(n_m*gap)
        
        if theta[x] <= -tt/2 and theta[x] >= -ts/2:
            gratio[x] = 1 - np.pi*r_s*(theta[x]+tt/2)/(n_m*gap)
    
    
    return gratio

if __name__ == '__main__':
    M = 30
    K_slm = np.zeros(M)

    #for m in range(0, M):
    K_slm = slot_correction(M)
    #print(K_slm)
    #plt.plot(K_slm)
    plt.show()
