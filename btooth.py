'''
This function computes magnetic flux in the stator teeth of a motor, described using a Fourier series with respect to angular position theta

Taken from Brushless Permanent Magnet Motor Design, Second Edition by Dr. Duane Hanselman, Chapter 7
Magnetic Flux Density Coefficient formula taken from Appendix B, pg 349
'''


import numpy as np
import matplotlib.pyplot as plt
import bmatrix as bm
import slotcorrection as sl

def B_tooth(N, M, params):
    n_s = params['n_s'] #number of Slots
    n_p = params['n_p'] #number of pole pairs
    k_st = params['k_st'] #lamination stacking factor
    w_tb = params['w_tb'] #tooth body width
    r_s = params['r_s'] #stator inner radius
    t_mag = params['t_mag'] #magnet thickness
    r_m = params['r_m'] #rotor outer radius (INCLUDING MAGNETS)
    r_r = params['r_r']
    b_r = params['b_r']
    tt = params['tt']
    ts = params['ts']
    mu_r = params['mu_r']
    alpha = params['alpha']
    
    n_m = n_p*2
    B_gn = np.zeros(N)
    B_tn = np.zeros(N)

    K_slm = sl.slot_correction(M, params)

    for n in range(1, N):
    
        b_f = bm.B_fourier_terms(n, params)
        B_gn[n] = bm.B_arn(b_f, n, params, r_s)
        slotsum = 0
        #From Chapter 7 of Hanselman Book 
        if n%2 != 0:
            for m in range(-M, M + 1):
                slotsum += K_slm[m + M]*np.sinc(np.pi*(m + n*(n_m/(2*n_s))))#/(np.pi*(m + n*(n_m/(2*n_s))))

        else:
            slotsum = 0
        #print(slotsum)
            
        B_tn[n] = B_gn[n]*(2*np.pi*r_s/(n_s*k_st*w_tb))*slotsum 
    print(B_gn)    
    return B_tn
    
if __name__ == '__main__':
    
    gap = .001
    r_s = .051
    n_p = 8
    n_s = 24
    ts = 2*np.pi/n_s #slot to slot angle 
    tt = .6*np.pi/10 #tooth width angle
    mu_r = 1.05
    t_mag = .003
    l_st = .008
    k_st = 1
    w_tb = .005
    alpha = .85
    b_r = 1.3
        
    params = {
        'gap':gap,
        'r_s':r_s,
        'n_p':n_p,
        'n_s':n_s,
        'tt':tt,
        'ts':ts,
        'mu_r':mu_r,
        't_mag':t_mag,
        'sta_ir':r_s,
        'l_st':l_st,
        'k_st':k_st,
        'w_tb':w_tb,
        'r_m':(r_s - gap),
        'r_r':(r_s - gap - t_mag),
        'alpha':alpha,
        'b_r':b_r
    }
        
    B_tn = B_tooth(17, 20, params)
    
    i = np.fft.irfft(B_tn)
    print(i)
    plt.plot(i)
    plt.show()
    
