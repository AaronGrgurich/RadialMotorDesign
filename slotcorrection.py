
import numpy as np
import matplotlib.pyplot as plt



# n_p = 8
# n_s = 24
# r_s = .051
# r_m = .05
# r_r = .045
# b_r = 1.3
# mu_r = 1.05
# mu_0 = 0.001 #value doesn't matter
# alpha = .85
# n_m = 2*n_p
# ts = 2*np.pi/n_s #slot to slot angle 
# tt = .6*np.pi/10 #tooth width angle
# t_mag = r_m - r_r
# gap = r_s - r_m

def slot_correction(M, params):
    n_p = params['n_p'] #number of pole pairs
    k_st = params['k_st'] #lamination stacking factor
    w_tb = params['w_tb'] #tooth body width
    r_s = params['r_s'] #stator inner radius
    r_m = params['r_m'] #rotor outer radius (INCLUDING MAGNETS)
    t_mag = params['t_mag'] #magnet thickness
    tt = params['tt']
    ts = params['ts']
    mu_r = params['mu_r']
    
    gap = r_s - r_m
    n_m = n_p*2
    T = ts
    Kslm = FourierCoeffsNumpy(kfunc, T, M, gap, r_s, n_m, tt, ts, mu_r, t_mag)
        
        
    return Kslm

def FourierCoeffsNumpy(func, T, N, gap, r_s, n_m, tt, ts, mu_r, t_mag):
    #func: one-dimensional function of interest
    #T: period of function, will compute from center
    #N: number of terms 1 and onward, this computes N + 1 coefficients
    
    fsample = 2*N 
    t = np.linspace(-T/2, T/2, fsample + 2, endpoint = False)
    y = np.fft.rfft(func(t, gap, r_s, n_m, tt, ts, mu_r, t_mag))/fsample
    
    c = y
    for k in range(1, N + 1):
        c = np.insert(c, 0, np.conj(y[k]))
    
    c = c.real
    return c

def kfunc(theta, gap, r_s, n_m, tt, ts, mu_r, t_mag):
    
    gratio = gfunc(theta, gap, r_s, n_m, tt, ts)
    
    Ksl = (1 + t_mag/(gap*mu_r))/(gratio + t_mag/(gap*mu_r))
    
    return Ksl

    
        
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
