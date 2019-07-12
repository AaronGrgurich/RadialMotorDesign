'''
This function computes magnetic flux in the stator teeth of a motor, described using a Fourier series with respect to angular position theta

Taken from Brushless Permanent Magnet Motor Design, Second Edition by Dr. Duane Hanselman, Chapter 7
Magnetic Flux Density Coefficient formula taken from Appendix B, pg 349
'''
import numpy as np
from math import pi, sin, cos, exp
import matplotlib.pyplot as plt

def toothflux(theta, nterms, params, Bn):
    n_m = params['n_m'] #number of poles
    n_s = params['n_s'] #number of Slots
    l_st = params['l_st'] #axial motor length
    k_st = params['k_st'] #lamination stacking factor
    w_tb = params['w_tb'] #tooth body width
    r_s = params['sta_ir'] #stator inner radius
    t_mag = params['t_mag'] #magnet thickness
    r_m = params['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
    gap = params['gap'] #air gaps
    t_mag = params['t_mag'] #magnet thickness
    tt = params['tt']
    ts = params['ts']
    alpha = 1/30 #magnetic fraction, fraction of rotor circumference occupied by one magnet
    
    r_r = r_m - t_mag
    
    b_r = 1.3 #magnetic remanence
    
    #nterms = 10 #flux fourier series terms
    mterms = 100 #slot correction fourier series terms
    
    n_p = n_m/2 #number of pole pairs
    
    

    Kslm = np.zeros(2*mterms + 1)
    Bgn = np.zeros(2*nterms + 1)
    phi = np.zeros(2*nterms + 1)
    Bn = np.zeros(2*nterms + 1)
    
    mu_r = .001 #magnet relative recoil permeability, aka relative permitivity, mu/mu_0
    
    phi = 0
    #flux term layer
    for n in range(-nterms, nterms+1):
        beta = n*n_p
        
        slotsum = 0
        
        ###slot correction factor function, to be used to find 
        #first determine g(theta)/g
        #gratio = gfunc(theta, gap, r_s, n_m, tt, ts)
        
        #Ksl = kfunc(theta, gap, r_s, n_m, tt, ts, mu_r, l_m)
        
        #fourier transform to obtain coeffificients in Kslm
        #compute using Riemann sum
        
        Kslm = FourierCoeffsNumpy(kfunc, T, mterms,  gap, r_s, n_m, tt, ts, mu_r, t_mag)
        
        #slot correction term layer
        if n%2 != 0:
            for m in range(-mterms, mterms + 1):

                slotsum += Kslm[m + mterms]*sin(pi*(m + n*(n_m/(2*n_s))))/(pi*(m + n*(n_m/(2*n_s))))
        
        else:
            slotsum = 0

        #
        # add k_rn and k_tn here based on magnetic field orientation of halbach array
        #
        ###for now, assume radial orientation
        k_rn = RadialR(n, alpha)
        k_tn = RadialT(n, alpha)
        
        k_mc = (k_rn + 1j*beta*k_tn)/(1 - beta**2)
        
        del_r = (mu_r + 1)*((r_r/r_m)**(2*beta) - (r_s/r_m)**(2*beta)) + (mu_r - 1)*(1 -((r_r/r_m)**(2*beta))*((r_s/r_m)**(2*beta)))
        #print(del_r)
        if n != 0:
            k_a = (k_mc*k_rn/del_r)*((1-(r_r/r_m)**(2*beta))*((beta + 1)*(r_r/r_m)**(2*beta) - (2*beta)*(r_r/r_m)**(beta+1) + beta - 1))
        else:
            k_a = 0
        
        k_a = k_a.real
        print(n)
        Bgn[n+nterms] = -b_r*k_a*(((r_s/r_m)**(beta-1)) + ((r_s/r_m)**(2*beta))*((r_m/r_s)**(beta+1)))
                    
        Bn[n+nterms] = Bgn[n+nterms]*(2*pi*l_st*r_s/n_s)*slotsum/(k_st*l_st*w_tb)
        print(Bn[n+nterms])
    return Bn
    
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
            gratio[x] = 1 + pi*r_s*(theta[x]-tt/2)/(n_m*gap)
        
        if theta[x] <= -tt/2 and theta[x] >= -ts/2:
            gratio[x] = 1 - pi*r_s*(theta[x]+tt/2)/(n_m*gap)
    
    
    return gratio
    
def RadialR(n, alpha):
    
    if n%2 != 0:
        return alpha*(sin(n*alpha*pi/2))/(n*alpha*pi/2)
    
    if n%2 == 0:
        return 0

def RadialT(n, alpha):
    return 0

def FourierCoeffsNumpy(func, T, N,  gap, r_s, n_m, tt, ts, mu_r, t_mag):
    #func: one-dimensional function of interest
    #T: period of function, will compute from center
    #N: number of terms 1 and onward, this computes N + 1 coefficients
    
    fsample = 2*N 
    t = np.linspace(-T/2, T/2, fsample + 1, endpoint = False)
    y = np.fft.rfft(func(t, gap, r_s, n_m, tt, ts, mu_r, t_mag))/fsample
    
    c = y
    for k in range(1, N + 1):
        c = np.insert(c, 0, np.conj(y[k]))
    
    c = c.real
    return c

if __name__ == '__main__':

    gap = .001
    r_s = .07
    n_m = 20
    n_s = 24
    ts = 2*pi/n_s #slot to slot angle 
    tt = .6*pi/10 #tooth width angle
    mu_r = 1
    t_mag = .003
    l_st = .008
    k_st = 1
    w_tb = .003
    rot_or = .05
    
    params = {
        'gap':gap,
        'n_m':n_m,
        'n_s':n_s,
        'tt':tt,
        'ts':ts,
        'mu_r':mu_r,
        't_mag':t_mag,
        'sta_ir':r_s,
        'l_st':l_st,
        'k_st':k_st,
        'w_tb':w_tb,
        'rot_or':(r_s-gap)
    }
    
    theta = 0#np.linspace(-ts/2,ts/2)
    
    T = ts
    N = 40
    
    # x = kfunc(theta, gap, r_s, n_m, tt, ts, mu_r, t_mag)
    # y = FourierCoeffsNumpy(kfunc, T, N, gap, r_s, n_m, tt, ts, mu_r, t_mag)
    
    Bn = np.zeros(2*N + 1)

    
    Bn = toothflux(theta, N, params, Bn)
    
    #print(Bn)
    
    Brn = np.concatenate((Bn[40:], Bn[0:39]))
    #print(Brn)
    
    i = np.fft.ifft(Brn)
    
    #print(i)
    
    # print(x)
    # print(y)
