from __future__ import absolute_import
import numpy as np
from math import pi
import openmdao.api as om

N = 4
M = 20

B_tna = np.zeros(N)

#need partials for rot_or, sta_ir, t_mag, alpha, w_tb, tt

class DiffAssemble(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', default=1., desc='index')

    def setup(self):
        self.add_input('n_m', 16, desc='number of poles')
        self.add_input('sta_ir', .51, desc='stator inner radius')
        self.add_input('rot_or', .50, desc='rotor outer radius w/ magnets')
        self.add_input('t_mag', .03, desc='magnet thickness')
        self.add_input('b_r', 1.3, desc='magnetic remanence')
        self.add_input('mu_r', 1.05, desc='relative recoil permitivity')
        self.add_input('alpha', .85, desc='magnet fraction')

        self.add_output('bmat', shape=(3,3), desc='system matrix')
        self.add_output('rhs', shape=3, desc='right hand side')

        self.declare_partials('bmat','sta_ir')
        self.declare_partials('bmat','rot_or')
        self.declare_partials('bmat','t_mag')
        self.declare_partials('bmat','alpha')
        self.declare_partials('rhs','sta_ir')
        self.declare_partials('rhs','rot_or')
        self.declare_partials('rhs','t_mag')
        self.declare_partials('rhs','alpha')

    def compute(self, inputs, outputs):
        n = self.options['n']
        n_p = inputs['n_m']/2 #number of pole pairs
        r_s = inputs['sta_ir'] #stator inner radius
        r_m = inputs['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
        r_r = r_m - inputs['t_mag'] #rotor outer radius
        b_r = inputs['b_r'] #magnetic remanence of stator material
        mu_r = inputs['mu_r'] #magnetic relative recoil permitivity
        alpha = inputs['alpha'] #magnet fraction in rotor

        mu_0 = 0.001

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
        outputs['bmat'] = bmat
        outputs['rhs'] = rhs

    def compute_partials(self, inputs, J):
        n = self.options['n']
        n_p = inputs['n_m']/2 #number of pole pairs
        r_s = inputs['sta_ir'] #stator inner radius
        r_m = inputs['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
        r_r = r_m - inputs['t_mag'] #rotor outer radius
        b_r = inputs['b_r'] #magnetic remanence of stator material
        mu_r = inputs['mu_r'] #magnetic relative recoil permitivity
        alpha = inputs['alpha'] #magnet fraction in rotor

        drrdrot = 1
        drrdtma = -1

        mu_0 = 0.001

        dbmatdsta = np.zeros((3, 3))
        dbmatdrot = np.zeros((3, 3))
        dbmatdtma = np.zeros((3, 3))
        dbmatdalp = np.zeros((3, 3))
        drhsdsta = np.zeros(3)
        drhsdrot = np.zeros(3)
        drhsdtma = np.zeros(3)
        drhsdalp = np.zeros(3)

        beta = n*n_p

        # magnetic profile coefficients
        k_rn = RadialR(n, alpha)
        dkrndalp = RadialRPart(n, alpha)

        k_tn = RadialT(n, alpha)
        dktndalp = 0

        # assume output is real, ignore imaginary part (Equation B.18)
        Cm = b_r*(k_rn)/((1-beta**2)*mu_r*mu_0)
        dCmdkrn = b_r/((1-beta**2)*mu_r*mu_0)

        #Hatheta boundary condition (excluded from linear system)
        Ea = -(r_s)**(2*beta)
        dEadsta = -(2*beta)*(r_s**(2*beta-1))

        #Hmtheta boundary condition
        bmat[0, 1] = (r_r**(-beta-1))
        dbmatdrot[0, 1] = (-beta-1)*(r_r**(-beta-2))*drrdrot
        dbmatdtma[0, 1] = (-beta-1)*(r_r**(-beta-2))*drrdtma

        bmat[0, 2] = (r_r**(beta-1))
        dbmatdrot[0, 2] = (beta-1)*(r_r**(beta-2))*drrdrot
        dbmatdtma[0, 2] = (beta-1)*(r_r**(beta-2))*drrdtma

        rhs[0] = -Cm
        drhsdalp[0] = -1*dCmdkrn*dkrndalp

        #Htheta boundary condition
        bmat[1, 0] = -Ea*(r_m**(-beta-1)) - r_m**(beta - 1)
        dbmatdrot[1, 0] = -Ea*(-beta-1)*(r_m**(-beta-2)) - (beta - 1)*(r_m**(beta - 2))
        dbmatdsta[1, 0] = -(r_m**(-beta-1))*dEadsta

        bmat[1, 1] = r_m**(-beta-1)
        dbmatdrot[1, 1] = (-beta-1)*(r_m**(-beta-2))

        bmat[1, 2] = r_m**(beta-1)
        dbmatdrot[1, 2] = (beta-1)*(r_m**(beta-2))

        rhs[1] = -Cm
        drhsdalp[1] = -1*dCmdkrn*dkrndalp

        #Hr boundary condition
        bmat[2, 0] = (beta/mu_r)*(Ea*(r_m**(-beta-1)) - r_m**(beta - 1))
        dbmatdrot[2, 0] = (beta/mu_r)*(Ea*(-beta-1)*(r_m**(-beta-2)) - (beta - 1)*(r_m**(beta - 2)))
        dbmatdsta[2, 0] = (beta/mu_r)*(r_m**(-beta-1))*dEadsta

        bmat[2, 1] = -beta*(r_m**(-beta-1))
        dbmatdrot[2, 1] = -beta*(-beta-1)*(r_m**(-beta-2))

        bmat[2, 2] = beta*(r_m**(beta-1))
        dbmatdrot[2, 2] = beta*(beta-1)*(r_m**(beta-2))


        rhs[2] = (b_r*k_rn/(mu_r*mu_0)) - Cm
        drhsdalp[2] = (b_r/(mu_r*mu_0))*dkrndalp - 1*dCmdkrn*dkrndalp

        J['bmat','sta_ir'] = dbmatdsta
        J['bmat','rot_or'] = dbmatdrot
        J['bmat','t_mag'] = dbmatdtma
        J['bmat','alpha'] = dbmatdalp
        J['rhs','sta_ir'] = drhsdsta
        J['rhs','rot_or'] = drhsdrot
        J['rhs','t_mag'] = drhsdtma
        J['rhs','alpha'] = drhsdalp


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

def RadialRPart(n, alpha):

    if n%2 != 0:
        return n*(np.pi/2)*np.cos(n*alpha*np.pi/2)/(n*np.pi/2)

    if n%2 == 0:
        return 0

def RadialT(n, alpha):
    return 0

class DiffLinearSolve(om.ExplicitComponent):

    def setup(self):
        self.add_input('bmat', shape=(3,3), desc='system matrix')
        self.add_input('rhs', shape=3, desc='right hand side')

        self.add_output('x', shape=3)


    def compute(self, inputs, outputs):
        bmat = inputs['bmat']
        rhs = inputs['rhs']
        print(bmat)
        bmat_inv = np.linalg.inv(bmat)
        x = bmat_inv.dot(rhs)
        outputs['x'] = x

    # def solve_linear(self, d_outputs, d_residuals, mode):
    #     if mode == 'fwd':
    #         d_outputs['x'] = self.inv_jac * d_residuals['x']
    #     elif mode == 'rev':
    #         d_residuals['x'] = self.inv_jac * d_outputs['x']



class BarnComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', default=1., desc='index')

    def setup(self):
        self.add_input('x', shape=3, desc='assumed solution coeffs')
        self.add_input('sta_ir', .51, desc='stator inner radius')
        self.add_input('n_m', desc='number of poles')

        self.add_output('B_arn')

        self.declare_partials('B_arn','x')
        self.declare_partials('B_arn','sta_ir')
    def compute(self, inputs, outputs):
        x = inputs['x']
        n = self.options['n']
        r_s = inputs['sta_ir'] #stator inner radius
        n_p = inputs['n_m']/2 #number of pole pairs
        mu_0 = 0.001

        Da = -x[0]
        beta = n*n_p
        Ea = -(r_s)**(2*beta)
        b_arn = mu_0*beta*Da*(Ea*(r_s**(-beta-1)) - r_s**(beta-1)) #Equation B.13
        outputs['B_arn'] = b_arn

    def compute_partials(self, inputs, J):
        x = inputs['x']
        n = self.options['n']
        r_s = inputs['sta_ir'] #stator inner radius
        n_p = inputs['n_m']/2 #number of pole pairs
        mu_0 = 0.001

        Da = -x[0]
        dDadx = [-1, 0, 0]
        beta = n*n_p
        Ea = -(r_s)**(2*beta)
        dEadsta = -(2*beta)*(r_s**(2*beta-1))

        b_arn = mu_0*beta*Da*(Ea*(r_s**(-beta-1)) - r_s**(beta-1)) #Equation B.13

        J['B_arn','x'] = mu_0*beta*(Ea*(r_s**(-beta-1)))*dDadx
        J['B_arn','sta_ir'] = mu_0*beta*Da*((r_s**(-beta-1))*dEadsta + Ea*(-beta-1)*(r_s**(-beta-2)) - (beta - 1)*(r_s**(beta-2)))

class SCFactor(om.ExplicitComponent):

    def setup(self):
        self.add_input('n_m', 16, desc='number of poles')
        self.add_input('k_st', desc='lamination stacking factor')
        self.add_input('w_tb', desc='tooth body width')
        self.add_input('sta_ir', .51, desc='stator inner radius')
        self.add_input('rot_or', .50, desc='rotor outer radius w/ magnets')
        self.add_input('t_mag', .03, desc='magnet thickness')
        self.add_input('n_s', 24, desc='number of slots')
        self.add_input('tt', .6*np.pi/10, desc='tooth width angle')
        self.add_input('mu_r', 1.05, desc='relative recoil permitivity')
        self.add_output('K_slm', shape=2*M + 2, desc='slot correction factor coeffs')

    def compute(self, inputs, outputs):
        n_p = inputs['n_m']/2 #number of pole pairs
        k_st = inputs['k_st'] #lamination stacking factor
        w_tb = inputs['w_tb'] #tooth body width
        r_s = inputs['sta_ir'] #stator inner radius
        r_m = inputs['rot_or'] #rotor outer radius (INCLUDING MAGNETS)
        t_mag = inputs['t_mag'] #magnet thickness
        n_s = inputs['n_s'] #max angle occupied by stator tooth
        tt = inputs['tt'] #tooth width angle
        mu_r = inputs['mu_r'] #magnetic relative recoil permeability

        ts = 2*np.pi/n_s
        gap = r_s - r_m #air gap thickness
        T = ts

        outputs['K_slm'] = FourierCoeffsNumpy(kfunc, T, M, gap, r_s, inputs['n_m'], tt, ts, mu_r, t_mag)


    # def compute_partials(self, inputs, outputs):
    #     i = 1

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

class CorrectSlot(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1., desc='index')

    def setup(self):
        self.add_input('K_slm', shape=2*M+2, desc='slot correction coeffs')
        self.add_input('B_arn', desc='B_arn')
        self.add_input('n_m', desc='number of poles')
        self.add_input('n_s', desc='number of slots')
        self.add_input('sta_ir')
        self.add_input('k_st')
        self.add_input('w_tb')

        self.add_output('B_tn', shape=N, desc='corrected B_arn terms')

        self.declare_partials('B_tn','K_slm')
        self.declare_partials('B_tn','B_arn')
        self.declare_partials('B_tn','sta_ir')
        self.declare_partials('B_tn','w_tb')

    def compute(self, inputs, outputs):
        n = self.options['n']
        K_slm = inputs['K_slm']
        B_arn = inputs['B_arn']
        n_m = inputs['n_m'] #number of pole pairs
        n_s = inputs['n_s']
        r_s = inputs['sta_ir']
        k_st = inputs['k_st']
        w_tb = inputs['w_tb']

        slotsum = 0
        # Compute characteristic summation of slot correction factor for the nth Fourier Coefficient
        if n%2 != 0:
            for m in range(-M, M + 1):
                slotsum += K_slm[m + M]*np.sinc(np.pi*(m + n*(n_m/(2*n_s))))#/(np.pi*(m + n*(n_m/(2*n_s))))
        else:
            slotsum = 0

        # Modify air gap solution with slot correction
        B_tn = B_arn*(2*np.pi*r_s/(n_s*k_st*w_tb))*slotsum

        B_tna[n] = B_tn
        outputs['B_tn'] = B_tna

    def compute_partials(self, inputs, J):
        n = self.options['n']
        K_slm = inputs['K_slm']
        B_arn = inputs['B_arn']
        n_m = inputs['n_m'] #number of pole pairs
        n_s = inputs['n_s']
        r_s = inputs['sta_ir']
        k_st = inputs['k_st']
        w_tb = inputs['w_tb']

        dssdksl = np.zeros(2*M + 1)

        if n%2 != 0:
            for m in range(-M, M + 1):
                dssdksl[m + M] = np.sinc(np.pi*(m + n*(n_m/(2*n_s))))#/(np.pi*(m + n*(n_m/(2*n_s))))
        else:
            dssdksl = 0

        dBtndBarn = (2*np.pi*r_s/(n_s*k_st*w_tb))*slotsum
        dBtndksl = B_arn*(2*np.pi*r_s/(n_s*k_st*w_tb))*dssdksl
        dBtndsta = B_arn*(2*np.pi/(n_s*k_st*w_tb))*slotsum
        dBtndwtb = -B_arn*(2*np.pi*r_s/(n_s*k_st*(w_tb**2)))*slotsum

        J['B_tn','K_slm'] = dBtndBarn
        J['B_tn','B_arn'] = dBtndksl
        J['B_tn','sta_ir'] = dBtndsta
        J['B_tn','w_tb'] = dBtndwtb

class MotorAirFourierComp(om.Group):
    def initialize(self):
        self.options.declare('ng', default=1., desc='index')

    def setup(self):
        ng = self.options['ng']
        self.add_subsystem('DiffAssemble', DiffAssemble(n = ng), promotes_inputs=['n_m','sta_ir','rot_or','t_mag','b_r','mu_r','alpha'], promotes_outputs=['bmat','rhs'])
        self.add_subsystem('DiffLinearSolve', DiffLinearSolve(), promotes_inputs=['bmat','rhs'], promotes_outputs=['x'])
        self.add_subsystem('BarnComp', BarnComp(n = ng), promotes_inputs=['x','sta_ir','n_m'], promotes_outputs=['B_arn'])
        self.add_subsystem('CorrectSlot', CorrectSlot(n = ng), promotes_inputs=['K_slm','B_arn','n_m','n_s','sta_ir', 'k_st', 'w_tb'], promotes_outputs=['B_tn'])


class InvFourier(om.ExplicitComponent):

    def setup(self):
        self.add_input('B_tn', shape=N)

        self.add_output('B_t', shape=(N-1)*2)

        self.declare_partials('B_t','B_tn', method='fd')
    def compute(self, inputs, outputs):
        print(inputs['B_tn'])
        B_t = np.fft.irfft(inputs['B_tn'])
        outputs['B_t'] = B_t

class MaxMeanSquared(om.ExplicitComponent):

    def setup(self):
        self.add_input('B_t', shape=(N-1)*2)
        self.add_input('f')

        self.add_output('B_pk')
        self.add_output('B_msq')

        self.declare_partials('B_pk','B_t')
        self.declare_partials('B_msq','B_t')

    def compute(self, inputs, outputs):
        B_t = inputs['B_t']
        f = inputs['f']

        omega = f*(2*np.pi/60)

        # peak magnetic flux density is the largest data point from an inverse fourier transform
        B_pk = max(B_t)

        Nl = len(B_t)
        #assume B_n is real, compute mean-square value of dB/dt
        B_msq = 0
        for t in range(1, Nl):
            B_msq += 2*((t*omega*B_t[t])**2)

        outputs['B_pk'] = B_pk
        outputs['B_msq'] = B_msq

    def compute_partials(self, inputs, outputs):
        B_t = inputs['B_t']
        f = inputs['f']

        omega = f*(2*np.pi/60)

        dBpkdBt = np.zeros(len(B_t))
        dBpkdBt[B_t.index(max(B_t))] = 1

        dBmsqdBt = np.zeros(len(B_t))
        for t in range(1, Nl):
            dBmsqdBt[t] = 4*(((t*omega)**2)*B_t[t])

        J['B_pk', 'B_t'] = dBpkdBt
        J['B_msq', 'B_t'] = dBmsqdBt

class CoreLoss(om.ExplicitComponent):

    def setup(self):
        self.add_input('B_pk')
        self.add_input('B_msq')
        self.add_input('f')

        self.add_output('L_core')

        self.declare_partials('L_core', 'B_pk')
        self.declare_partials('L_core', 'B_msq')

    def compute(self, inputs, outputs):
        B_pk = inputs['B_pk']
        B_msq = inputs['B_msq']
        f = inputs['f']
        # k_h, k_e, n, m are material constants and need to be fitted to real material data
        n = 2
        m = 0.01
        k_h = 0.5
        k_e = 0.5

        # compute core loss
        L_core = k_h*f*(B_pk**(n + m*B_pk)) + B_msq*k_e/(2*(np.pi**2))

        outputs['L_core'] = L_core

    def compute_partials(self, inputs, J):
        B_pk = inputs['B_pk']
        B_msq = inputs['B_msq']
        f = inputs['f']

        n = 2
        m = 0.01
        k_h = 0.5
        k_e = 0.5

        J['L_core', 'B_msq'] = k_e/(2*(np.pi**2))
        J['L_core', 'B_pk'] = k_h*f*(B_pk**(n + m*B_pk))*(m*np.ln(B_pk) + (n + m*B_pk)/B_pk)

if __name__ == '__main__':
    p = om.Problem()
    model = p.model

    ind = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

    ind.add_output('t_mag', val=.003)#, units='m')    # Magnet thickness
    ind.add_output('n_s', val=24)                # Number of Slots
    ind.add_output('n_m', val=16)                # Number of poles
    ind.add_output('rot_or', val = 0.050)#, units='m')
    ind.add_output('sta_ir', val = 0.051)#, units='m')
    ind.add_output('f', val = 5400/20)#, units='rpm')
    ind.add_output('b_r', val = 1.3)
    ind.add_output('mu_r', val = 1.05)
    ind.add_output('alpha', val = .85)
    ind.add_output('k_st', val = 1)
    ind.add_output('w_tb', val = .005)
    ind.add_output('tt', val = .6*np.pi/10)

    #model.nonlinear_solver = om.NewtonSolver()
    #model.linear_solver = om.DirectSolver()
    model.add_subsystem('SCFactor', SCFactor(), promotes_inputs=['n_m', 'k_st', 'w_tb', 'sta_ir', 'rot_or', 't_mag', 'n_s', 'tt', 'mu_r'], promotes_outputs=['K_slm'])

    for nl in range(1, N):
        model.add_subsystem('MotorAirFourierComp{}'.format(nl), MotorAirFourierComp(ng=nl), promotes_inputs=['n_m','sta_ir','rot_or','t_mag','b_r','mu_r','alpha', 'K_slm', 'n_s','w_tb','k_st'])

    model.add_subsystem('InvFourier', InvFourier(), promotes_outputs=['B_t'])

    model.connect('MotorAirFourierComp{}.B_tn'.format(nl), 'InvFourier.B_tn')

    model.add_subsystem('MaxMeanSquared', MaxMeanSquared(), promotes_inputs=['B_t', 'f'], promotes_outputs=['B_pk', 'B_msq'])
    model.add_subsystem('CoreLoss', CoreLoss(), promotes_inputs=['B_pk', 'B_msq', 'f'], promotes_outputs=['L_core'])

    p.setup()

    p.run_model()

    print(p.get_val('L_core'))
