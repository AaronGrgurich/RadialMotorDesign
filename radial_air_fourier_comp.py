from __future__ import absolute_import
import numpy as np
from math import pi
import openmdao.api as om

class DiffAssemble(om.ExplicitComponent()):
    def setup(self):
        self.add_input('n_m', 16, desc='number of poles')
        self.add_input('sta_ir', .51, desc='stator inner radius')
        self.add_input('rot_or', .50, desc='rotor outer radius w/ magnets')
        self.add_input('t_mag', .03, desc='magnet thickness')
        self.add_input('b_r', 1.3, desc='magnetic remanence')
        self.add_input('mu_r', 1.05, desc='relative recoil permitivity')
        self.add_input('alpha', .85, desc='magnet fraction')
        
        self.add_output('x', desc='PDE solution coefficients')
    
    def compute(self, inputs, outputs):
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
        
        #invert matrix
        bmat_inv = np.linalg.inv(bmat)
        
        #solve system using inverse of system matrix
        outputs['x'] = bmat_inv.dot(rhs)

class MotorAirFourierComp(om.Group()):
    
    def setup(self):
        self.add_subsystem('DiffAssemble', DiffAssemble())
        self.add_subsystem('DiffLinearSolve', DiffLinearSolve())
        self.add_subsystem('BarnComp', BarnComp())
        self.add_subsystem('CorrectSlot', CorrectSlot())
        self.add_subsystem()
        
    
