import openmdao.api as om
import os
import DynamicModel as dmm
import numpy as np
import scipy.io as sio
import numpy.matlib as npm
from scipy import sparse  
import time
import scipy.linalg as spla

# from LinFuncs_L4T0H1 import LF
import LinFuncs_L4T0H1 as LF

name = 'L4T0H1'

np.random.seed(1242)
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['Vert']

name = 'L4T0H1_NORM'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['V_NRM']

def comm_mat(m,n):
    # determine permutation applied by K
    w = np.arange(m*n).reshape((m,n),order='F').T.ravel(order='F')

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m*n)[w,:]


#%% Equilibrium State
class getStateEquilibrium(om.ImplicitComponent):
    
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nx', types=int, default = 2)
                


    def setup(self):  
        Np = self.options['Np']
        Nx = self.options['Nx']
        
        self.add_input(name='rho', shape=(Np,1))   
        
        self.add_output(name='xe', shape=(Nx,1))
        
        d_Beq = LF.calcBeq()
        Beq = np.zeros(d_Beq['sz'])
        Beq[[d_Beq['r']],[d_Beq['c']]]=d_Beq['val']
        self.Beq = Beq


        
    def setup_partials(self):
        self.declare_partials('xe', '*',method='exact')
        
        
        
    def apply_nonlinear(self,inputs,outputs,res):
        rho   = inputs['rho']
        xe  = outputs['xe']
        
        Aeq = LF.calcAeq(rho[:,0])  
        Beq = self.Beq
        
        res['xe'] = Aeq@xe-Beq
        
        
        
    def solve_nonlinear(self, inputs, outputs):
        rho   = inputs['rho']
        # xe  = outputs['xe']
        
        Aeq = LF.calcAeq(rho[:,0])  
        Beq = self.Beq
        
        outputs['xe'] = np.linalg.inv(Aeq)@Beq
        
        
    
    def linearize(self,inputs,outputs,J):
        # tic = time.time()
        rho   = inputs['rho']
        xe  = outputs['xe']
        
        Aeq = LF.calcAeq(rho[:,0])  
        
        J['xe','xe'] = Aeq
        J['xe','rho'] = LF.calcdXe_dp(xe)
        # toc = time.time(); print('Partials Time:', toc-tic)




#%% All SS matrics
class getSSMatrix(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nx', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Na', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
    
    
    def setup(self):
        Np = self.options['Np']
        Nx = self.options['Nx']
        Nd = self.options['Nd']
        Na = self.options['Na']
        Nld = self.options['Nld']
        
        self.add_input(name='rho', shape=(Np,1))   
        self.add_input(name='xe', shape=(Nx,1))
        self.add_input(name='R', shape=(Np,))
        
        self.add_output('ue', shape=(Np,1))
        self.add_output('A', shape=(Nd,Nd))
        # self.add_output('B', shape=(Nd,Np))
        self.add_output('F', shape=(Np,Np-Nld))
        self.add_output('FRF', shape=(Np-Nld,Np-Nld))
        self.add_output('BF', shape=(Nd,Np-Nld))
        
        
        d_dAdd_dp = LF.calcdAdd_dp()
        self.dAdd_dp = np.zeros(d_dAdd_dp['sz'])
        self.dAdd_dp[[d_dAdd_dp['r']],[d_dAdd_dp['c']]]=d_dAdd_dp['val']
        
        d_dAda_dp = LF.calcdAda_dp()
        self.dAda_dp = np.zeros(d_dAda_dp['sz'])
        self.dAda_dp[[d_dAda_dp['r']],[d_dAda_dp['c']]]=d_dAda_dp['val']
        
        d_dAad_dp = LF.calcdAad_dp()
        self.dAad_dp = np.zeros(d_dAad_dp['sz'])
        self.dAad_dp[[d_dAad_dp['r']],[d_dAad_dp['c']]]=d_dAad_dp['val']
        
        d_dAaa_dp = LF.calcdAaa_dp()
        self.dAaa_dp = np.zeros(d_dAaa_dp['sz'])
        self.dAaa_dp[[d_dAaa_dp['r']],[d_dAaa_dp['c']]]=d_dAaa_dp['val']
        
        self.Kaa = comm_mat(Na, Na)
        self.Kad = comm_mat(Na, Nd)
        self.Kda = comm_mat(Nd, Na)
        
    def setup_partials(self):
        self.declare_partials('A', 'rho', method='exact')
        # self.declare_partials('B', ['rho','xe'], method='exact')
        self.declare_partials('F', 'rho', method='exact')
        self.declare_partials('FRF', ['rho','R'], method='exact')
        self.declare_partials('BF', ['rho','xe'], method='exact')
        self.declare_partials('ue', 'rho', method='exact')



    def compute(self, inputs, outputs):
        rho   = inputs['rho']
        xe  = inputs['xe']
        R  = np.diag(inputs['R'])
        
        Add = LF.calcAdd(rho)
        Ada = LF.calcAda(rho)
        Aad = LF.calcAad(rho)
        Aaa = LF.calcAaa(rho)
        Bdu = LF.calcBdu(rho,xe)
        Bau = LF.calcBau(rho,xe)
        F = LF.calcF(rho)
        ue = LF.calcUe(rho)
        
        Aaa_inv = np.linalg.inv(Aaa)
        X = Ada@-Aaa_inv
        
        A = Add + X@Aad
        B = Bdu + X@Bau
        
        outputs['A'] = A
        outputs['ue'] = ue
        # outputs['B'] = B
        outputs['F'] = F
        outputs['FRF'] = F.T@R@F
        outputs['BF'] = B@F
        
        
        
    def compute_partials(self, inputs, partials):
        # tic = time.time()
        Np = self.options['Np']
        Nd = self.options['Nd']
        Na = self.options['Na']
        Nld = self.options['Nld']
        
        rho   = inputs['rho']
        xe  = inputs['xe']
        R  = inputs['R']
       
        dAdd_dp = self.dAdd_dp
        dAda_dp = self.dAda_dp
        dAad_dp = self.dAad_dp
        dAaa_dp = self.dAaa_dp
        Ia = np.eye(Na)
        Id = np.eye(Nd)
        Ip = np.eye(Np)
        Ild = np.eye(Nld)
        Ili = np.eye(Np-Nld)
        
        # Add = calcAdd(rho)
        Ada = LF.calcAda(rho)
        Aad = LF.calcAad(rho)
        Aaa = LF.calcAaa(rho)
        Bdu = LF.calcBdu(rho,xe)
        Bau = LF.calcBau(rho,xe)
        F = LF.calcF(rho)
        Aaa_inv = np.linalg.inv(Aaa)
        X = Ada@-Aaa_inv
        
        dBdu_dp = LF.calcdBdu_dp(rho,xe)
        dBdu_dxe = LF.calcdBdu_dxe(rho,xe)
        dBau_dp = LF.calcdBau_dp(rho,xe)
        dBau_dxe = LF.calcdBau_dxe(rho,xe)        
        dF_dp = LF.calcdF_dp(rho)        
        dFRF_dp = LF.calcdFRF_dp(rho,R)        
        dFRF_dr = LF.calcdFRF_dr(rho)        
        dUe_dr = LF.calcdUe_dp(rho)        
         
        B = Bdu + X@Bau
        dAaa_inv_dp=-np.kron(Aaa_inv,Ia)@np.kron(Ia,Aaa_inv.T)@dAaa_dp
        dX_dp = -(np.kron(Id,Aaa_inv.T)@dAda_dp + np.kron(Ada,Ia)@dAaa_inv_dp)

        # compute the necessary partials
        dA_dp = dAdd_dp + np.kron(Id,Aad.T)@dX_dp + np.kron(X,Id)@dAad_dp
        dB_dp = dBdu_dp + np.kron(Id,Bau.T)@dX_dp + np.kron(X,Ip)@dBau_dp
        dB_dxe = dBdu_dxe + np.kron(X,Ip)@dBau_dxe
        dBF_dp = np.kron(Ild,F.T)@dB_dp + np.kron(B,Ili)@dF_dp
        dBF_dxe = np.kron(Ild,F.T)@dB_dxe

        partials['A','rho'] = dA_dp
        partials['ue','rho'] = dUe_dr
        # partials['B','rho'] = dB_dp
        # partials['B','xe'] = dB_dxe
        partials['F','rho'] = dF_dp
        partials['FRF','rho'] = dFRF_dp
        partials['FRF','R'] = dFRF_dr
        partials['BF','rho'] = dBF_dp
        partials['BF','xe'] = dBF_dxe
        # toc = time.time(); print('Partials Time:', toc-tic)


#%% LQR Synthesis
class RiccatiTune(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
        


    def setup(self): 
        Np = self.options['Np']
        Nd = self.options['Nd']
        Nld = self.options['Nld']
        
        self.add_input(name='A', shape=(Nd,Nd))
        self.add_input(name='BF', shape=(Nd,Np-Nld))     
        self.add_input(name='Q', shape=(Nd,Nd))     
        self.add_input(name='FRF', shape=(Np-Nld,Np-Nld))     
        
        
        self.add_output(name='P', shape=(Nd,Nd))        
         
        
        
    def setup_partials(self):        
        self.declare_partials(of='P', wrt=['*'],method='exact')
             
        
        
    def apply_nonlinear(self,inputs,outputs,res):
        A   = inputs['A']
        B   = inputs['BF']
        Q   = inputs['Q']
        R   = inputs['FRF']
        P   = outputs['P']
                
        eqn = P@A + A.T@P + Q - P@B@np.linalg.inv(R)@B.T@P
    
        res['P'] = eqn
        
        
        
    def solve_nonlinear(self,inputs,outputs):
        A   = inputs['A']
        B   = inputs['BF']
        Q   = inputs['Q']
        R   = inputs['FRF']
        
        outputs['P'] = spla.solve_continuous_are(A, B, Q, R)

        

    def linearize(self,inputs,outputs,J):
        # tic = time.time()
        Np = self.options['Np']
        Nd = self.options['Nd']
        Nld = self.options['Nld']
        Nli = Np - Nld
        
        A   = inputs['A']
        B   = inputs['BF']
        # Q   = inputs['Q']
        R   = inputs['FRF']
        P   = outputs['P']
        
        Id = np.eye(Nd)
        Ili = np.eye(Nli)
        Kdd = comm_mat(Nd,Nd)
        Klid = comm_mat(Nli,Nd)
        Rinv = np.linalg.inv(R)
        
        dP_dRbar = np.kron(P@B,Id)@np.kron(Ili,(B.T@P).T)@np.kron(Rinv,Ili)@np.kron(Ili,Rinv.T)
        J['P','A'] = np.kron(P,Id) + Kdd@np.kron(P.T,Id)
        J['P','BF'] = -np.kron(P,Id)@np.kron(Id,P.T)@(np.kron(Id,(Rinv@B.T).T)+np.kron(B@Rinv,Id)@Klid)
        J['P','Q'] = np.eye(Nd*Nd)
        J['P','FRF'] = dP_dRbar
        J['P','P'] = np.kron(Id,A.T) + np.kron(A.T,Id) - np.kron(Id,(B@Rinv@B.T@P).T) - np.kron(P@B@Rinv@B.T,Id)
        # toc = time.time(); print('Partials Time:', toc-tic)


#%% compute the control Gain
class CtrlGain(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)   
        
        
        
    def setup(self):  
        Np = self.options['Np']
        Nd = self.options['Nd']
        Nld = self.options['Nld']
        
        self.add_input(name='BF', shape=(Nd,Np-Nld))     
        self.add_input(name='FRF', shape=(Np-Nld,Np-Nld))     
        self.add_input(name='P', shape=(Nd,Nd))    
        
        self.add_output(name='K', shape=(Np-Nld,Nd))        

    
        
    def setup_partials(self):        
        self.declare_partials(of='K', wrt=['*'],method='exact')
        
    def compute(self, inputs, outputs):
        B   = inputs['BF']
        R   = inputs['FRF']
        P   = inputs['P'] 
        
        
        outputs['K'] = np.linalg.inv(R)@B.T@P
        
    def compute_partials(self, inputs, partials):
        # tic = time.time()
        B   = inputs['BF']
        R   = inputs['FRF']
        P   = inputs['P']
        
        Np = self.options['Np']
        Nd = self.options['Nd']
        Nld = self.options['Nld']
        Nli = Np - Nld
        
        Id = np.eye(Nd)
        Ip = np.eye(Np)
        Ili = np.eye(Nli)
        Klid = comm_mat(Nli,Nd)
        Rinv = np.linalg.inv(R)    
        
        partials['K','P'] = np.kron(Rinv@B.T,Id)
        partials['K','BF'] = np.kron(Rinv,Id)@Klid @ np.kron(P.T,Ili)
        partials['K','FRF'] = -np.kron(Ili,(B.T@P).T)@np.kron(Rinv,Ili)@np.kron(Ili,Rinv.T)
        # toc = time.time(); print('Partials Time:', toc-tic)


#%% full synthesis
class CtrlSynth(om.Group):
    
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nx', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Na', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)

    
    
    def setup(self):
        Np = self.options['Np']
        Nx = self.options['Nx']
        Nd = self.options['Nd']
        Na = self.options['Na']
        Nld = self.options['Nld']
        
        self.add_subsystem('EQ', subsys=getStateEquilibrium(Np=Np,Nx=Nx),promotes=['*'])
        self.add_subsystem('SS', subsys=getSSMatrix(Np=Np,Nx=Nx,Nd=Nd,Na=Na,Nld=Nld),promotes=['*'])
        self.add_subsystem('LQR', subsys=RiccatiTune(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])
        self.add_subsystem('Gain', subsys=CtrlGain(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])




#%%

# L = np.arange(682)/sum(np.arange(682))
# rho = np.concatenate((np.array([1]),V@L),axis=0)
# R = np.arange(1,40)     
# Q = np.diag(np.arange(1,8))     

# p = om.Problem()
# p.model.add_subsystem('Test', subsys=CtrlSynth(Np=39,Nx=15,Nd=7,Na=8,Nld=7),promotes=['*'])
# # p.model.nonlinear_solver = om.NewtonSolver(iprint=0,maxiter=10,solve_subsystems=False)
# # p.model.linear_solver    = om.DirectSolver(iprint=0)

# tic = time.time()
# p.setup(force_alloc_complex=False)
# toc = time.time(); print('Setup Time:', toc-tic)

# p['rho'] = rho 
# p['R'] = R 
# # p['Q'] = Q

# tic = time.time()
# p.run_model()  
# toc = time.time(); print('Run Time:', toc-tic)
# tic = time.time()
# a = p.check_partials(method='fd',compact_print=True)
# toc = time.time(); print('Partials Time:', toc-tic)
# om.n2(p)
