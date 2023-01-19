import openmdao.api as om
import os
import DynamicModel as dmm
import numpy as np
import scipy.io as sio
import numpy.matlib as npm
from scipy import sparse  
import time

# name = 'L4T1H1'

np.random.seed(1242)


name = 'L4T0H1'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['Vert']

name = 'L4T0H1_NORM'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['V_NRM']

#%%
class VertexSet1(om.ExplicitComponent):
    
    def initialize(self):        
        self.options.declare('V', types=np.ndarray, desc='Vertices')
        self.options.declare('order', types=int, desc='Penalty Order')
 
    
 
    def setup(self):
        V = self.options['V']
        (nr,nc) = np.shape(V)  
        
        self.add_input('L',shape=(nc,1),desc='lambda vector')
        self.add_input('rho',shape=(nr+1,1), desc='desvars')   
        
        self.add_output('del',shape=(nr,1), desc='Set Constraint')        
        self.add_output('sumL',val=1, desc='Sum Constraint')
        self.add_output('p',shape=(nr+1,1), desc='Penalty Vars')


        
    def setup_partials(self):
        V = self.options['V']
        (nr,nc) = np.shape(V)
        nz = np.nonzero(V)
        Vnz = V[nz[0],nz[1]]
        
        self.declare_partials('del', 'rho', 
                              rows=np.arange(nr), cols=np.arange(1,nr+1), 
                              val=np.ones([nr]), method='exact')
        
        self.declare_partials('del', 'L', rows=nz[0], cols=nz[1], 
                              val=-Vnz, method='exact')
                              # val=-np.ones([len(nz[0])]), method='exact')
        
        self.declare_partials('sumL', 'L',val = np.ones(nc),method='exact')
        
        self.declare_partials('p', 'rho',
                              rows=np.arange(nr+1), cols=np.arange(nr+1),
                              val=np.ones(nr+1), method='exact')  
            
            

    def compute(self, inputs, outputs):
        p = self.options["order"]
        V = self.options['V']
        (nr,nc) = np.shape(V)
        
        L = inputs['L']
        rho = inputs['rho']

        outputs['del'] = rho[1:nr+1] - V@L
        outputs['sumL'] = sum(L)
        
        outputs['p'] = (rho/(1+p*(1-rho))) # RAMP approach
        # outputs['p'] = np.sinh(p*(rho-1))/np.sinh(p) + 1 # SinH approach
        # outputs['p'] = rho # no penalty



    def compute_partials(self, inputs, partials):
        # tic = time.time()
        p = self.options["order"]
        V = self.options['V']
        (nr,nc) = np.shape(V)
        
        rho = np.reshape(inputs['rho'],(nr+1,))
        
        partials['p','rho'] = (p+1) / ((1+p*(1-rho))**2) # RAMP approach
        # partials['p','rho'] = (p*np.cosh(p*(rho - 1)))/np.sinh(p) # SINH approach 
        # partials['p','rho'] = 1 # no penalty
        # toc = time.time(); print('Partials Time:', toc-tic)


#%%
class VertexSet2(om.ExplicitComponent):
    
    def initialize(self):        
        self.options.declare('V', types=np.ndarray, desc='Vertices')
        self.options.declare('order', types=int, desc='Penalty Order')
 
    
 
    def setup(self):
        V = self.options['V']
        (nr,nc) = np.shape(V)
        
        self.add_input('L',shape=(nc,1),desc='lambda vector')
        
        self.add_output('rho',shape=(nr+1,1), desc='desvars')           
        self.add_output('sumL',val=1, desc='Sum Constraint')
        self.add_output('p',shape=(nr+1,1), desc='Penalty Vars')



        
    def setup_partials(self):
        V = self.options['V']
        (nr,nc) = np.shape(V)
        nz = np.nonzero(V)
        Vnz = V[nz[0],nz[1]]
        
        
        self.declare_partials('rho', 'L', rows=nz[0]+1, cols=nz[1], 
                              val=Vnz, method='exact')
                              # val=np.ones([len(nz[0])]), method='exact')
        
        self.declare_partials('sumL', 'L',val = np.ones(nc),method='exact')
        self.declare_partials('p', 'L',method='exact')     
            
            

    def compute(self, inputs, outputs):
        p = self.options["order"]
        V = self.options['V']
        (nr,nc) = np.shape(V)
        
        L = inputs['L']

        rho = np.concatenate((np.array([[1]]),V@L),axis=0)
        outputs['rho'] = rho
        outputs['sumL'] = sum(L)
        
        outputs['p'] = (rho/(1+p*(1-rho))) # RAMP approach
        # outputs['p'] = np.sinh(p*(rho-1))/np.sinh(p) + 1 # SinH approach
        # outputs['p'] = rho # no penalty



    def compute_partials(self, inputs, partials):
        # tic = time.time()
        p = self.options["order"]
        V = self.options['V']
        (nr,nc) = np.shape(V)
        
        L = inputs['L']
        rho = np.reshape(np.concatenate((np.array([[1]]),V@L),axis=0),(nr+1,))

        dp_drho = (p+1) / ((1+p*(1-rho))**2) # RAMP approach
        # dp_drho = (p*np.cosh(p*(rho - 1)))/np.sinh(p) # SINH approach 
        # dp_drho = 1 # no penalty
        
        partials['p','L'] = np.diag(dp_drho)@np.concatenate((np.zeros((1,nc)),V),axis=0)
        # toc = time.time(); print('Partials Time:', toc-tic)   
    




#%%
# L = np.arange(682)/sum(np.arange(682))
# # L = np.zeros(682); L[10] = 1 
# rho = np.concatenate((np.array([1]),V@L),axis=0)
  
# p = om.Problem()
# p.model.add_subsystem('TopSet', subsys=VertexSet1(V=V,order=10),promotes=['*'])

# tic = time.time()
# p.setup(force_alloc_complex=True)
# toc = time.time(); print('Setup Time:', toc-tic)

# p['L'] = L
# p['rho'] = rho 

# tic = time.time()
# p.run_model() 
# toc = time.time(); print('Run Time:', toc-tic)   
# a = p.check_partials(method='cs',compact_print=True)
# om.n2(p)

