import openmdao.api as om
import os
import DynamicModel as dmm
import StaticModel as smm
import numpy as np
import numpy.matlib
import time
import scipy.io as sio
import matplotlib.pyplot as plt
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
 
#%%        
class calcU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int, default = 2)
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
    
    
    def setup(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        Nd = self.options["Nd"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        
        for i in range(Nd):
            self.add_input('x_{}'.format(i+1),shape=(nn,))

        self.add_input('xe',shape=(Nd,1),tags=['dymos.static_target'])
        self.add_input('K',shape=(Nli,Nd),tags=['dymos.static_target'])
        self.add_output('u',shape=(Nli,nn))
            
        
        
    def setup_partials(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        Nd = self.options["Nd"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        
        r = np.reshape(np.matlib.repmat(np.arange(Nli*nn),Nd,1),(Nli*nn*Nd,),order='F')
        c1 = np.reshape(np.matlib.repmat(np.arange(Nd),1,Nli*nn),(Nli*nn*Nd,),order='F')
        c2 = Nd*np.reshape(np.matlib.repmat(np.arange(Nli),Nd*nn,1),(Nli*Nd*nn,),order='F')
        c = c1+c2
        
        self.declare_partials('u', 'xe',
                              method='exact')
        self.declare_partials('u', 'K', rows=r, cols=c, val=np.zeros((Nli*Nd*nn,)),
                              method='exact')   
        self.declare_partials('u', 'x_*', 
                                  rows=np.arange(Nli*nn), cols=np.reshape(np.matlib.repmat(np.arange(nn),1,Nli),(Nli*nn)),
                                  val=np.zeros((Nli*nn)), method='exact')



    def compute(self,inputs,outputs):
        nn = self.options["num_nodes"]
        Nd = self.options["Nd"]
        
        xe = inputs['xe']
        K = inputs['K']
        
        x = np.reshape(inputs['x_1'],(1,nn))
        for i in (np.arange(Nd-1)+1):
            x_i = np.reshape(inputs['x_{}'.format(i+1)],(1,nn))
            x = np.concatenate((x,x_i),axis=0)
            
        
        outputs['u'] = -K@(x-xe)
    
    
    
    def compute_partials(self, inputs, partials):
        # tic = time.time()
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        Nd = self.options["Nd"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        
        K = inputs['K']
        xe = inputs['xe']
        x = np.reshape(inputs['x_1'],(1,nn))
        for i in (np.arange(Nd-1)+1):
            x_i = np.reshape(inputs['x_{}'.format(i+1)],(1,nn))
            x = np.concatenate((x,x_i),axis=0)
        

        partials['u','K'] = -np.reshape(np.matlib.repmat(x-xe, 1, Nli),(Nli*nn*Nd,),order='F')
        partials['u','xe'] = np.reshape(np.matlib.repmat(K,1,nn).T,(Nd,Nli*nn),order='F').T

        for i in range(7):
            partials['u','x_{}'.format(i+1)] = -np.reshape(np.matlib.repmat(K[:,i],nn,1),(Nli*nn,),order='F')
        # toc = time.time(); print('Partials Time:', toc-tic)
                

#%% 
class Saturate(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int, default = 2)
        self.options.declare('p', types=int, default = 10)
        self.options.declare('L', types=float, default = 0.0)
        self.options.declare('U', types=float, default = 1.0)
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
    
    
    def setup(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        
        
        self.add_input('u',shape=(Nli,nn))
        self.add_input('F', shape=(Np,Np-Nld),tags=['dymos.static_target'])
        self.add_input('ue', shape=(Np,1),tags=['dymos.static_target'])
        self.add_output('us',shape=(Np,nn))   
        
        
        
    def setup_partials(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        
        self.declare_partials('us', 'u', method='exact')
    
        r = np.reshape(np.matlib.repmat(np.arange(Np*nn),Nli,1),(Nli*nn*Np,),order='F')
        c1 = np.reshape(np.matlib.repmat(np.arange(Nli),1,Np*nn),(Np*nn*Nli,),order='F')
        c2 = Nli*np.reshape(np.matlib.repmat(np.arange(Np),Nli*nn,1),(Np*Nli*nn,),order='F')
        c = c1+c2
        self.declare_partials('us', 'F', rows=r, cols=c, 
                              val=np.zeros((Nli*Np*nn,)), method='exact')
        
        c = np.reshape(np.matlib.repmat(np.arange(Np),nn,1),(Np*nn,),order='F')
        self.declare_partials('us', 'ue', rows=np.arange(Np*nn), cols=c,
                              val=np.ones((Np*nn,)), method='exact')
                              

        
    def compute(self, inputs, outputs):
        p = self.options["p"]
        L = self.options["L"]
        U = self.options["U"]
        u = inputs['u']
        F = inputs['F']
        ue = inputs['ue']
        
        us = L/2+U/2-(L+U-2*u)/(2*(((L+U-2*u)/(L-U))**p+1)**(1/p))
        outputs['us'] = ue + F@us
        
        
        
        
    def compute_partials(self, inputs, partials):
        # tic = time.time()
        nn = self.options["num_nodes"]
        p = self.options["p"]
        L = self.options["L"]
        U = self.options["U"]
        Np = self.options["Np"]
        Nld = self.options["Nld"]
        Nli = Np-Nld
        u = inputs['u']
        F = inputs['F']
        
        us = L/2+U/2-(L+U-2*u)/(2*(((L+U-2*u)/(L-U))**p+1)**(1/p))
        dus_du = 1/(((L+U-2*u)/(L-U))**p+1)**(1/p)-((L+U-2*u)/(L-U))**p/(((L+U-2*u)/(L-U))**p+1)**((p+1)/p)
        partials['us','u'] = np.kron(F,np.eye(nn))@np.diag(np.reshape(dus_du,(Nli*nn,),order='C'))
        partials['us','F'] = np.reshape(np.matlib.repmat(us, 1, Np),(Np*nn*Nli,),order='F')
        # toc = time.time(); print('Partials Time:', toc-tic)
        
#%%
class Obj(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int, default = 2)
        self.options.declare('Np', types=int, default = 2)
    
    
    
    def setup(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]        
        
        self.add_input('us',shape=(Np,nn))
        self.add_input('rho', shape=(Np,1),tags=['dymos.static_target'])
        self.add_output('J_dot',shape=(nn,))   
        
        
        
    def setup_partials(self):
        nn = self.options["num_nodes"]
        Np = self.options["Np"]
        
        self.declare_partials('J_dot', 'rho', method='exact')
        (r,c) = np.nonzero(np.kron(np.ones((1,Np)),np.eye(nn)))
        self.declare_partials('J_dot', 'us', 
                              rows=r, cols=c,
                              val=np.zeros(len(r)),method='exact')
    
    
        
    def compute(self, inputs, outputs):
        rho = inputs['rho']
        us = inputs['us']
        
        # up2 = (us)**2
        up2 = (us*rho)**2
        outputs['J_dot'] = sum(up2)
        
        
        
    def compute_partials(self, inputs, partials):
        # tic = time.time()
        nn = self.options["num_nodes"]
        Np = self.options["Np"]

        rho = inputs['rho']
        us = inputs['us']
        unew = np.reshape(2*us*rho**2,(1,Np*nn),order='F')
        
        partials['J_dot','rho'] = (2*us**2*rho).T
        partials['J_dot','us'] = unew
        # toc = time.time(); print('Partials Time:', toc-tic)


#%%
class TrajCtrl(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('p', types=int)
        self.options.declare('L', types=float)
        self.options.declare('U', types=float)
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)     
    
    
    
    def setup(self):
        nn = self.options["num_nodes"]
        p = self.options["p"]
        L = self.options["L"]
        U = self.options["U"]
        Np = self.options["Np"]
        Nd = self.options["Nd"]
        Nld = self.options["Nld"]
        
        self.add_subsystem('Input', subsys=calcU(num_nodes=nn,Np=Np,Nd=Nd,Nld=Nld), promotes=['*'])
        self.add_subsystem('Sat', subsys=Saturate(num_nodes=nn,p=p,Np=Np,Nld=Nld,L=L,U=U), promotes=['*'])
        self.add_subsystem('Obj', subsys=Obj(num_nodes=nn,Np=Np), promotes=['*'])

#%%
# nn=8
# L = np.arange(682)/sum(np.arange(682))
# rho = np.concatenate((np.array([1]),V@L),axis=0)
# p = om.Problem(model=TrajCtrl(num_nodes=nn,p=20,Np=39,Nd=7,Nld=7,L=0.0,U=1.0))

# tic = time.time()
# p.setup(force_alloc_complex=True)
# toc = time.time(); print('Setup Time:', toc-tic)

# p['K'] = (np.reshape(np.arange(7*32)+1,(32,7)))/10000
# p['F'] = (np.reshape(np.arange(39*32)+1,(32,39)))/10000
# p['xe'] = np.array([[2,1,6,7,3,9,4]]).T
# p['ue'] = np.array([[2,1,6,7,3,9,4,6,6,9,3,5,7,4,6,8,2,4,6,4,6,8,7,2,9,3,1,7,4,8,2,3,9,7,3,5,4,7,2]]).T
# p['rho'] = rho
# for i in range(7):
#     p['x_{}'.format(i+1)] = np.arange(i,i+nn)+1
    
# tic = time.time()
# p.run_model()
# toc = time.time(); print('Run Time:', toc-tic)

# # tic = time.time()
# a = p.check_partials(method='cs',compact_print=True)
# # toc = time.time(); print('Partials Time:', toc-tic)
# om.n2(p)






            