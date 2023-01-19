import openmdao.api as om
import os
import DynamicModel as dmm
import numpy as np
import scipy.io as sio
import numpy.matlib as npm
from scipy import sparse  
import time
import scipy.linalg as spla


import TopologySetModels as top
import CtrlSynthModels as ctrl
import TrajModels as trm


name = 'L4T0H1'
# name = 'L4T1H1'

np.random.seed(1242)
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['Vert']

name = 'L4T0H1_NORM'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['V_NRM']

name = 'L4T0H1'

#%% TMS from MATLAB
class TMS(dmm.DynamicModel):
    def initialize(self):
        super().initialize()
        
        mdl = "TopPlant_"+name
        # mdl_path = os.path.join(os.path.dirname(__file__), "../MM_Model1") # Point Model to folder containing the Model folder.  This is required for !openmdao check functions
        self.options["Model"] = mdl
        self.options["Path"] = ""
        self.options["Functions"] = ["h", "f"]
        # self.options["Functions"] = ["g","h", "f"]
        self.options["StaticVars"] = ["p"]

#%% Total Model
class TotalModel(om.Group):
    def initialize(self):
        self.options.declare('V', types=np.ndarray, desc='Vertices')
        self.options.declare('order', types=int, desc='Penalty Order')
        
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nx', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Na', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
        self.options.declare('num_nodes', types=int)
        self.options.declare('p', types=int)
        self.options.declare('L', types=int)
        self.options.declare('U', types=int)
    
    
    def setup(self):
        V = self.options['V']
        o = self.options['order']
        
        Np = self.options['Np']
        Nx = self.options['Nx']
        Nd = self.options['Nd']
        Na = self.options['Na']
        Nld = self.options['Nld']

        nn = self.options['num_nodes']
        U = self.options['U']
        L = self.options['L']
        p = self.options['p']
        
        self.add_subsystem('TopSet', subsys=top.VertexSet2(V=V,order=o),promotes=['*'])
        
        self.add_subsystem('EQ', subsys=ctrl.getStateEquilibrium(Np=Np,Nx=Nx),promotes=['*'])
        self.add_subsystem('SS', subsys=ctrl.getSSMatrix(Np=Np,Nx=Nx,Nd=Nd,Na=Na,Nld=Nld),promotes=['*'])
        self.add_subsystem('LQR', subsys=ctrl.RiccatiTune(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])
        self.add_subsystem('Gain', subsys=ctrl.CtrlGain(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])
        
        self.add_subsystem('Input', subsys=trm.calcU(num_nodes=nn,Np=Np,Nd=Nd,Nld=Nld))
        self.promotes('Input',inputs=['xe'],src_indices=np.arange(7))
        self.promotes('Input',inputs=['K'])
        self.promotes('Input',outputs=['u'])
        self.add_subsystem('Sat', subsys=trm.Saturate(num_nodes=nn,p=p,Np=Np,Nld=Nld,L=L,U=U), promotes=['*'])
        self.add_subsystem('TMS', subsys=TMS(num_nodes=nn))
        for i in range(Np):
            self.connect('p','TMS.p{}'.format(i+1), src_indices=[i])
            self.connect('us','TMS.u{}'.format(i+1), src_indices=([i]*nn,range(nn)))
        for i in range(Nd):
            self.promotes('Input',inputs=[('x_{}'.format(i+1),'TMS.x{}'.format(i+1))])

#%% External to Trajectory Model
class ExtModel(om.Group):
    def initialize(self):
        self.options.declare('V', types=np.ndarray, desc='Vertices')
        self.options.declare('order', types=int, desc='Penalty Order')
        
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nx', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Na', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
    
    def setup(self):
        V = self.options['V']
        o = self.options['order']
        
        Np = self.options['Np']
        Nx = self.options['Nx']
        Nd = self.options['Nd']
        Na = self.options['Na']
        Nld = self.options['Nld']
        
        self.add_subsystem('TopSet', subsys=top.VertexSet1(V=V,order=o),promotes=['*'])
        # self.add_subsystem('TopSet', subsys=top.VertexSet2(V=V,order=o),promotes=['*'])
        
        self.add_subsystem('EQ', subsys=ctrl.getStateEquilibrium(Np=Np,Nx=Nx),promotes=['*'])
        self.add_subsystem('SS', subsys=ctrl.getSSMatrix(Np=Np,Nx=Nx,Nd=Nd,Na=Na,Nld=Nld),promotes=['*'])
        self.add_subsystem('LQR', subsys=ctrl.RiccatiTune(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])
        self.add_subsystem('Gain', subsys=ctrl.CtrlGain(Np=Np,Nd=Nd,Nld=Nld),promotes=['*'])
        
        epsilon = 1e-6
        self.add_constraint('sumL',lower=1-epsilon,upper=1+epsilon,ref0=.9,ref=1.1,linear=True)
        self.add_constraint('del',lower=-epsilon,upper=+epsilon,ref0=-0.1,ref=0.1,linear=True)

#%% Internal to Trajectory Model
class IntModel(om.Group):
    def initialize(self):
        self.options.declare('Np', types=int, default = 2)
        self.options.declare('Nd', types=int, default = 2)
        self.options.declare('Nld', types=int, default = 2)
    
        self.options.declare('num_nodes', types=int, default = 2)
        self.options.declare('p', types=int, default = 10)
        self.options.declare('L', types=float, default = 0.0)
        self.options.declare('U', types=float, default = 1.0)
    
    
    def setup(self):
        
        Np = self.options['Np']
        Nd = self.options['Nd']
        Nld = self.options['Nld']

        nn = self.options['num_nodes']
        if nn==0:
            nn=2
        U = self.options['U']
        L = self.options['L']
        p = self.options['p']
        
        self.add_subsystem('Input', subsys=trm.calcU(num_nodes=nn,Np=Np,Nd=Nd,Nld=Nld))
        self.promotes('Input',inputs=['xe'])
        self.promotes('Input',inputs=['K'])
        self.promotes('Input',outputs=['u'])
        self.add_subsystem('Sat', subsys=trm.Saturate(num_nodes=nn,p=p,Np=Np,Nld=Nld,L=L,U=U), promotes=['*'])
        self.add_subsystem('TMS', subsys=TMS(num_nodes=nn))
        self.add_subsystem('Obj', subsys=trm.Obj(num_nodes=nn,Np=Np),promotes=['*'])
        for i in range(Np):
            self.connect('us','TMS.u{}'.format(i+1), src_indices=([i]*nn,list(range(nn))))
        for i in range(Nd):
            self.promotes('Input',inputs=[('x_{}'.format(i+1),'TMS.x{}'.format(i+1))])


#%%
# L = np.arange(682)/sum(np.arange(682))
# rho = np.concatenate((np.array([1]),V@L),axis=0)
# R = np.arange(1,40)     
# Q = np.diag(np.arange(1,8))  


# p = om.Problem()
# # p.model.add_subsystem('Top',subsys=ExtModel(V=V,order=0,
# #                               Np=39,Nx=15,Nd=7,Na=8,Nld=7),promotes=['*'])
# p.model.add_subsystem('traj',subsys=IntModel(Np=39,Nd=7,Nld=7,
#                               num_nodes=4,U=1.0,L=0.0,p=10))
# # for i in range(39):
# #             p.model.connect('p','traj.TMS.p{}'.format(i+1), src_indices=[i])
# # p.model.connect('ue','traj.ue')
# # p.model.connect('F','traj.F')
# # p.model.connect('xe','traj.xe',src_indices=range(7))
# # p.model.connect('K','traj.K')
# # p.model.connect('rho','traj.rho')
       
# tic = time.time()
# p.setup(force_alloc_complex=True)
# toc = time.time(); print('Setup Time:', toc-tic)

# # p['L'] = L
# # p['rho'] = rho 
# # p['R'] = R 
# # p['Q'] = Q

# tic = time.time()
# p.run_model()   
# toc = time.time(); print('Run Time:', toc-tic)
# tic = time.time()
# a = p.check_partials(method='cs',compact_print=True)
# toc = time.time(); print('Partials Time:', toc-tic)
# om.n2(p)



#%% old stuff
# L = np.arange(682)/sum(np.arange(682))
# rho = np.concatenate((np.array([1]),V@L),axis=0)
# R = np.arange(1,40)     
# Q = np.diag(np.arange(1,8))  


# p = om.Problem(model=TotalModel(V=V,order=0,
#                               Np=39,Nx=15,Nd=7,Na=8,Nld=7,
#                               num_nodes=4,U=1,L=0,p=10))


# p.setup(force_alloc_complex=True)
# p['L'] = L
# p['rho'] = rho 
# p['R'] = R 
# p['Q'] = Q

# tic = time.time()
# p.run_model()    
# a = p.check_partials(method='cs',compact_print=True)
# toc = time.time()
# print('Time:', toc-tic)
# om.n2(p)




# p = om.Problem(model=TMS(num_nodes=4))
# p.setup(force_alloc_complex=True)

# tic = time.time()
# p.run_model()    
# a = p.check_partials(method='cs',compact_print=True)
# toc = time.time()
# print('Time:', toc-tic)
# om.n2(p)