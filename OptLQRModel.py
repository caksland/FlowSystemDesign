import os
from SharedFunctions import plot_results
import AllLqrSynthModels as mdls
import dymos as dm
import DynamicModel as dmm
import openmdao.api as om
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np
import scipy.io as sio
from sioLoad import loadmat 
from my_simulate import my_simulate


name = 'L4T0H1'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['Vert']

name = 'L4T0H1_NORM'
mat = sio.loadmat('Vert_'+name+'.mat')
V = mat['V_NRM']
V_LB = mat['V_LB']
V_UB = mat['V_UB']


n_phases = 1 #number of phases
nn = [10,10,10] # number of nodes in each phase
t_seg = [100,100,100] # time duration of each phase
fix_dur = [True,True,True] # fixed time for each phase
tx_meth = ['GL','GL','GL'] # transcription for each phase 'GL' or 'PS'
epsilon = 1e-6
xi = [25,25,25,25] # initial state guess
d_ =  [.1,25,600,400,700,300] # disturbances
dLpulse =  [400,1500,700,1000] # other distrubnaces
xlim = [150,150,150,50,55,60,65] # state limits

nAll = np.ones(682) #np.arange(682)+1
L = nAll/sum(nAll)
rho = np.concatenate((np.array([1]),V@L),axis=0)

#%% Setup Problem

p = om.Problem()
p.driver = om.ScipyOptimizeDriver()
p.driver.options['maxiter']=200
# p.driver.options['tol']=10**-2
p.model.linear_solver = om.DirectSolver(iprint=0)
# p.driver.options['debug_print'] = ['objs']
warnings.filterwarnings('ignore', category=om.UnitsWarning) # surpress unit warnings.
warnings.filterwarnings('ignore', category=om.DerivativesWarning) # surpress unit warnings.



#%% add topology Model
p.model.add_subsystem('Top',subsys=mdls.ExtModel(V=V,order=0,
                              Np=39,Nx=15,Nd=7,Na=8,Nld=7),promotes=['*'])



#%% Add phases 
phases = []
for cnt in range(n_phases):
    
    # create the phase
    if tx_meth[cnt]=='PS':
        tx = dm.Radau(num_segments=nn[cnt],compressed=False)
    else:
        tx = dm.GaussLobatto(num_segments=nn[cnt],compressed=False,order=3)        
        # tx = dm.ExplicitShooting(num_segments=10, num_steps_per_segment=10, method='rk4')        
    phase = dmm.DynamicPhase(ode_class=mdls.IntModel, transcription=tx, 
                              model_kwargs={'Np':39,'Nd':7,'Nld':7,'U':0.05,
                                            'L':-0.05,'p':20})
    if cnt>0 :
        fix_init = False
    else:
        fix_init = True
            
    # add PowerSystem Model functions
    phase.init_vars(openmdao_path="TMS", 
                    state_names=["x"], #all the states
                    control_names = [], # sink flow rate
                    parameter_names = ['d'], # sink temperature, heat loads
                    output_names=[],
                    var_opts = {
                        "x":{'fix_initial':fix_init, "fix_final":False,
                             "lower":10,"upper":150,"ref0":10,"ref":150},
                                # "upper":55,"ref0":10,"ref":55},
                        "d":{'opt':False},
                            })
    phase.add_state('J',rate_source='J_dot',fix_initial=fix_init,ref0=0,ref=10)
    phase.add_parameter('K')
    phase.add_parameter('F')
    phase.add_parameter('ue')
    phase.add_parameter('xe')
    phase.add_parameter('rho')

    # time options       
    phase.set_time_options(fix_initial=fix_init, fix_duration=True, duration_val=t_seg[cnt])
    
    # add number of phases
    phases.append(phase)
   
# manually set segment time becasue its not be setting above
for i in range(n_phases):
    p.model.set_input_defaults('traj.phase{}.t_duration'.format(i), val=t_seg[i]) 

#%% add the trajectory
traj = dmm.DynamicTrajectory(phases, linked_vars=['*'], phase_names="phase")
p.model.add_subsystem('traj', traj)



#%% Add trajectory parameters and connections
traj.init_vars(openmdao_path="TMS", 
                        parameter_names = ["p"],
                        var_opts = {
                            "p":{'opt':False,'static_target':True}, #penalty variable
                                })

# add other parameters
traj.add_parameter('K', opt=False)
traj.add_parameter('F', opt=False)
traj.add_parameter('ue', opt=False)
traj.add_parameter('xe', opt=False)
traj.add_parameter('rho', opt=False)

# connect/promote parameters from upstream analysis into the trajactory
for i in range(39):
    p.model.connect('p','traj.parameters:TMS_p{}'.format(i+1), src_indices=[i])
p.model.connect('K','traj.parameters:K')
p.model.connect('F','traj.parameters:F')
p.model.connect('ue','traj.parameters:ue')
p.model.connect('xe','traj.parameters:xe',src_indices=range(7))
p.model.promotes('Top',any=[('rho','traj.parameters:rho')]) 



#%% add state options
for cnt,phase in enumerate(phases):
    for i in [4,5,6,7]:
        phase.set_state_options('TMS_x{}'.format(i),upper=xlim[i-1], ref=xlim[i-1]) 



#%% specify Disturbances
for cnt,phase in enumerate(phases):
    for i in range(len(d_)):
        phase.set_parameter_options('TMS_d{}'.format(i+1),val=d_[i],opt=False) 



#%% add design Variables
# p.model.add_design_var('L',lower=0,upper=1,ref0=0,ref=1)
# p.model.add_design_var('traj.parameters:rho',lower=0,upper=1,ref0=0,ref=1)
# p.model.add_design_var('traj.parameters:rho',lower=epsilon,upper=1,ref0=0,ref=1,indices=np.arange(1,39))
# p.model.add_design_var('R',lower=1000,upper=100000,ref0=1000,ref=100000)


#%% Objective
# phases[-1].add_objective('time', loc='final',ref0=0,ref=100)
phases[-1].add_objective('J', loc='final',ref0=0,ref=10)



#%% set initial default values to appease openmdao 3.23.0 and dymos 1.6.1
## UNCOMMMENT IF RUNNING IN NEWER OPENMDAO AND DYMOS VERSIONS
# p.model.set_input_defaults('traj.parameters:rho', val=rho)



#%% setup model   
tic = time.time()
p.setup(force_alloc_complex=False)
toc = time.time(); print('Setup Time:', toc-tic)

# set initial guesses. 
for cnt,phase in enumerate(phases):
    p.set_val('traj.phase{}.states:TMS_x1'.format(cnt), phase.interp('TMS_x1', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x2'.format(cnt), phase.interp('TMS_x2', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x3'.format(cnt), phase.interp('TMS_x3', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x4'.format(cnt), phase.interp('TMS_x4', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x5'.format(cnt), phase.interp('TMS_x5', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x6'.format(cnt), phase.interp('TMS_x6', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:TMS_x7'.format(cnt), phase.interp('TMS_x7', ys=xi[cnt:cnt+2]))
    p.set_val('traj.phase{}.states:J'.format(cnt), phase.interp('J', ys=[0,10]))


p.set_val('L',L)
p.set_val('Q',np.eye(7))
p.set_val('R',10000*np.ones(39))
p.set_val('traj.parameters:rho',rho) # comment out if useing vertex set 2

# p.run_model()
om.n2(p)



#%% Optimize the system
tic = time.time()
dm.run_problem(p,run_driver=True)
toc = time.time(); print(' Optimization Time:', toc-tic)
om.n2(p)



#%% Plot Results
exp_out = my_simulate(traj,times_per_seg=10)

plot_results([('time', 'states:TMS_x1', 't (s)', 'Sink',None),
              ('time', 'states:TMS_x2', 't (s)', 'Tank',None),
              ('time', 'states:TMS_x3', 't (s)', 'Hx',None),
              ('time', 'states:TMS_x4', 't (s)', 'Load 1',None),
              ('time', 'states:TMS_x5', 't (s)', 'Load 2',None),
              ('time', 'states:TMS_x6', 't (s)', 'Load 3',None),
              ('time', 'states:TMS_x7', 't (s)', 'Load 4',None),
                ('time', 'states:J', 't (s)', 'Obj',None),
              ],
              title='TMS',
              p_sol=p, p_sim=exp_out,nrows=4,ncols=2,figsize=(10,12))
