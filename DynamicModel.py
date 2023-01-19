# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:28:42 2021

@author: renkert2
"""
import numpy as np
import openmdao.api as om
import dymos as dm
import os
import sys
from SharedFunctions import *
import logging
import copy
import time
import re
from functools import reduce  # forward compatibility for Python 3
import operator
import copy
from dymos.utils.misc import _unspecified
from TopPlant_L4T0H1.hA import hA
from TopPlant_L4T0H1.hB import hB

##########################################
### OpenMDAO modeling from python file ###
##########################################

class DynamicModel(om.Group): 
    def initialize(self):
        self.options.declare('Model', types=str, default = 'None') # "Model" property used to store name of folder containing Model .py functions and variable tables
        self.options.declare('Path', types=str, default = '') # Path to directory containing model folder "Model", i.e. abs_path = Path/Model/
        self.options.declare('Functions', default=None) # List of ModelFunctions, added in order 
        self.options.declare('Metadata', default=None, recordable=False) # Calculated in setup() if not passed as argument
        
        self.options.declare('num_nodes', types=int) # Number of nodes property, required for Dymos models
        self.options.declare('StaticVars', types=list, default=[]) # List of variables to be treated as static parameters
        self.options.declare('AddSolvers', types=bool, default=False) # Option to add top level solver to the model.  Useful when model contains algebraic states
    
    def setup(self):
        meta = self.options["Metadata"]
        if meta:
            self.Metadata = meta
        else:
            mdl = self.options['Model']
            mdl_path = os.path.join(self.options["Path"], mdl)
            self.ModelPath = mdl_path
            self.Metadata = ImportMetadata(mdl_path)
        
        nn = self.options["num_nodes"]
        sv = self.options["StaticVars"]

        funcs = self.options["Functions"]
        if not funcs:
            funcs = list(self.Metadata["Function"].keys())
            
        for f in funcs:
            # Create copies of metadata dictionaries for components.  
            meta = self.Metadata.copy()
            f_meta = self.Metadata["Function"][f].copy()
            meta["Function"] = f_meta
            j_meta = self.Metadata["Jacobian"][f].copy()
            meta["Jacobian"] = j_meta
            
            # Create a ModelFunction for each component and add it to the group as a subsystem
            shared_opts = {"FunctionName":f, "Metadata":meta, "num_nodes":nn, "StaticVars":sv}
            if f_meta["Type"] == "Implicit":
                subsys = ImplicitModelFunction(**shared_opts)
            elif f_meta["Type"] == "Explicit":
                subsys = ExplicitModelFunction(**shared_opts)
 
            self.add_subsystem(name=f, subsys=subsys,
                           promotes = ["*"])
            
        ### SOLVERS (Top Level) ###
        if self.options["AddSolvers"]:
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False,iprint=0)
            self.linear_solver = om.DirectSolver(iprint=0)
        
class ImplicitModelFunction(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('FunctionName', types=str)
        self.options.declare('Metadata', default=None, recordable=False)
                
        self.options.declare("num_nodes", types=int)
        self.options.declare("StaticVars", types=list)
        
    def setup(self):
        setupInputsOutputs(self)
        
        ### SOLVERS (Component Level) ###
        # Implicit Model Functions require an additional solver to converge their state values
        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False,iprint=0,maxiter=20)
        # self.linear_solver = om.DirectSolver(iprint=0)
        # self.nonlinear_solver.linesearch=om.BoundsEnforceLS()
        # self.nonlinear_solver.linesearch.options['iprint'] = 0
        
    def setup_partials(self):
        setupPartials(self)

    def apply_nonlinear(self, inputs, outputs, residuals):
        meta = self.options["Metadata"]
        out_names = self._out_var_elems
        
        # compile arguements in a compatible shape for the function call and call the function
        arg_list = AssembleArguments(meta, inputs, outputs) 
        residuals = ComputeModel(residuals, arg_list, meta, out_names)
        
    # TODO: Add solve_nonlinear method using scipy's linsolve method which implements LU decomposition to solve the system
    
    def solve_nonlinear(self, inputs, outputs):
        meta = self.options["Metadata"]
        nn = self.options["num_nodes"]
        out_names = self._out_var_elems
        
        # compile arguements in a compatible shape for the function call and call the function
        arg_list = AssembleArguments(meta, inputs, outputs)
        A = hA(*arg_list,nn)
        B = hB(*arg_list,nn)
        
        outs = np.reshape(np.linalg.inv(A)@B,(8,nn),order='F')
        # Assign to Outputs
        if len(out_names) == 1:
            outputs[out_names[0]] = outs
        else:
            for i, out_var in enumerate(out_names): 
                outputs[out_var] = outs[i,:] 
        
    
    def linearize(self, inputs, outputs, partials):
        meta = self.options["Metadata"]
        out_names = self._out_var_elems
        
        # Assemble Vectors and call the function that computes partials
        arg_list = AssembleArguments(meta, inputs, outputs)
        partials = ComputeModelPartials(partials, arg_list, meta, out_names)
        
        
                
class ExplicitModelFunction(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('FunctionName', types=str)
        self.options.declare('Metadata', default=None, recordable=False)
        
        self.options.declare("num_nodes", types=int)
        self.options.declare("StaticVars", types=list)
        
    def setup(self):
        setupInputsOutputs(self)
        
    def setup_partials(self):
        setupPartials(self)
    
    def compute(self, inputs, outputs):
        meta = self.options["Metadata"]
        out_names = self._out_var_elems
        
        # compile arguements in a compatible shape for the function call and call the function
        arg_list = AssembleArguments(meta, inputs, [])    
        outputs = ComputeModel(outputs, arg_list, meta, out_names)
        
    
    def compute_partials(self, inputs, partials):
        meta = self.options["Metadata"]
        out_names = self._out_var_elems
        
        # Assemble Vectors and call the function that computes partials
        arg_list = AssembleArguments(meta, inputs, [])
        partials = ComputeModelPartials(partials, arg_list, meta, out_names)

        
        
######################
### Dymos modeling ###
######################

class DynamicProblem(om.Problem):
    # TODO: Can wrap all the problem definition stuff here if we want to
    pass

class DynamicTrajectory(dm.Trajectory):
        # mdl: model directory name (Only required for ode_class = DynamicModel)
        # path: path to model directory (Only required for ode_class = DynamicModel)
        # model_kwargs: Additional keyword arguments for model
        # phase_kwargs: Additional keyword arguments for phase
        # kwargs: additional arguments for trajectory constructor
        # linked_vars: list of variables that should be linked between phases. See dymos documentation for link_phases()
        
        # we may want to consider
    
    def __init__(self, phases, linked_vars = ['*'], phase_names = "phase", phase_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(phases, int):
            phase_count = phases
            phases = []
            for i in range(phase_count):
                phases.append(DynamicPhase(**phase_kwargs))
                
        for (i, phase) in enumerate(phases):
            if isinstance(phase_names, list):
                name = phase_names[i]
            elif isinstance(phase_names, str):
                name = f'{phase_names}{i}'
            self.add_phase(name, phase)

        if linked_vars:
            if len(phases) > 1:
                self.link_phases([*self._phases],vars=linked_vars)
            
    def set_phase_transcription(self,phases,tx):
        for phase in phases:
            self._phases[phase].options['transcription'] = tx
        
    def set_phase_vars(self, openmdao_path = "",
                  state_names = [], control_names = [], parameter_names = [], output_names = [],
                  var_opts = {}):
        # loop through each phase and initialize the states, controls, and parameters
        for key in self._phases:
            self._phases[key].init_vars(openmdao_path=openmdao_path,
                state_names=state_names, control_names=control_names, 
                parameter_names=parameter_names, output_names=output_names,
                var_opts=var_opts)
            
    def set_phase_time_options(self,phases=[],**kwargs):        
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].set_time_options(**kwargs)
            
    
    def set_phase_state_options(self,name,phases=[],**kwargs):       
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].set_state_options(name, **kwargs)
            
            
    def add_phase_state(self,name,phases=[],**kwargs):
        for phase in phases:
            self._phases[phase].add_state(name, **kwargs)    
            
    
    def set_phase_control_options(self,name,phases=[],**kwargs):      
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].set_control_options(name, **kwargs)
            
    
    def add_phase_control(self,name,phases=[],**kwargs):
        for phase in phases:
            self._phases[phase].add_control(name, **kwargs) 
            
            
    def set_phase_parameter_options(self,name,phases=[],**kwargs):        
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].set_parameter_options(name, **kwargs)
    
    
    def add_phase_objective(self,name,phases=[],**kwargs):        
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].add_objective(name, **kwargs)
    
    
    def add_phase_path_constraint(self,name,phases=[],**kwargs):        
        # add functionality to set the options for multiple variables in the same function call      
        for phase in phases:
            self._phases[phase].add_path_constraint(name, **kwargs)
    
    def phase_interp(self,name, phase='', **kwargs):        
        # add functionality to set the options for multiple variables in the same function call      
        value = self._phases[phase].interp(name, **kwargs)
        
        return value
    
    def set_parameter_options(self, name, val=_unspecified, units=_unspecified, opt=False,
                              desc=_unspecified, lower=_unspecified, upper=_unspecified,
                              scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                              ref=_unspecified, targets=_unspecified, shape=_unspecified,
                              dynamic=_unspecified, static_target=_unspecified):

        if units is not _unspecified:
            self.parameter_options[name]['units'] = units

        if opt is not _unspecified:
            self.parameter_options[name]['opt'] = opt

        if val is not _unspecified:
            self.parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.parameter_options[name]['desc'] = desc

        if lower is not _unspecified:
            self.parameter_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.parameter_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.parameter_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.parameter_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.parameter_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.parameter_options[name]['ref'] = ref

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.parameter_options[name]['targets'] = (targets,)
            else:
                self.parameter_options[name]['targets'] = targets

        if shape is not _unspecified:
            self.parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.parameter_options[name]['static_target'] = not dynamic

        if static_target is not _unspecified:
            self.parameter_options[name]['static_target'] = static_target

        if dynamic is not _unspecified and static_target is not _unspecified:
            raise ValueError("Both the deprecated 'dynamic' option and option 'static_target' were "
                             f"specified for parameter '{name}'. "
                             f"Going forward, please use only option static_target.  Option "
                             f"'dynamic' will be removed in Dymos 2.0.0.")
            
    def init_vars(self, openmdao_path = "", parameter_names = [], var_opts = {}):
        
        # openmdao_path: OpenMDAO Path to level containing promoted variables.  
        #   E.X: If the variable "x" are accessed by "PT.x", openmdao_path = "PT"
        # {type}_names: names of variables to be added as states, controls, or inputs in Dymos
        # E.X.: parameters_names = ["theta"]
        # var_opts: Dictionary of options passed to add_{parameter}, where the keys are the variable name
        # E.X.: var_opts = {"theta":{}}
        # - for parameter options, see add_parameter docs
        
        
        
        def getFromDict(dataDict, mapList):
            # Used to access nested dictionary values with a list of keys
            return reduce(operator.getitem, mapList, dataDict)
        
        # Transform path into list of subsystem names
        openmdao_path_split = openmdao_path.split(".")
        def CreateVarPath(var):
            # If path to OpenMDAO Variable is specified, prepend it to the variable name
            if openmdao_path:
                var_path_split = openmdao_path_split.copy()
                var_path_split.append(var)
                name = "_".join(var_path_split)
                _target = ".".join(var_path_split)
            else:
                name = var
                _target = var
            target = {[*self._phases][i]: [_target] for i in range(len([*self._phases]))}
            return name, target

        # Get Variable Metadata
        # just pull it from the first phase
        meta = getFromDict(self._phases['phase0'].Metadata, openmdao_path_split)
        
        ### ADD VARIABLES ###
        for var in meta["Variable"]:
            var_meta = meta["Variable"][var]
            if var_meta["N"]>0:
                if var in var_opts:
                    _var_opts = var_opts[var]
                else:
                    _var_opts = {}
                
                for var_elem in var_meta["Variables"]:
                    elem_name = var_elem["Variable"]
                    name, target = CreateVarPath(elem_name)                    
                    if (var in parameter_names) or (elem_name in parameter_names):
                        if "Value" in var_elem:
                            self.add_parameter(name, val=var_elem["Value"], targets=target, **_var_opts)                   
                        else:
                            self.add_parameter(name, targets=target, **_var_opts)                       
        
                 
        

class DynamicPhase(dm.Phase):
    def __init__(self, ode_class = DynamicModel, model = '', path = '', model_kwargs = {}, **kwargs):
        # tx: Transcription
        # mdl: model directory name (Only required for ode_class = DynamicModel)
        # path: path to model directory (Only required for ode_class = DynamicModel)
        # model_kwargs: Additional keyword arguments for model
        # kwargs: additional arguments for Phase constructor

        if ode_class.__name__ == "DynamicModel":
            # self.ODEGroupFlag = False 
            # # Flag that says whether the ode_class is a DynamicModel or a group contatining 
            # # dynamic models.  This is useful to know the structure of self.Metadata in init_vars

            # Import metadata
            mdl_path = os.path.join(path, model)
            meta = ImportMetadata(mdl_path)
            # self.Metadata = meta
            self.Metadata = {"":meta}
            
            # Instantiate a Phase and add it to the Trajectory.
            # Here the transcription is necessary but not particularly relevant.
            model_opts = {"Model":model, "Path":path, "Metadata":meta, **model_kwargs}
        elif issubclass(ode_class, om.Group) or ode_class==om.Group:
            # self.ODEGroupFlag = True

            # Instantiate the Object (can't see a way around this)
            group = ode_class()

            # Make sure the group has a "num_nodes" option and set its value so that we 
            if "num_nodes" not in group.options:
                raise Exception("OpenMDAO Group used as the Phase ODE class must have a num_nodes option declared in it's initialize() method")
            group.options["num_nodes"] = 0
            
            # We need to call the group's setup method so that are subsystems are added to the group
            # and its list of attributes
            group.setup()


            def parseDynamicModels(group):
                meta_dict = {}
                group_dir = dir(group) # Get all attributes of the group
                for attr in group_dir:
                    try:
                        attr_val = getattr(group, attr) # get the value of the attribute
                        if isinstance(attr_val, DynamicModel):
                            # Get the Metadata associated with the DynamicModel, 
                            # Assign the Metdata into a dictionary: {path1:meta1, path2:meta2}
                            attr_val.setup()
                            meta_dict[attr] = attr_val.Metadata
                        if isinstance(attr_val, om.Group):
                            # Recursively call function on nested groups
                            attr_val.setup()
                            meta_dict[attr] = parseDynamicModels(attr_val)
                    except Exception as err:
                        msg = f'Unable to access attribute: {attr} \n Error message: {err}'
                        logging.debug(msg)
                return meta_dict
            
            # Parse group for instances of DynamicModels
            dyn_model_dict = parseDynamicModels(group)
            self.Metadata = dyn_model_dict
            model_opts = model_kwargs

        super().__init__(ode_class=ode_class, ode_init_kwargs=model_opts, **kwargs)           
        
    def duplicate(self, N):
        selves = [self]
        for i in range(N - 1):
            self_copy = copy.deepcopy(self) # Make a deep copy of the phase
            self_copy.Metadata = self.Metadata # Each phase can use the same metadata object
            self_copy.options._dict = copy.deepcopy(self.options._dict) # For some reason this wasn't working
            selves.append(self_copy)
        return selves
    
    def init_vars(self, openmdao_path = "",
                  state_names = [], control_names = [], parameter_names = [], output_names = [],
                  var_opts = {}):
        # openmdao_path: OpenMDAO Path to level containing promoted variables.  
        #   E.X: If the variable "x" are accessed by "PT.x", openmdao_path = "PT"
        # {type}_names: names of variables to be added as states, controls, or inputs in Dymos
        # E.X.: state_names = ["x"], control_names = ["u", "d"], parameter_names = ["theta"], output_names = ["y"]
        # var_opts: Dictionary of options passed to add_{parameter}, where the keys are the variable name
        # E.X.: var_opts = {"x":{},"u":{},"d":{},"theta":{}}
        # - For state variable options, see add_state docs
        # - for control variable options, see add_control docs
        # - for parameter options, see add_parameter docs
        # - for output options, see add_timeseries_output docs
        
        """
        we could update the naming convention to use the wild card operator '*'
        """
        
        def getFromDict(dataDict, mapList):
            # Used to access nested dictionary values with a list of keys
            return reduce(operator.getitem, mapList, dataDict)
        
        # Transform path into list of subsystem names
        openmdao_path_split = openmdao_path.split(".")
        def CreateVarPath(var):
            # If path to OpenMDAO Variable is specified, prepend it to the variable name
            if openmdao_path:
                var_path_split = openmdao_path_split.copy()
                var_path_split.append(var)
                name = "_".join(var_path_split)
                target = ".".join(var_path_split)
            else:
                name = var
                target = var
            return name, target

        # Get Variable Metadata
        meta = getFromDict(self.Metadata, openmdao_path_split)
        
        ### ADD VARIABLES ###
        for var in meta["Variable"]:
            var_meta = meta["Variable"][var]
            if var_meta["N"]>0:
                if var in var_opts:
                    _var_opts = var_opts[var]
                else:
                    _var_opts = {}
                
                for var_elem in var_meta["Variables"]:
                    elem_name = var_elem["Variable"]
                    name, target = CreateVarPath(elem_name)                    
                    if (var in state_names) or (elem_name in state_names):
                        self.add_state(name, targets=target, rate_source=target+"_dot", **_var_opts)
                    if (var in control_names) or (elem_name in control_names):
                        self.add_control(name, targets=target, **_var_opts)
                    if (var in parameter_names) or (elem_name in parameter_names):
                        if "Value" in var_elem:
                            self.add_parameter(name, val=var_elem["Value"], targets=target, **_var_opts)                   
                        else:
                            self.add_parameter(name, targets=target, **_var_opts)                   
                    if (var in output_names) or (elem_name in output_names):
                        self.add_timeseries_output(target, output_name="outputs:"+name, **_var_opts)
                        
######################################
### Dynamic Model Helper Functions ###
######################################


def setupInputsOutputs(self):
    meta = self.options["Metadata"]
    nn = self.options['num_nodes']
    sv = self.options['StaticVars']

    def add_vars(add_, var, var_meta,sv,nn,arg_meta=None):
        if var_meta["N"]>0:
            for var_elem in var_meta["Variables"]:
                if (arg_meta==None) or (var_elem['Variable'] in arg_meta):
                    if "Value" in var_elem:
                        val = var_elem["Value"]
                    else:
                        val = 1.0
                    if "Lower" in var_elem:
                        lower = var_elem["Lower"]
                    else:
                        lower=None
                    if "Upper" in var_elem:
                        upper = var_elem["Upper"]
                    else:
                        upper=None
                        
                    #TODO: Size of val may need to change depending on whether the variable is static or not
                    
                    if (var in sv) or (var_elem['Variable'] in sv):
                        type_args = {"shape":None, "tags":['dymos.static_target']}
                    else:
                        type_args = {"shape":(nn,), "tags":None}
                        val = np.tile(val, (nn,)) # Recast as vector
                    
                    if upper != None:
                        add_(var_elem["Variable"], 
                             desc = var_elem["Description"],
                             val = val,
                             lower=lower,
                             upper=upper,
                             **type_args)
                            #TODO: Add Units
                            #units= var_elem["Unit"])
                            # logging.debug("Added %s: Shape: %s, Value: %s", var_elem["Variable"], str(type_args["shape"]), str(val))
                    
                    else:
                            
                        add_(var_elem["Variable"], 
                             desc = var_elem["Description"],
                             val = val,
                             **type_args)
                            #TODO: Add Units
                            #units= var_elem["Unit"])
                            # logging.debug("Added %s: Shape: %s, Value: %s", var_elem["Variable"], str(type_args["shape"]), str(val))
                

    ### INPUTS ###
    add_ = self.add_input
    for var in meta["Function"]["Inputs"]:
        var_meta = meta["Variable"][var] # get metadata
        argvar_meta = meta["Function"]['AllArgVars'] # get metadata
        add_vars(add_, var, var_meta,sv,nn,argvar_meta) # add inputs
                                
        
    ### OUTPUTS ###
    add_ = self.add_output
    for var in meta["Function"]["Outputs"]:
        # Test if output is a derivative and get metadata
        (der_test, der_parent) = isDerivative(var)
        if der_test:
            var = der_parent
            var_meta = DerivativeMetadata(meta["Variable"][var])
        else:
            var_meta = meta["Variable"][var]      
        add_vars(add_, var, var_meta,sv,nn) # add outputs
        
    # get additional data for the outputs        
    out_var = meta["Function"]["Outputs"][0]
    (der_test, der_parent) = isDerivative(out_var)
    if der_test:
        out_var_meta = DerivativeMetadata(meta["Variable"][der_parent])
    else:
        out_var_meta = meta["Variable"][out_var]
    out_names = [x["Variable"] for x in out_var_meta["Variables"]]
    
    self._out_var_elems = out_names
    self._out_var_meta = out_var_meta
                                
def setupPartials(self):
    meta = self.options["Metadata"]
    nn = self.options['num_nodes']
    sv = self.options['StaticVars']
    out_var_meta = self._out_var_meta
    
    logging.debug("Setting up partials for %s", self.options["FunctionName"])
    
    ### PARTIALS ###
    arange = np.arange(nn) # 0 ... nn
    c = np.zeros(nn)
    for var in meta["Function"]["Args"]:
        var_meta = meta["Variable"][var]
        # arg_meta = meta['Function']['ArgsVars'][var]
        if var_meta["N"]>0:
            jac_meta = meta["Jacobian"][var] # Get the Jacobian information of f w.r.t v
            if jac_meta["Flat"] == True:
                flat_meta = jac_meta["FlatMetadata"]
                
                # Calculated Derivatives
                if flat_meta["NCalc"]:
                    rCalc = np.array(flat_meta["rCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    cCalc = np.array(flat_meta["cCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    for i in range(flat_meta["NCalc"]):                                               
                        of = out_var_meta["Variables"][rCalc[i]]["Variable"]
                        wrt = meta["Variable"][var]["Variables"][cCalc[i]]["Variable"]
                        rows = arange
                        cols = (c if ((var in sv) or (wrt in sv)) else arange)
                        self.declare_partials(of=of, wrt=wrt, rows=rows, cols=cols, method='exact')
                        logging.debug("Declared Calculated Partial of %s w.r.t. %s with sparsity pattern rows=%s, cols=%s", of, wrt, str(rows), str(cols))
                
                # Constant Derivatives
                if flat_meta["NConst"]:
                    rConst = np.array(flat_meta["rConst"]) - 1 # Convert from 1 indexing to 0 indexing
                    cConst = np.array(flat_meta["cConst"]) - 1 # Convert from 1 indexing to 0 indexing
                    for i in range(flat_meta["NConst"]):                     
                        of = out_var_meta["Variables"][rConst[i]]["Variable"]
                        wrt = meta["Variable"][var]["Variables"][cConst[i]]["Variable"]
                        rows = arange
                        cols = (c if ((var in sv) or (wrt in sv)) else arange)
                        val = flat_meta["valConst"][i]
                        self.declare_partials(of=of, wrt=wrt, rows=rows, cols=cols, val=val, method='exact')
                        logging.debug("Declared Constant Partial of %s w.r.t. %s with value %s with sparsity pattern rows=%s, cols=%s", of, wrt, val, str(rows), str(cols))
            else:
                # TODO: How do we handle the non-flat case?
                raise ValueError("Flat Jacobians Required")
    pass