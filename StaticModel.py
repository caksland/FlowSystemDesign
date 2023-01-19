# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:28:42 2021

@author: renkert2
"""
import numpy as np
import openmdao.api as om
import os
import sys
from SharedFunctions import *
import logging
import copy

class StaticProblem(om.Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def Metadata(self):
        return self.model.Metadata
    
    
    def setInitialValues(self): 
        # - setInitialValues():
            # - calls problem.set_val() to set the value of each variable to the 
            # value in its metadata. 
        var_meta = self.Metadata["Variable"]
        
        for var in var_meta:
            if var_meta[var]["N"]:
                for elem in var_meta[var]["Variables"]:
                    if "Value" in elem:
                        #TODO: Add Units here
                        self.set_val(elem["Variable"], val = elem["Value"])
                        
    def getData(self):
        # - getData():
            # - calls problem.get_val() to get the current variable values and 
            # - stores it in the variable metadata dictionary
        var_meta = copy.deepcopy(self.Metadata["Variable"])
        for var in var_meta:
            if var_meta[var]["N"]:
                for elem in var_meta[var]["Variables"]:
                        elem["Value"] = self.get_val(elem["Variable"])
        return copy.deepcopy(var_meta)
        

class StaticModel(om.Group):
    # Identical to DynamicModel, except it uses an attached solver to 
    def initialize(self):
        self.options.declare('Model', types=str, default = 'None') # "Model" property used to store name of folder containing Model .py functions and variable tables
        self.options.declare('Path', types=str, default = '') # Path to directory containing model folder "Model", i.e. abs_path = Path/Model/
        self.options.declare('Functions', default=None) # List of ModelFunctions, added in order 
    def setup(self):
        mdl = self.options['Model']
        mdl_path = os.path.join(self.options["Path"], mdl)
        self.ModelPath = mdl_path
        self.Metadata = ImportMetadata(mdl_path)

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
            
            # Create an implicit ModelFunction for each component and add it to the group as a subsystem
            if f_meta["Type"] == "Implicit":
                subsys = ImplicitModelFunction(FunctionName = f, Metadata=meta)
            elif f_meta["Type"] == "Explicit":
                # Check whether the output of the function is a derivative
                outs = f_meta["Outputs"][0]
                (der_test, der_parent) = isDerivative(outs)
                
                if der_test:
                    # If output is a derivative, change to an implicit function whose output is the parent of the derivative term
                    # i.e. if the output is x_dot, the new output should be x
                    # In Python, meta contains a pointer to f_meta so changes to f_meta are reflected in meta
                    f_meta["Type"] == "Implicit"
                    f_meta["Outputs"][0] = der_parent # Replace derivative output with parent
                    f_meta["Inputs"].remove(der_parent)
                    
                    subsys = ImplicitModelFunction(FunctionName = f, Metadata=meta)
                else:
                    # If output is not a derivative, calculate explicitly as normal
                    subsys = ExplicitModelFunction(FunctionName = f, Metadata=meta)
 
            self.add_subsystem(name=f, subsys=subsys,
                           promotes = ["*"])
            
        ### SOLVERS (Top Level) ###
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)
        self.linear_solver = om.DirectSolver()
        
class ImplicitModelFunction(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('FunctionName', types=str)
        self.options.declare('Metadata', default=None, recordable=False)
        
    def setup(self):
        setupInputsOutputs(self)
        
        ### SOLVERS ###
        self.linear_solver = om.DirectSolver()
        
    def setup_partials(self):
        setupPartials(self)

    def apply_nonlinear(self, inputs, outputs, residuals):
        meta = self.options["Metadata"]
        out_var = meta["Function"]["Outputs"][0]
        out_names = [x["Variable"] for x in meta["Variable"][out_var]["Variables"]]

        arg_list = AssembleArguments(meta, inputs, outputs)
        residuals = ComputeModel(residuals, arg_list, meta, out_names)
    
    def linearize(self, inputs, outputs, partials):
        meta = self.options["Metadata"]
        out_var = meta["Function"]["Outputs"][0]
        out_names = [x["Variable"] for x in meta["Variable"][out_var]["Variables"]]
        
        # Assemble Vectors
        arg_list = AssembleArguments(meta, inputs, outputs)
        partials = ComputeModelPartials(partials, arg_list, meta, out_names)
                
class ExplicitModelFunction(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('FunctionName', types=str)
        self.options.declare('Metadata', default=None, recordable=False)
        
    def setup(self):
        setupInputsOutputs(self)
        
    def setup_partials(self):
        setupPartials(self)

    def compute(self, inputs, outputs):
        meta = self.options["Metadata"]
        out_var = meta["Function"]["Outputs"][0]
        out_names = [x["Variable"] for x in meta["Variable"][out_var]["Variables"]]

        arg_list = AssembleArguments(meta, inputs, [])
        outputs = ComputeModel(outputs, arg_list, meta, out_names)
    
    def compute_partials(self, inputs, partials):
        meta = self.options["Metadata"]
        out_var = meta["Function"]["Outputs"][0]
        out_names = [x["Variable"] for x in meta["Variable"][out_var]["Variables"]]
        
        # Assemble Vectors
        arg_list = AssembleArguments(meta, inputs, [])
        partials = ComputeModelPartials(partials, arg_list, meta, out_names)
         
### Static Model Helper Functions ###
def setupInputsOutputs(self):
    meta = self.options["Metadata"]

    ### INPUTS ###
    for var in meta["Function"]["Inputs"]:
        var_meta = meta["Variable"][var]
        if var_meta["N"]>0:
            for var_elem in var_meta["Variables"]:
                if "Value" in var_elem:
                    val = var_elem["Value"]
                else:
                    val = 1
                self.add_input(var_elem["Variable"], 
                                desc = var_elem["Description"],
                                shape = 1, 
                                val = val)
                                #TODO: Add Units
                                #units= var_elem["Unit"])
        
    ### OUTPUTS ###
    for var in meta["Function"]["Outputs"]:
        var_meta = meta["Variable"][var]
        if var_meta["N"]>0:
            for var_elem in var_meta["Variables"]:
                if "Value" in var_elem:
                    val = var_elem["Value"]
                else:
                    val = 1
                self.add_output(var_elem["Variable"], 
                                desc = var_elem["Description"],
                                shape = 1, 
                                val = val)
                                #TODO: Add Units
                                #units= var_elem["Unit"])
                                
def setupPartials(self):
    meta = self.options["Metadata"]
    out_var = meta["Function"]["Outputs"][0]
    
    logging.debug("Setting up partials for %s", self.options["FunctionName"])
    
    ### PARTIALS ###
    for var in meta["Function"]["Args"]:
        var_meta = meta["Variable"][var]
        if var_meta["N"]>0:
            jac_meta = meta["Jacobian"][var] # Get the Jacobian information of f w.r.t v
            if jac_meta["Flat"] == True:
                flat_meta = jac_meta["FlatMetadata"]
                # Calculated Derivatives
                if flat_meta["NCalc"]:
                    rCalc = np.array(flat_meta["rCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    cCalc = np.array(flat_meta["cCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    for i in range(flat_meta["NCalc"]):
                        of = meta["Variable"][out_var]["Variables"][rCalc[i]]["Variable"]
                        wrt = meta["Variable"][var]["Variables"][cCalc[i]]["Variable"]
                        self.declare_partials(of=of, wrt=wrt, method='exact')
                        logging.debug("Declared Calculated Partial of %s w.r.t. %s", of, wrt)
                
                # Constant Derivatives
                if flat_meta["NConst"]:
                    rConst = np.array(flat_meta["rConst"]) - 1 # Convert from 1 indexing to 0 indexing
                    cConst = np.array(flat_meta["cConst"]) - 1 # Convert from 1 indexing to 0 indexing
                    for i in range(flat_meta["NConst"]):
                        of = meta["Variable"][out_var]["Variables"][rConst[i]]["Variable"]
                        wrt = meta["Variable"][var]["Variables"][cConst[i]]["Variable"]
                        val = flat_meta["valConst"][i]
                        self.declare_partials(of=of, wrt=wrt, val=val)
                        logging.debug("Declared Constant Partial of %s w.r.t. %s with value %s", of, wrt, val)
            else:
                # TODO: How do we handle the non-flat case?
                raise ValueError("Flat Jacobians Required")
    