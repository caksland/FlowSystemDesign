# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:23:12 2021

@author: renkert2
"""
import importlib as impL
import json
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

### HELPER FUNCTIONS ###
def ImportMetadata(mdl):
    def importModule(name):
        # Reload a module if it has already been imported.  Allows us to import 
        # modules of the same name from different model folders.  
        if name in sys.modules:
            module = sys.modules[name]
            impL.reload(module)
        else:
            module = impL.import_module(name)
        return module
    
    sys.path.append(mdl)
    
    # Functions
    func_meta_file = open(mdl + '/FunctionMetadata.json', 'r')
    func_meta_dict = json.load(func_meta_file) # List of dictionaries with function metadata
    for f in func_meta_dict:
        module = importModule(f)
        handle = getattr(module, f)
        f_meta = func_meta_dict[f]
        f_meta["Handle"] = handle
        
        # Ensure Args, Inputs, and Outputs are Lists
        for list_key in ["Args", "Inputs", "Outputs"]:
            if not isinstance(f_meta[list_key], list):
                f_meta[list_key] = [f_meta[list_key]]
    
    # Jacobian
    jac_meta_file = open(mdl + '/JacobianMetadata.json', 'r')
    jac_meta_dict = json.load(jac_meta_file)
    for f in jac_meta_dict:
        for v in jac_meta_dict[f]:
            j = jac_meta_dict[f][v]
            partial = j["Name"]
            flat = j["Flat"]
            exp_flag = True
            if flat:
                Nc = j["FlatMetadata"]["NCalc"]
                if Nc == 0:
                    exp_flag = False
                
                # Ensure all FlatMetadata Arguments except NConst and NCalc are Lists
                for list_key in ["iCalc","rCalc","cCalc","rConst","cConst","valConst"]:
                    if not isinstance(j["FlatMetadata"][list_key], list):
                           j["FlatMetadata"][list_key] = [j["FlatMetadata"][list_key]]
            if exp_flag:
                module = importModule(partial)
                handle = getattr(module, partial)
                j["Handle"] = handle           
    # Variables
    var_meta_file = open(mdl + '/VariableMetadata.json', 'r')
    var_meta_dict = json.load(var_meta_file)
    # Ensure var_meta_dict[var]["Variables"] is a list of dictionaries
    for var in var_meta_dict:
        var_meta = var_meta_dict[var]
        if var_meta["N"]:
            if isinstance(var_meta["Variables"], dict):
                var_meta["Variables"] = [var_meta["Variables"]]

    meta_dict = {"Function":func_meta_dict, "Jacobian":jac_meta_dict, "Variable":var_meta_dict}

    sys.path.remove(mdl)
    return meta_dict

def isDerivative(name_string):
    name_string_split = name_string.split("_")
    if name_string_split[-1] == "dot":
        der_flag = True
        parent = "_".join(name_string_split[0:-1])
    else:
        der_flag = False
        parent = ""
    return der_flag, parent

def DerivativeMetadata(var_meta):
    # Converts metadata corresponding to a variable to metadata coresponding to the variable's derivative
    der_var_meta = copy.deepcopy(var_meta)
    
    for var_elem in der_var_meta["Variables"]:
        var_elem["Variable"] += "_dot"
        var_elem["Description"] += " (Rate of Change)"
        var_elem["Unit"] += "/s"
        if "Value" in var_elem:
            # Remove default value if one given
            del var_elem["Value"]
        
    return der_var_meta

def AssembleArguments(meta, inputs, outputs):
    arg_list = []
    for a in meta["Function"]["Args"]:
        # Get metadata associated with the argument
        v_meta = meta["Variable"][a]
        
        # Determine whether the argument is an input or an output
        if a in meta["Function"]["Inputs"]:
            source = inputs
        elif a in meta["Function"]["Outputs"]:
            source = outputs
        
        # Create numerical value for variable if variable contains more than zero elements
        if v_meta["N"] > 0:
            arg_list_inner = []
            for e in v_meta["Variables"]:
                if e["Variable"] in meta["Function"]["AllArgVars"]:
                    arg_list_inner.append(source[e["Variable"]])
        else:
            arg_list_inner = np.array([])
        arg_list.append(arg_list_inner)
    return arg_list

# compute the model outputs for explicit and implicit matlab models
def ComputeModel(outputs, inputs, meta, out_names):
    # Call Calc function
    f = meta["Function"]["Handle"]
    outs = f(*inputs)
    
    # Assign to Outputs
    if len(out_names) == 1:
        outputs[out_names[0]] = outs
    else:
        for i, out_var in enumerate(out_names): 
            outputs[out_var] = outs[i]
    
    return outputs

# compute the model partials for explicit and implicit matlab models
def ComputeModelPartials(partials, inputs, meta, out_names):    
    # tic=time.time()
    # Calculate and assign computed derivatives to Partials
    jac_meta = meta["Jacobian"] # Jacobian metadata associated with this function
    for a in meta["Function"]["Args"]:
        var_meta = meta["Variable"][a]
        if var_meta["N"]>0:
            J = jac_meta[a] # Get Metadata corresponding to argument
            if J["Flat"]:
                flat_meta = J["FlatMetadata"]
                if flat_meta["NCalc"]:
                    J_out = J["Handle"](*inputs) # Returns tuple of numpy.ndarray objects or floats
                    rCalc = np.array(flat_meta["rCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    cCalc = np.array(flat_meta["cCalc"]) - 1 # Convert from 1 indexing to 0 indexing
                    if flat_meta["NCalc"] == 1:
                        of = out_names[rCalc[0]]
                        wrt = meta["Variable"][a]["Variables"][cCalc[0]]["Variable"]
                        partials[of, wrt] = J_out
                    else:
                        for i in range(flat_meta["NCalc"]):
                            of = out_names[rCalc[i]]
                            wrt = meta["Variable"][a]["Variables"][cCalc[i]]["Variable"]
                            partials[of, wrt] = J_out[i]
            else:
                # TODO: How do we handle the nonflat case?
                pass
    # toc = time.time(); print('Partials Time:', toc-tic)
    pass

def plot_results(axes, title, nrows, ncols, figsize=(10, 8), p_sol=None, p_sim=None):
    
    phases= [*p_sol.model.traj._phases]
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)
    axs = axs.flatten()

    if nrows == 1:
        axs = [axs]
        
    # function to plot the full timeseries across phases           
    def plot_TS(p,marker,color,linestyle,label):
        valx = dict((phs, p.get_val(f'traj.{phs}.timeseries.{x}'.format(phs)))
                        for phs in phases)
        valy = dict((phs, p.get_val(f'traj.{phs}.timeseries.{y}'.format(phs)))
                        for phs in phases)
            
        cnt=0
        for phs in phases:
            axs[i].plot(valx[phs],
                        valy[phs],
                        marker=marker,
                        ms=4,
                        color=color,
                        linestyle=linestyle,
                        label=label if (i==0 and cnt==0) else None)
            if label=='Discrete':
                if not(Lim==None):
                    sz = len(valx[phs])
                    axs[i].plot(valx[phs],Lim[0]*np.ones(sz),color='k',linestyle='--',label='Bounds' if (i==0 and cnt==0) else None)
                    axs[i].plot(valx[phs],Lim[1]*np.ones(sz),color='k',linestyle='--')
                
            cnt = cnt+1
        
    # iterate through axes to plot data
    for i, (x, y, xlabel, ylabel, Lim) in enumerate(axes):
        
        if p_sol is not None:        
            plot_TS(p_sol,'o','b','None','Optimization')
        if p_sim is not None:
            plot_TS(p_sim,None,'r','-','Simulation')
       
        axs[i].grid(True)
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        fig.suptitle(title)
        fig.legend(loc='lower center', ncol=3)
        
    fig.tight_layout(pad=1.0,w_pad=1.0,h_pad=0.0)
    return fig, axs

def save_results(axes, name, filename, data=None):
    
    phases= [*data.model.traj._phases]
        
    # function to plot the full timeseries across phases           
    def get_data(p):
        valx = dict((phs, p.get_val(f'traj.{phs}.timeseries.{x}'.format(phs)))
                        for phs in phases)
        valy = dict((phs, p.get_val(f'traj.{phs}.timeseries.{y}'.format(phs)))
                        for phs in phases)
        
        x_data = np.empty((0,1))
        y_data = np.empty((0,1))
        for phs in phases:
            x_data = np.concatenate((x_data,valx[phs]))
            y_data = np.concatenate((y_data,valy[phs]))

        return x_data,y_data        
    # iterate through axes to plot data
    data_dict = dict()
    for i, (x, y, xlabel, ylabel, Lim) in enumerate(axes):
        
        data_vec = get_data(data)
        data_dict.update({name[i]+'x':data_vec[0],name[i]+'y':data_vec[1]})
        
    savemat(filename, data_dict)  










