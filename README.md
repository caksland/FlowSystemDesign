# FlowSystemDesign

This model is used to design a flow system. Currently, the code doesn't 
optimize any control or parameters values. It only converges the 
ODE dynamics.
The setup times are really long and strongly dependent on the number of nodes 
in the phase nn. eg. when nn=5, the setup time is about ~7s. when nn=20, the 
setup time inceasese to ~150s.
The mjority of the setup time happens when 
transcription.configure_timeseries_outputs(self) is called in phase.py 
configure(self) function (149 of the 150s when nn=20)

Package versions:
dymos 1.2.0
openmdao 3.16.0

NOTE 1. there is a manual bug fix that is required for the code to run.
in trajectory.py, the the add_parameter function definition needs to 
specify None units as default:
e.g.
def add_parameter(self, name, units=None, val=_unspecified, desc=_unspecified, opt=False,

NOTE 2. I've also run the code in the latest version of dymos and openmdao
dymos 1.6.1
openmdao 3.23.0
and the setup time is still slow. However, the optimization does not complete.
