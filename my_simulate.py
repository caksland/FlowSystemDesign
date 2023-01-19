# from collections import OrderedDict
# from collections.abc import Sequence
# from copy import deepcopy
# import itertools
# import warnings
try:
    from itertools import izip
except ImportError:
    izip = zip
# import numpy as np

import openmdao.api as om
from dymos.trajectory.trajectory import Trajectory
# from openmdao.utils.mpi import MPI

# from ..utils.constants import INF_BOUND

# from .options import LinkageOptionsDictionary
# from .phase_linkage_comp import PhaseLinkageComp
# from ..phase.options import TrajParameterOptionsDictionary
# from ..utils.misc import get_rate_units, _unspecified
from dymos.utils.misc import get_rate_units, _unspecified
# from ..utils.introspection import get_source_metadata

def my_simulate(self, times_per_seg=10, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None):
        """
        Simulate the Trajectory using scipy.integrate.solve_ivp.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally spaced times per segment at which output is requested.  If None,
            output will be provided at all Nodes.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        sim_traj = Trajectory(sim_mode=True)

        for name, phs in self._phases.items():
            sim_phs = phs.get_simulation_phase(times_per_seg=times_per_seg, method=method,
                                               atol=atol, rtol=rtol, first_step=first_step,
                                               max_step=max_step)
            sim_traj.add_phase(name, sim_phs)

        sim_traj.parameter_options.update(self.parameter_options)

        sim_prob = om.Problem(model=om.Group())

        traj_name = self.name if self.name else 'sim_traj'
        sim_prob.model.add_subsystem(traj_name, sim_traj)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.add_recorder(rec)
            # record_inputs is needed to capture potential input parameters that aren't connected
            sim_prob.recording_options['record_inputs'] = True
            # record_outputs is need to capture the timeseries outputs
            sim_prob.recording_options['record_outputs'] = True

        sim_prob.setup()

        # Assign trajectory parameter values
        param_names = [key for key in self.parameter_options.keys()]
        cnt = 1
        for name in param_names:
            prom_path = f'{self.name}.parameters:{name}'
            src = self.get_source(prom_path)

            # We use this private function to grab the correctly sized variable from the
            # auto_ivc source.
            val = self._abs_get_val(src, False, None, 'nonlinear', 'output', False, from_root=True)
            sim_prob_prom_path = f'{traj_name}.parameters:{name}'
            if cnt<40:
                sim_prob[sim_prob_prom_path][...] = val[cnt-1]
            elif cnt == 43:
                sim_prob[sim_prob_prom_path][...] = val[0:7]    
            else:
                sim_prob[sim_prob_prom_path][...] = val
            cnt = cnt+1

        for phase_name, phs in sim_traj._phases.items():
            skip_params = set(param_names)
            for name in param_names:
                targets = self.parameter_options[name]['targets']
                if targets and phase_name in targets:
                    targets_phase = targets[phase_name]
                    if targets_phase is not None:
                        if isinstance(targets_phase, str):
                            targets_phase = [targets_phase]
                        skip_params = skip_params.union(targets_phase)

            phs.initialize_values_from_phase(sim_prob, self._phases[phase_name],
                                             phase_path=traj_name,
                                             skip_params=skip_params)

        print('\nSimulating trajectory {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating trajectory {0}'.format(self.pathname))
        if record_file:
            sim_prob.record('final')
        sim_prob.cleanup()

        return sim_prob