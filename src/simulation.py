# simulation.py

import numpy as np
from scipy.integrate import solve_ivp

from parameters import get_default_parameters
from model import pbpk_ode_system, get_initial_conditions


def run_simulation(params=None, y0=None, t_end=24.0, dt=0.01):
    """
    Runs the PBPK model ODE system.

    params : dict
        Parameter dictionary. If None → load defaults.
    y0 : array
        Initial state vector. If None → computed from params.
    t_end : float
        Total simulation time (hours)
    dt : float
        Output sampling interval (hours)
    """

    if params is None:
        params = get_default_parameters()

    if y0 is None:
        y0 = get_initial_conditions(params)

    t_eval = np.arange(0, t_end + dt, dt)
    t_eval = t_eval[t_eval <= t_end]
    if len(t_eval) == 0:
        t_eval = np.array([0.0])
    elif t_eval[-1] < t_end:
        t_eval = np.append(t_eval, t_end)

    sol = solve_ivp(
        lambda t, y: pbpk_ode_system(t, y, params),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )

    return sol.t, sol.y.T

