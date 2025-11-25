"""
plasma_sweep.py

PBPK sweep on plasma UA (ABUA) for different mechanistic dissolution
parameters (k_diss, k_prec).

Design goals:
    - Only depend on stable public APIs:
          parameters.get_default_parameters
          simulate.run_simulation
          model.UA_COMPARTMENTS
    - No use of flows_original, no custom builders like build_default_params.
    - If USE_MECH_DISSOLUTION is False, this script has no effect on the
      underlying PBPK equations.
"""

import numpy as np
import matplotlib.pyplot as plt

from pbpk import (
        get_default_parameters,
        UA_COMPARTMENTS,
        get_initial_conditions,
        run_simulation,
        get_initial_conditions  )

from comparator.plot_config import (SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.15, -0.15
# If/where set_panel_title(..., panel='A'/ 'B'), add x_pos_A / x_pos_B as needed.


# Which compartment is plasma/blood UA?
ABUA_IDX = UA_COMPARTMENTS.index("ABUA")
ASIUA_IDX = UA_COMPARTMENTS.index("ASIUA")
ASIUA_INSOL_IDX = UA_COMPARTMENTS.index("ASIUAinsol")


# Parameter grids for sweep
K_DISS_VALUES = [0.5, 2.0, 8.0]    # 1/h (spread wider so plasma signal shifts)
K_PREC_VALUES = [0.05, 0.5, 5.0]   # 1/h (legend)
SUPERSAT_RATIO = 1.4               # seed SI lumen above solubility to activate precipitation


def run_plasma_sweep(t_end=24.0, dt=0.05):
    """
    For each (k_diss, k_prec) pair:
        - clone default params
        - enable mechanistic dissolution
        - set k_diss, k_prec
        - run PBPK
        - plot ABUA vs time
    """
    print("[Plasma sweep] starting...")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(K_DISS_VALUES),
        figsize=(15, 4),
        sharey=True,
    )
    fig.suptitle("PBPK plasma UA sweep", fontsize=12)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(K_PREC_VALUES)))

    for ax, k_diss in zip(np.atleast_1d(axes), K_DISS_VALUES):
        for color, k_prec in zip(colors, K_PREC_VALUES):
            params = get_default_parameters()
            params["USE_MECH_DISSOLUTION"] = True
            params["k_diss"] = k_diss
            params["k_prec"] = k_prec
            params["gamma_prec"] = 2.0
            params["USE_DIFFUSION_LAYER"] = False

            # Force mild supersaturation so k_prec has a visible impact
            y0 = get_initial_conditions(params)
            baseline_sol = y0[ASIUA_IDX]
            target_sol = SUPERSAT_RATIO * params["parent"]["SOL"] * params["VSI"]
            delta = max(target_sol - baseline_sol, 0.0)
            y0[ASIUA_IDX] += delta
            y0[ASIUA_INSOL_IDX] = max(y0[ASIUA_INSOL_IDX] - delta, 0.0)

            t, Y = run_simulation(params=params, y0=y0, t_end=t_end, dt=dt)
            A_plasma = Y[:, ABUA_IDX]

            ax.plot(t, A_plasma, lw=1.4, color=color, label=f"k_prec={k_prec}")

        ax.set_title(f"k_diss = {k_diss} 1/h")
        ax.set_xlabel("Time (h)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Precipitation", loc="upper right", framealpha=0.8)

    axes[0].set_ylabel("ABUA (µmol)")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()
    print("[Plasma sweep] done.")


if __name__ == "__main__":
    run_plasma_sweep()
