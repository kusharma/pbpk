"""
dissolution_sweep.py

Parameter sweep for mechanistic dissolution of Urolithin A (UA) in SI.

- Uses the same mechanistic model as dissolution_modes.py
- Standalone: no dependency on simulate/model/parameters/flows
- Sweeps over:
    k_diss ∈ [2, 5, 10]  [1/h]
    k_prec ∈ [0.1, 0.5, 1.0]  [1/h]
    S0     ∈ [100, 200, 400]  [cm²]

For each S0, produces a 3×3 grid of subplots:
    rows    = k_diss
    columns = k_prec
    y-axis  = dissolved UA (µmol) vs time
"""

import numpy as np
import matplotlib.pyplot as plt

from comparator.plot_config import (
    SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B
)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.15, -0.15
# If/where set_panel_title(..., panel='A'/ 'B'), add x_pos_A / x_pos_B as needed.


# ============================================================
# UA-like constants (same logic as in dissolution_modes.py)
# ============================================================

MW_UA   = 228.2      # g/mol
DOSE_MG = 500.0      # mg
A_DOSE  = DOSE_MG * 1000.0 / MW_UA   # total dose [µmol]

V_SI    = 0.5        # L
C_EQ    = 0.12       # µmol/mL (~120 µM)
CAPACITY = C_EQ * (V_SI * 1000.0)    # max dissolved [µmol]

print(f"[Sweep] Total dose: {A_DOSE:.1f} µmol")
print(f"[Sweep] Capacity:   {CAPACITY:.1f} µmol at solubility in SI")

OVERSAT_RATIO = 1.2  # allow transient supersaturation to highlight precipitation
A_sol0 = min(CAPACITY * OVERSAT_RATIO, A_DOSE * 0.25)
A_insol0 = A_DOSE - A_sol0

T_H  = 24.0
DT_H = 0.05
T    = np.arange(0.0, T_H + DT_H, DT_H)


# ============================================================
# Mechanistic dissolution (same structure as dissolution_modes)
# ============================================================

def dissolution_mech(t, A_insol, A_sol, params):
    """
    Mechanistic dissolution model for UA in SI:

        - Shrinking-core area:
            S = S0 * (A_insol / A0)^(2/3)

        - Dissolution:
            J_diss = k_diss_eff * S * max(Ceq - C, 0)

        - Precipitation (when supersaturated):
            R_prec = k_prec * (C/Ceq - 1)^gamma * A_sol

    All in µmol/h.
    """
    S0     = params["S0"]
    A0     = params["A0"]
    V_L    = params["V_SI"]
    Ceq    = params["C_eq"]
    k_diss_fallback = params["k_diss"]
    k_prec = params["k_prec"]
    gamma  = params["gamma_prec"]

    D_ua   = params.get("D_ua", None)
    h_diff = params.get("h_diff", None)

    V_mL   = V_L * 1000.0
    C_sol  = A_sol / V_mL    # µmol/mL

    # shrinking-core surface area
    if A_insol > 0 and A0 > 0:
        S = S0 * (A_insol / A0) ** (2.0 / 3.0)
    else:
        S = 0.0

    # effective k_diss from D/h, otherwise fallback
    if (D_ua is not None) and (h_diff is not None) and (h_diff > 0):
        k_diss_eff = (D_ua / h_diff) * 3600.0  # [1/h]
    else:
        k_diss_eff = k_diss_fallback

    # dissolution
    driving = max(Ceq - C_sol, 0.0)
    J_diss = k_diss_eff * S * driving
    if J_diss < 0:
        J_diss = 0.0

    # precipitation
    R_prec = 0.0
    if Ceq > 0 and A_sol > 0:
        ss_ratio = C_sol / Ceq
        if ss_ratio > 1.0:
            R_prec = k_prec * (ss_ratio ** gamma - 1.0) * A_sol
            if R_prec < 0:
                R_prec = 0.0

    dA_insol = -J_diss + R_prec
    dA_sol   = +J_diss - R_prec
    return dA_insol, dA_sol


# ============================================================
# Generic Euler simulator
# ============================================================

def run_sim(model_func, A_insol0, A_sol0, params, t_grid):
    A_ins = np.zeros_like(t_grid)
    A_sol = np.zeros_like(t_grid)
    A_ins[0] = A_insol0
    A_sol[0] = A_sol0
    dt = t_grid[1] - t_grid[0]

    for i in range(1, len(t_grid)):
        dAi, dAs = model_func(t_grid[i-1], A_ins[i-1], A_sol[i-1], params)
        A_ins[i] = max(A_ins[i-1] + dAi * dt, 0.0)
        A_sol[i] = max(A_sol[i-1] + dAs * dt, 0.0)
    return A_ins, A_sol


# ============================================================
# Parameter grids
# ============================================================

k_diss_values = [2.0, 5.0, 12.0]      # [1/h]
k_prec_values = [0.1, 1.0, 10.0]      # [1/h]
S0_values     = [60.0, 200.0, 800.0]  # [cm²]


# ============================================================
# Main sweep
# ============================================================

def run_dissolution_sweep():
    print("[Sweep] Starting parameter sweep...")

    for S0 in S0_values:
        fig, axes = plt.subplots(
            nrows=len(k_diss_values),
            ncols=len(k_prec_values),
            figsize=(12, 9),
            sharex=True,
            sharey=True,
        )
        fig.suptitle(f"Mechanistic Dissolution Sweep – S0 = {S0:.0f} cm²", fontsize=13)

        for i, k_diss in enumerate(k_diss_values):
            for j, k_prec in enumerate(k_prec_values):
                ax = axes[i, j]
                params = dict(
                    V_SI=V_SI,
                    C_eq=C_EQ,
                    S0=S0,
                    A0=A_insol0,
                    k_diss=k_diss,
                    k_prec=k_prec,
                    gamma_prec=1.0,
                    D_ua=None,
                    h_diff=None,
                )
                _, A_sol = run_sim(dissolution_mech, A_insol0, A_sol0, params, T)
                ax.plot(T, A_sol, lw=1.4, color="#2c7fb8")
                ax.set_title(f"k_diss={k_diss} 1/h\nk_prec={k_prec}", fontsize=9)
                if i == len(k_diss_values) - 1:
                    ax.set_xlabel("Time (h)")
                if j == 0:
                    ax.set_ylabel("Dissolved UA (µmol)")
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.show()

    print("[Sweep] Done.")


if __name__ == "__main__":
    run_dissolution_sweep()
