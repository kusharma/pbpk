"""
dh_gamma_sweep.py

Mechanistic dissolution sweep for UA in SI:
    - Varies D/h (via D_ua) and supersaturation exponent gamma_prec
    - k_diss and k_prec are held fixed.

Standalone:
    - Does NOT import mechanistic_dissolution or PBPK code.
    - Uses a simple 2-state ODE: undissolved (A_insol), dissolved (A_sol).
"""

import numpy as np
import matplotlib.pyplot as plt
from comparator.plot_config import (SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.15, -0.15
# If/where set_panel_title(..., panel='A'/ 'B'), include x_pos_A/x_pos_B as needed.


# ============================================================
# UA-like constants (same as in dissolution_modes.py)
# ============================================================

MW_UA   = 228.2      # g/mol
DOSE_MG = 500.0      # mg
A_DOSE  = DOSE_MG * 1000.0 / MW_UA   # total dose [µmol]

V_SI    = 0.5        # L
C_EQ    = 0.12       # µmol/mL (~120 µM)
CAPACITY = C_EQ * (V_SI * 1000.0)    # max dissolved [µmol]

print(f"[D/h sweep] Total dose: {A_DOSE:.1f} µmol")
print(f"[D/h sweep] Capacity:   {CAPACITY:.1f} µmol at solubility in SI")

OVERSAT_RATIO = 1.2
A_sol0 = min(CAPACITY * OVERSAT_RATIO, A_DOSE * 0.25)
A_insol0 = A_DOSE - A_sol0

T_H  = 24.0
DT_H = 0.05
T    = np.arange(0.0, T_H + DT_H, DT_H)


# ============================================================
# Mechanistic dissolution (shrinking core + D/h + gamma)
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

    All fluxes in µmol/h; A_insol, A_sol in µmol.
    """
    S0     = params["S0"]
    A0     = params["A0"]
    V_L    = params["V_SI"]
    Ceq    = params["C_eq"]
    k_diss_fallback = params["k_diss"]   # [1/h] if no D/h
    k_prec = params["k_prec"]           # [1/h]
    gamma  = params["gamma_prec"]       # dimensionless

    D_ua   = params.get("D_ua", None)   # [cm^2/s]
    h_diff = params.get("h_diff", None) # [cm]

    V_mL   = V_L * 1000.0
    C_sol  = A_sol / V_mL    # µmol/mL

    # shrinking-core surface area
    if A_insol > 0 and A0 > 0:
        S = S0 * (A_insol / A0) ** (2.0 / 3.0)
    else:
        S = 0.0

    # effective k_diss from D/h, otherwise fallback [1/h]
    if (D_ua is not None) and (h_diff is not None) and (h_diff > 0):
        k_diss_eff = (D_ua / h_diff) * 3600.0
    else:
        k_diss_eff = k_diss_fallback

    # dissolution
    driving = max(Ceq - C_sol, 0.0)
    J_diss = k_diss_eff * S * driving
    if J_diss < 0:
        J_diss = 0.0

    # precipitation (supersaturation only)
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
# Simple Euler integrator for the 2-state ODE
# ============================================================

def run_sim(model_func, A_insol0, A_sol0, params, t_grid):
    A_ins = np.zeros_like(t_grid)
    A_sol = np.zeros_like(t_grid)
    A_ins[0] = A_insol0
    A_sol[0] = A_sol0
    dt = t_grid[1] - t_grid[0]

    for i in range(1, len(t_grid)):
        dAi, dAs = model_func(t_grid[i - 1], A_ins[i - 1], A_sol[i - 1], params)
        A_ins[i] = max(A_ins[i - 1] + dAi * dt, 0.0)
        A_sol[i] = max(A_sol[i - 1] + dAs * dt, 0.0)
    return A_ins, A_sol


# ============================================================
# Parameter grids (D/h + gamma)
# ============================================================

# Fix these to UA-like central values
BASE_S0    = 200.0      # cm^2
BASE_KDISS = 5.0        # 1/h
BASE_KPREC = 0.3        # 1/h
H_DIFF     = 70e-4      # 70 µm = 0.007 cm

# Diffusion coefficients [cm^2/s] (spread wider for visible change)
D_values      = [5e-7, 2e-6, 5e-5]   # rows
gamma_values  = [1.0, 2.0, 4.0]      # columns


def run_dh_gamma_sweep():
    print("[D/h sweep] Starting D_ua vs gamma_prec sweep...")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(D_values),
        figsize=(15, 4),
        sharey=True,
    )
    fig.suptitle("Mechanistic Dissolution – diffusion vs γ", fontsize=12)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(gamma_values)))

    for ax, D_ua in zip(np.atleast_1d(axes), D_values):
        for color, gamma in zip(colors, gamma_values):
            params = dict(
                V_SI=V_SI,
                C_eq=C_EQ,
                S0=BASE_S0,
                A0=A_insol0,
                k_diss=BASE_KDISS,
                k_prec=BASE_KPREC,
                gamma_prec=gamma,
                D_ua=D_ua,          # <-- KEY: varies for subplot
                h_diff=H_DIFF,
            )
            _, A_sol = run_sim(dissolution_mech, A_insol0, A_sol0, params.copy(), T)
            ax.plot(T, A_sol, lw=1.4, color=color, label=f"γ={gamma}")
        ax.set_title(f"D = {D_ua:.1e} cm²/s")
        ax.set_xlabel("Time (h)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Supersaturation", loc="upper right", framealpha=0.8)
    axes[0].set_ylabel("Dissolved UA (µmol)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()
    print("[D/h sweep] Done.")


if __name__ == "__main__":
    run_dh_gamma_sweep()
