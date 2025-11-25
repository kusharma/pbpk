"""
4-mode dissolution comparison for UA-like settings.

All modes share the same solubility ceiling; they differ only in kinetics:

    1) MM:           simple first-order undersaturation (constant area)
    2) Mech k_diss:  shrinking-core, solubility-limited, no D/h, gamma=1
    3) Mech D/h:     Noyes–Whitney (D/h), gamma=1
    4) Mech D/h + γ: Noyes–Whitney (D/h) + supersaturation exponent
"""

import numpy as np
import matplotlib.pyplot as plt
from comparator.plot_config import (
    set_pub_defaults,
    apply_standard_2panel_layout,
    set_common_time_axis,
    PUB_FONT_SIZES,
    PUB_FIGSIZE_2PANEL,
)

# -------------------------------------------------------------------
# UA-like dose & solubility
# -------------------------------------------------------------------
MW_UA   = 228.2
DOSE_MG = 500.0
A_DOSE  = DOSE_MG * 1000.0 / MW_UA       # total dose [µmol]

V_SI    = 0.5                            # L
C_EQ    = 0.12                           # µmol/mL (~120 µM)
CAPACITY = C_EQ * (V_SI * 1000.0)        # max dissolved [µmol]

print(f"Total dose: {A_DOSE:.1f} µmol")
print(f"Capacity:   {CAPACITY:.1f} µmol at solubility")

A_insol0 = A_DOSE
A_sol0   = 0.0

T_H  = 24.0
DT_H = 0.02
T    = np.arange(0.0, T_H + DT_H, DT_H)


# -------------------------------------------------------------------
# MM-style but solubility-limited
# -------------------------------------------------------------------
def dissolution_mm(t, A_insol, A_sol, params):
    k_mm = params["k_mm"]
    S0   = params["S0_MM"]
    V_L  = params["V_SI"]
    Ceq  = params["C_eq"]

    V_mL = V_L * 1000.0
    C_sol = A_sol / V_mL

    driving = max(Ceq - C_sol, 0.0)
    J_diss = k_mm * S0 * driving

    dA_insol = -J_diss
    dA_sol   = +J_diss
    return dA_insol, dA_sol


# -------------------------------------------------------------------
# Mechanistic dissolution (shrinking core + D/h + gamma)
# -------------------------------------------------------------------
def dissolution_mech(t, A_insol, A_sol, params):
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
    C_sol  = A_sol / V_mL

    # shrinking-core surface area
    if A_insol > 0 and A0 > 0:
        S = S0 * (A_insol / A0) ** (2.0 / 3.0)
    else:
        S = 0.0

    # effective k_diss from D/h, otherwise fallback
    if (D_ua is not None) and (h_diff is not None) and (h_diff > 0):
        k_diss_eff = (D_ua / h_diff) * 3600.0
    else:
        k_diss_eff = k_diss_fallback

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


# -------------------------------------------------------------------
# Generic simulator
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Parameter sets for the four modes
# -------------------------------------------------------------------
base_mech = dict(
    V_SI = V_SI,
    C_eq = C_EQ,
    S0   = 200.0,
    A0   = A_insol0,
    k_diss = 5.0,
    k_prec = 0.3,
    gamma_prec = 1.0,
)

params_MM = {
    "k_mm": 0.5,
    "S0_MM": 200.0,
    "V_SI": V_SI,
    "C_eq": C_EQ,
}

params_kd = base_mech.copy()
params_kd["D_ua"] = None
params_kd["h_diff"] = None
params_kd["gamma_prec"] = 1.0

params_Dhg = base_mech.copy()
params_Dhg["D_ua"] = 5e-6
params_Dhg["h_diff"] = 70e-4
params_Dhg["gamma_prec"] = 2.0


# -------------------------------------------------------------------
# Run all four modes
# -------------------------------------------------------------------
ins_MM,  sol_MM  = run_sim(dissolution_mm,   A_insol0, A_sol0, params_MM,  T)
ins_kd,  sol_kd  = run_sim(dissolution_mech, A_insol0, A_sol0, params_kd,  T)
ins_Dhg, sol_Dhg = run_sim(dissolution_mech, A_insol0, A_sol0, params_Dhg, T)


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=PUB_FIGSIZE_2PANEL, sharex=True)
set_pub_defaults()
apply_standard_2panel_layout(fig)

ax = axes[0]
ax.plot(T, ins_MM,   label="MM", linewidth=2)
ax.plot(T, ins_kd,   label="Mech (k_diss only)")
ax.plot(T, ins_Dhg,  label="Mech (D/h + γ)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Undissolved UA (µmol)", fontsize=PUB_FONT_SIZES["axes"])
ax.set_title("Undissolved in SI", loc="left", fontsize=PUB_FONT_SIZES["title"] + 2)
ax.set_ylim(0, 2200)
set_common_time_axis(ax)
ax.legend(fontsize=PUB_FONT_SIZES["legend"])

ax = axes[1]
ax.plot(T, sol_MM,   label="MM", linewidth=2)
ax.plot(T, sol_kd,   label="Mech (k_diss only)")
ax.plot(T, sol_Dhg,  label="Mech (D/h + γ)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Dissolved UA (µmol)", fontsize=PUB_FONT_SIZES["axes"])
ax.set_title("Dissolved in SI", loc="left", fontsize=PUB_FONT_SIZES["title"] + 2)
set_common_time_axis(ax)
ax.legend(fontsize=PUB_FONT_SIZES["legend"])

fig.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()
