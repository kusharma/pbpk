"""
motility_sweep.py

Motility-modulated SI transit and dissolution sweep (1×3 layout).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pbpk import (
        get_default_parameters,
        UA_COMPARTMENTS,
        run_simulation,
        calculate_ua_concentrations)

from src.flows import calculate_gi_flows, calculate_ua_concentrations

from comparator.plot_config import (
    set_pub_defaults,
    set_panel_title,
    PUB_FONT_SIZES,
    PUB_FIGSIZE_2PANEL,
    time_axis_locators,
    SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B, SUBPLOT_TITLE_X_C
)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B, SUBPLOT_TITLE_X_C = -0.15, -0.13, -0.13

MOTILITY_AMP_VALUES = [0.0, 0.5, 1.0, 2.0]
ASIUA_IDX = UA_COMPARTMENTS.index("ASIUA")
ASIUA_INSOL_IDX = UA_COMPARTMENTS.index("ASIUAinsol")
ALIUA_IDX = UA_COMPARTMENTS.index("ALIUA")

def compute_panel_timeseries(t, Y, params):
    n = len(t)
    M = np.zeros(n)
    ASIUA = np.zeros(n)
    transSitoG = np.zeros(n)
    for i in range(n):
        params["_time"] = t[i]
        y = Y[i, :]
        n_ua = len(UA_COMPARTMENTS)
        A_ua = y[:n_ua]
        params['ABUA'] = A_ua[UA_COMPARTMENTS.index("ABUA")]
        C_ua = calculate_ua_concentrations(params, A_ua)
        # Motility index
        M[i] = flows.motility_index(params)
        ASIUA[i] = A_ua[ASIUA_IDX]
        F_gi = flows.calculate_gi_flows(
            params,
            A_ua[ASIUA_IDX], C_ua["CSIUA"], A_ua[ASIUA_INSOL_IDX], C_ua["CSIUAinsol"],
            A_ua[ALIUA_IDX], C_ua["CLIUA"]
        )
        transSitoG[i] = F_gi.get("transSitoG", 0.0)
    return M, ASIUA, transSitoG

def run_motility_sweep_1x3(t_end=24.0, dt=0.1):
    print("[Motility sweep] 1×3 panel")
    set_pub_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(PUB_FIGSIZE_2PANEL[0]*1.5, PUB_FIGSIZE_2PANEL[1]))
    fig.subplots_adjust(wspace=0.28)
    ax_M = axes[0]
    ax_asiua = axes[1]
    ax_flux = axes[2]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(MOTILITY_AMP_VALUES)))
    for color, amp in zip(colors, MOTILITY_AMP_VALUES):
        params = get_default_parameters()
        params["USE_MOTILITY_MODULATION"] = True
        params["MOTILITY_AMP"] = amp
        params["MOTILITY_T_PEAK"] = 1.0
        params["MOTILITY_DECAY"] = 3.0
        params["MOTILITY_TRANSIT_ALPHA"] = 1.0   # Example: strong coupling
        t, Y = run_simulation(params=params, t_end=t_end, dt=dt)
        M, ASIUA, transSitoG = compute_panel_timeseries(t, Y, params)
        ax_M.plot(t, M, lw=2, color=color, label=f"AMP={amp:.1f}")
        ax_asiua.plot(t, ASIUA, lw=2, color=color)
        ax_flux.plot(t, transSitoG, lw=2, color=color)
    total_hours, major_loc, minor_loc = time_axis_locators(t_end * 60.0)
    for ax in axes:
        ax.set_xlim(0, total_hours)
        ax.xaxis.set_major_locator(major_loc)
        ax.xaxis.set_minor_locator(minor_loc)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(axis="x", direction="out", labelsize=PUB_FONT_SIZES["ticks"] + 1)
        ax.tick_params(axis="y", direction="out", labelsize=PUB_FONT_SIZES["ticks"] + 1)
    ax_M.legend(fontsize=PUB_FONT_SIZES["legend"], loc="upper right", framealpha=0.92)
    set_panel_title(ax_M, "A", panel="A", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_A=SUBPLOT_TITLE_X_A)
    set_panel_title(ax_asiua, "B", panel="B", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_B=SUBPLOT_TITLE_X_B)
    set_panel_title(ax_flux, "C", panel="C", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_C=SUBPLOT_TITLE_X_C)
    for ax in axes:
        ax.set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_M.set_ylabel("Motility Index M(t) (dimensionless)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_asiua.set_ylabel("UA in Small Intestinal Lumen (µmol)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_flux.set_ylabel("Absorption Flux to Liver (µmol/h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "motility_amp_sweep_1x3.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Motility sweep] Saved to {output_path}")
    print("[Motility sweep] Done.")

if __name__ == "__main__":
    run_motility_sweep_1x3()