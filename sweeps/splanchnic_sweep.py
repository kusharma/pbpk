"""
splanchnic_sweep.py

Parameter sweep for time-varying splanchnic blood flow multiplier.

Sweeps over SPLANCHNIC_AMP values and plots in a 1×2 layout:
    A. Effective Splanchnic Blood Flow to Gut (QG_eff)
    B. UA Delivery to Liver From Gut (venG)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pbpk import (
        get_default_parameters,
        UA_COMPARTMENTS,
        run_simulation,
        calculate_ua_concentrations,
    )
from src import flows

from comparator.plot_config import (
    set_pub_defaults,
    set_panel_title,
    PUB_FONT_SIZES,
    PUB_FIGSIZE_2PANEL,
    time_axis_locators,
    apply_standard_2panel_layout, SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B
)

from src.config import AMP_VALUES, AMP_LABELS, HYPEREMIA_SHAPE

SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.175, -0.15

# Sweep values (meal amplitudes) and legend labels
SPLANCHNIC_AMP_VALUES = AMP_VALUES
LEGEND_LABELS = AMP_LABELS

# Hyperemia shape parameters used for this sweep (can be tuned)
SPLANCHNIC_ONSET_SWEEP = HYPEREMIA_SHAPE["onset"]
SPLANCHNIC_TAU_RISE_SWEEP = HYPEREMIA_SHAPE["tau_rise"]

# Simple venous smoothing to avoid instantaneous spikes (first-order portal mixing)
VEN_G_SMOOTH_TAU = 0.3           # h (~18 min)
VEN_G_SMOOTH_GAMMA = 2.0         # dimensionless shape for the ramp


def calculate_flows_from_state(t, Y, params):
    """
    Calculate effective splanchnic flows and gut-to-liver flux from state variables.
    
    Returns:
        QG_eff: effective gut arterial flow (L/h)
        venG: gut venous flow (UA flux to liver, µmol/h)
    """
    n_points = len(t)
    QG_eff = np.zeros(n_points)
    venG = np.zeros(n_points)
    
    for i in range(n_points):
        y = Y[i, :]
        n_ua = len(UA_COMPARTMENTS)
        A_ua = y[:n_ua]
        
        params['ABUA'] = A_ua[UA_COMPARTMENTS.index("ABUA")]
        
        # Calculate concentrations
        C_ua = calculate_ua_concentrations(params, A_ua)
        
        # Get splanchnic multiplier
        params["_time"] = t[i]
        M_spl = flows.splanchnic_flow_multiplier(params)
        QG_eff[i] = params["QG"] * M_spl
        
        # Calculate blood flows
        F_blood = flows.calculate_blood_flows(
            params,
            C_ua['CVBUA'],
            C_ua['CVLUA'],
            C_ua['CVPUA'],
            C_ua['CVGUA'],
            C_ua['CVBRUA'],
            C_ua['CVSUA'],
            C_ua['CVRUA'],
            C_ua['CVFUA'],
            C_ua['CVKUA'],
            C_ua['CVMUA']
        )
        venG[i] = F_blood['venG']
    
    return QG_eff, venG


def run_splanchnic_sweep(t_end=24.0, dt=0.1):
    """
    Sweep over SPLANCHNIC_AMP values and create 1×2 figure.
    """
    print("[Splanchnic sweep] Starting...")

    set_pub_defaults()
    fig, axes = plt.subplots(1, 2, figsize=PUB_FIGSIZE_2PANEL, sharex=True)
    apply_standard_2panel_layout(fig, wspace=0.35)

    ax_qg = axes[0]  # Panel A: QG_eff
    ax_venG = axes[1]  # Panel B: venG

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(SPLANCHNIC_AMP_VALUES)))

    for color, amp in zip(colors, SPLANCHNIC_AMP_VALUES):
        params = get_default_parameters()
        params["USE_SPLANCHNIC_PROFILE"] = True
        params["SPLANCHNIC_AMP"] = amp
        params["SPLANCHNIC_T_PEAK"] = 1.0  # legacy, unused by new profile but kept for BC
        # Use sweep-specific hyperemia shape (keeps global defaults untouched)
        params["SPLANCHNIC_ONSET"] = SPLANCHNIC_ONSET_SWEEP
        params["SPLANCHNIC_TAU_RISE"] = SPLANCHNIC_TAU_RISE_SWEEP

        t, Y = run_simulation(params=params, t_end=t_end, dt=dt)

        # Calculate flows
        QG_eff, venG = calculate_flows_from_state(t, Y, params)

        # Apply a smooth "portal mixing" ramp so venG does not jump instantly.
        # This mimics finite transit/mixing before hepatic delivery.
        if VEN_G_SMOOTH_TAU is not None and VEN_G_SMOOTH_TAU > 0:
            ven_smooth_factor = (1.0 - np.exp(-t / VEN_G_SMOOTH_TAU)) ** VEN_G_SMOOTH_GAMMA
            venG_plot = venG * ven_smooth_factor
        else:
            venG_plot = venG

        label = f"AMP = {amp:.1f} , {LEGEND_LABELS.get(amp, '')}"

        # Plot Panel A: Effective Splanchnic Blood Flow
        ax_qg.plot(t, QG_eff, lw=2, color=color, label=label)

        # Plot Panel B: UA Delivery to Liver From Gut
        ax_venG.plot(t, venG_plot, lw=2, color=color, label=label)

    # Configure axes
    total_hours, major_loc, minor_loc = time_axis_locators(t_end * 60.0)
    for ax in axes:
        ax.set_xlim(0, total_hours)
        ax.xaxis.set_major_locator(major_loc)
        ax.xaxis.set_minor_locator(minor_loc)
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(axis="x", direction="out", labelsize=PUB_FONT_SIZES["ticks"] + 1)
        ax.tick_params(axis="y", direction="out", labelsize=PUB_FONT_SIZES["ticks"] + 1)
        ax.legend(fontsize=PUB_FONT_SIZES["legend"], loc="upper right", framealpha=0.92)

    # y-axis limits here
    ax_qg.set_ylim(40, 65)        # left panel
    ax_venG.set_ylim(0, 0.8)    # right panel
    # y-axis tick settings (major + minor ticks)
    ax_qg.yaxis.set_major_locator(plt.MultipleLocator(10))   # major ticks every 10
    ax_qg.yaxis.set_minor_locator(plt.MultipleLocator(5))    # minor ticks every 5

    ax_venG.yaxis.set_major_locator(plt.MultipleLocator(0.2))  # major ticks every 0.2
    ax_venG.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # minor ticks every 0.1

    # y-axis tick labels here
    # Panel titles: pass x_pos_A and x_pos_B from config to shift left
    set_panel_title(ax_qg, "A", panel="A", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_A=SUBPLOT_TITLE_X_A)
    set_panel_title(ax_venG, "B", panel="B", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_B=SUBPLOT_TITLE_X_B)

    # Axis labels
    ax_qg.set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_venG.set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_qg.set_ylabel("Mesenteric Blood Flow to Gut(L/h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_venG.set_ylabel("UA Delivery to Liver From Gut (µmol/h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')

    # Save figure
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "splanchnic_amp_sweep.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[Splanchnic sweep] Saved to {output_path}")
    print("[Splanchnic sweep] Done.")


if __name__ == "__main__":
    run_splanchnic_sweep()
