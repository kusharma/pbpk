"""
k_ehr_sweep.py

Parameter sweep for bile emptying rate constant (k_EHR) in enterohepatic recirculation.

Sweeps over k_EHR values and plots in a 1×2 layout:
    - entRC(t) — bile emptying flux
    - ABILUA(t) — bile pool amount
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
    apply_standard_2panel_layout,
    SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B
)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.15, -0.15

# Sweep values
K_EHR_VALUES = [0.2, 0.5, 1.0, 2.0]  # h^-1 as before

# Index for ABILUA
ABILUA_IDX = UA_COMPARTMENTS.index("ABILUA")


def calculate_entRC_from_state(t, Y, params):
    """
    Compute bile emptying flux entRC(t).
    """
    n_points = len(t)
    entRC = np.zeros(n_points)

    if "k_EHR" not in params:
        params["k_EHR"] = 1.0

    for i in range(n_points):
        y = Y[i, :]
        n_ua = len(UA_COMPARTMENTS)
        A_ua = y[:n_ua]

        params['ABUA'] = A_ua[UA_COMPARTMENTS.index("ABUA")]

        C_ua = calculate_ua_concentrations(params, A_ua)

        F_met = flows.calculate_metabolism_flows(
            params,
            C_ua['CVLUA'],
            C_ua['CVGUA'],
            A_ua[ABILUA_IDX],
            C_ua['CVBUA']
        )
        entRC[i] = F_met['entRC']

    return entRC



def run_k_ehr_sweep(t_end=48.0, dt=0.1):
    """
    Sweep k_EHR and plot in 1x2 layout.
    """
    print("[k_EHR sweep] Starting...")
    set_pub_defaults()
    # Increase figure height for more space below xlabels
    fig, axes = plt.subplots(1, 2, figsize=(PUB_FIGSIZE_2PANEL[0], PUB_FIGSIZE_2PANEL[1] + 0.55), sharex=True)
    apply_standard_2panel_layout(fig, wspace=0.3, left=0.1, right=0.95, bottom=0.15, top=0.9)  # more space at top/bottom
    ax_entRC = axes[0]
    ax_ABILUA = axes[1]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(K_EHR_VALUES)))

    for color, k_ehr in zip(colors, K_EHR_VALUES):
        params = get_default_parameters()
        params["k_EHR"] = k_ehr

        t, Y = run_simulation(params=params, t_end=t_end, dt=dt)

        ABILUA = Y[:, ABILUA_IDX]
        entRC = calculate_entRC_from_state(t, Y, params)

        ax_entRC.plot(t, entRC, lw=2, color=color, label=f"k_EHR = {k_ehr:.1f}/h")
        ax_ABILUA.plot(t, ABILUA, lw=2, color=color, label=f"k_EHR = {k_ehr:.1f}/h")

        print(f"  k_EHR={k_ehr:.1f}: max entRC={entRC.max():.3f} at {t[entRC.argmax()]:.1f} h")

    # Time-axis locators
    total_hours, major_loc, minor_loc = time_axis_locators(t_end * 60.0)

    for ax in axes:

        # X-axis range
        ax.set_xlim(0, total_hours)

        # ------------------------------------------------------------
        # >>> X-AXIS TICK CUSTOMIZATION (CHANGE HERE) <<<
        # If you want custom ticks, use MultipleLocator and comment out default locators.
        # ------------------------------------------------------------
        from matplotlib.ticker import MultipleLocator

        # Example: Major ticks every 6h, minor every 1h:
        ax.xaxis.set_major_locator(MultipleLocator(6))
        ax.xaxis.set_minor_locator(MultipleLocator(3))

        # Comment these out if using custom ticks above:
        # ax.xaxis.set_major_locator(major_loc)
        # ax.xaxis.set_minor_locator(minor_loc)
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # >>> Y-AXIS TICK CUSTOMIZATION (CHANGE HERE) <<<
        # Example: set y ticks every 1 µmol/h or 2 µmol depending on subplot
        # ------------------------------------------------------------
        # ax.yaxis.set_major_locator(MultipleLocator(1.0))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        # ------------------------------------------------------------

        ax.grid(True, alpha=0.2, which="both")

        # ------------------------------------------------------------
        # >>> TICK FONT SIZES (CHANGE HERE) <<<
        # ------------------------------------------------------------
        ax.tick_params(
            axis="x",
            direction="out",
            labelsize=PUB_FONT_SIZES["ticks"] + 1,
        )
        ax.tick_params(
            axis="y",
            direction="out",
            labelsize=PUB_FONT_SIZES["ticks"] + 1,
        )
        # ------------------------------------------------------------

        ax.legend(
            fontsize=PUB_FONT_SIZES["legend"],
            loc="upper right",
            framealpha=0.92
        )

    # Panel titles: shift titles slightly further up for visual separation
    set_panel_title(ax_entRC, "A", panel="A", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_A=SUBPLOT_TITLE_X_A, y=1.06)
    set_panel_title(ax_ABILUA, "B", panel="B", fontsize=PUB_FONT_SIZES["title"] + 1, x_pos_B=SUBPLOT_TITLE_X_B, y=1.06)

    # Axis labels
    ax_entRC.set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_ABILUA.set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')

    ax_entRC.set_ylabel("Bile Emptying Rate (µmol/h)",
                        fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    ax_ABILUA.set_ylabel("Biliary Urolithin A Amount (µmol)",
                         fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')

    # Save
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "k_ehr_sweep.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[k_EHR sweep] Saved to {output_path}")
    print("[k_EHR sweep] Done.")


if __name__ == "__main__":
    run_k_ehr_sweep()
