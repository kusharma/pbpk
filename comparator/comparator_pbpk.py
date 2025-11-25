# comparator/comparator_pbpk.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pbpk import get_default_parameters, run_simulation, UA_COMPARTMENTS
from comparator.plot_config import (
    set_pub_defaults,
    apply_standard_2panel_layout,
    set_common_time_axis,
    set_panel_title,
    PUB_FONT_SIZES,
    PUB_FIGSIZE_2PANEL,
    WSPACE_FRAC,
    M_L_IN,
    M_R_IN,
    M_B_IN,
    M_T_IN,
    FIG_W_IN,
    FIG_H_IN,
    SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B
)
SUBPLOT_TITLE_X_A, SUBPLOT_TITLE_X_B = -0.15, -0.15

# DIFFUSION_PARAMS_origA = {
#     "D_ua": 1e-6,
#     "h_diff": .05,
# }

# DIFFUSION_PARAMS_B = {
#     "D_ua": 2e-6,
#     "h_diff": .05,
# }

DIFFUSION_PARAMS = {
    "D_ua": 1e-6,
    "h_diff": .05,
}

def run_pbpk_4way_comparison():
    configs = [
        {
            "title": "No Mech / No Diffusion",
            "legend": "MM",
            "MECH": False,
            "DIFF": False,
            "gamma_prec": 1.0,
        },
        {
            "title": "Mech only",
            "legend": "Mech",
            "MECH": True,
            "DIFF": False,
            "gamma_prec": 1.0,
        },
        {
            "title": "Mech + Diffusion layer",
            "legend": "Mech (D/h)",
            "MECH": True,
            "DIFF": True,
            "gamma_prec": 2.0,
        },
    ]
    set_pub_defaults()
    # Increase figure size and reduce white space (too much space above and below)
    # Increase width more than height, and reduce top/bottom margins
    fig, axes = plt.subplots(1, 2, figsize=(PUB_FIGSIZE_2PANEL[0]*1.15, PUB_FIGSIZE_2PANEL[1]*1.05), sharex=True)
    # Reduce top and bottom margins to minimize white space
    left   = M_L_IN / (FIG_W_IN * 1.15)
    right  = 1.0 - (M_R_IN / (FIG_W_IN * 1.15))
    bottom = (M_B_IN * 0.7) / (FIG_H_IN * 1.05)  # Reduce bottom margin
    top    = 1.0 - ((M_T_IN * 0.6) / (FIG_H_IN * 1.05))  # Reduce top margin
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=WSPACE_FRAC)
    t_max = 24.0
    abua_idx = UA_COMPARTMENTS.index("ABUA")
    asiua_idx = UA_COMPARTMENTS.index("ASIUA")
    asiua_insol_idx = UA_COMPARTMENTS.index("ASIUAinsol")
    colors = ["#1b9e77", "#d95f02", "#e7298a"]
    sim_results = []
    for i, cfg in enumerate(configs):
        params = get_default_parameters()
        params["USE_MECH_DISSOLUTION"] = cfg["MECH"]
        params["USE_DIFFUSION_LAYER"]  = cfg["DIFF"]
        params["gamma_prec"] = cfg["gamma_prec"]
        if cfg["DIFF"]:
            params.update(DIFFUSION_PARAMS)
        else:
            params["D_ua"] = None
            params["h_diff"] = None
        # Debug: report effective k_diss being used
        if params["USE_MECH_DISSOLUTION"]:
            if params.get("D_ua") is not None and params.get("h_diff") is not None and params["h_diff"] > 0:
                k_eff = (params["D_ua"] / params["h_diff"]) * 3600.0
                print(f"[DEBUG] {cfg['title']}: D_ua={params['D_ua']}, h_diff={params['h_diff']}, k_diss_eff={k_eff:.4f} 1/h")
            else:
                k_eff = params["k_diss"]
                print(f"[DEBUG] {cfg['title']}: using fallback k_diss={k_eff:.4f} 1/h (no D/h)")
        else:
            print(f"[DEBUG] {cfg['title']}: legacy MM dissolution (no mechanistic engine)")
        t, Y = run_simulation(params=params, t_end=t_max, dt=0.09)
        sim_results.append((cfg, t, Y))
        axes[0].plot(t, Y[:,abua_idx], lw=2, color=colors[i], label=cfg["legend"])
        # Change to `Y[:, asiua_idx]` here if you want dissolved instead of undissolved
        axes[1].plot(
            t,
            Y[:,asiua_idx],
            lw=2,
            color=colors[i],
            label=cfg["legend"],
        )
    # Apply axes config
    for ax in axes:
        set_common_time_axis(ax)
        ax.tick_params(axis="x", labelsize=PUB_FONT_SIZES["ticks"] + 1)
        ax.tick_params(axis="y", labelsize=PUB_FONT_SIZES["ticks"] + 1)
    # Panel labels (A and B only, no descriptive titles)
    # Can override title positions: x_pos_A=-0.15, x_pos_B=-0.15
    set_panel_title(
        axes[0],
        "A",
        panel="A",
        fontsize=PUB_FONT_SIZES["title"] + 1,
        x_pos_A=SUBPLOT_TITLE_X_A
    )
    set_panel_title(
        axes[1],
        "B",
        panel="B",
        fontsize=PUB_FONT_SIZES["title"] + 1,
        x_pos_B=SUBPLOT_TITLE_X_B
    )
    # X-axis labels (bold, font size +1)
    axes[0].set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    axes[1].set_xlabel("Time (h)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    # Y-axis labels (bold, font size +1)
    axes[0].set_ylabel("UA in Plasma (µmol)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    axes[1].set_ylabel("Dissolved UA in SI (µmol)", fontsize=PUB_FONT_SIZES["axes"] + 1, weight='bold')
    axes[0].legend(fontsize=PUB_FONT_SIZES["legend"], loc='upper right')
    axes[1].legend(fontsize=PUB_FONT_SIZES["legend"], loc='upper right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "ua_plasma_si.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved 1x2 plasma + SI plot to {output_path}")

if __name__ == "__main__":
    run_pbpk_4way_comparison()