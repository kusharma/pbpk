# simulate.py

import argparse
import time
import numpy as np

from pbpk import (
        get_default_parameters,
        UA_COMPARTMENTS,
        UAGluc_COMPARTMENTS,
        run_simulation)
from src.config import AMP_VALUES, AMP_LABELS

AMP_VALUES = [0.0, 0.5, 1.0, 1.5]
AMP_LABELS = {amp: f"AMP={amp:.1f}" for amp in AMP_VALUES}


def _build_results_dataframe(t, Y):
    cols = UA_COMPARTMENTS + UAGluc_COMPARTMENTS
    df = pd.DataFrame(Y, columns=cols)
    df.insert(0, "Time (h)", t)
    return df


def _calculate_plasma_profiles(Y, params):
    idx_abua = UA_COMPARTMENTS.index("ABUA")
    idx_abug = UA_COMPARTMENTS.__len__() + UAGluc_COMPARTMENTS.index("ABUAGluc")

    abua = Y[:, idx_abua]
    abug = Y[:, idx_abug]

    ua_ng_ml = (abua / params["VB"]) * params["MWUA"] * params["FUUA"]
    uag_ng_ml = (abug / params["VB"]) * params["MWUAGluc"] * params["FUUAGluc"]
    return ua_ng_ml, uag_ng_ml


def _print_summary(t, Y, ua_ng_ml, uag_ng_ml, elapsed, params, df=None):
    ua_cmax = ua_ng_ml.max()
    ua_tmax = t[ua_ng_ml.argmax()]
    uag_cmax = uag_ng_ml.max()
    uag_tmax = t[uag_ng_ml.argmax()]

    print("\n--- PBPK Simulation Summary (Single 500 mg Dose) ---")
    print(f"Total Simulation Time: {t[-1]:.0f} hours")
    print(f"Initial Dose (umol): {params['AODOSE']:.2f}")
    print("-" * 50)
    print("UA (Urolithin A) Free Plasma Concentration:")
    print(f"  Cmax: {ua_cmax:.2f} ng/mL (at Tmax: {ua_tmax:.2f} h)")
    print("UAGluc (Urolithin A Glucuronide) Free Plasma Concentration:")
    print(f"  Cmax: {uag_cmax:.2f} ng/mL (at Tmax: {uag_tmax:.2f} h)")
    print("-" * 50)
    print(f"ODE Solver wall time: {elapsed:.2f} seconds")

    if df is not None:
        df = df.copy()
        df["UA_Cblood_ng_mL"] = ua_ng_ml
        df["UAGluc_Cblood_ng_mL"] = uag_ng_ml
        cols = [
            "Time (h)", "ASIUA", "ASIUAinsol", "ABUA",
            "ABUAGluc", "UA_Cblood_ng_mL", "UAGluc_Cblood_ng_mL",
        ]
        print("\nFirst 5 rows of the results DataFrame:")
        print(df[cols].head())
    else:  # pragma: no cover
        print("\n[pandas unavailable] Showing first 5 rows (numpy):")
        for i in range(min(5, len(t))):
            asiua = Y[i, UA_COMPARTMENTS.index("ASIUA")]
            asiua_insol = Y[i, UA_COMPARTMENTS.index("ASIUAinsol")]
            abua = Y[i, UA_COMPARTMENTS.index("ABUA")]
            abug = Y[i, len(UA_COMPARTMENTS) + UAGluc_COMPARTMENTS.index("ABUAGluc")]
            print(
                f"t={t[i]:5.2f} h, ASIUA={asiua:.2f}, ASIUAinsol={asiua_insol:.2f}, "
                f"ABUA={abua:.2f}, ABUAGluc={abug:.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PBPK simulation with different dissolution modes"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mm", "mech", "mech_dh"],
        default="mech_dh",
        help="Dissolution mode: 'mm' (Michaelis-Menten), 'mech' (Mechanistic k_diss), "
             "'mech_dh' (Mechanistic with D/h diffusion layer, default)",
    )
    parser.add_argument(
        "--splanchnic-amp",
        type=float,
        choices=AMP_VALUES,
        default=AMP_VALUES[3], # 1
        help=(
            "Postprandial hyperemia amplitude (choices match the shared sweep config; "
            "labels = " + ", ".join(f"{amp} ({AMP_LABELS.get(amp, 'UNK')})"
                                    for amp in AMP_VALUES)
        ),
    )
    args = parser.parse_args()

    params = get_default_parameters()
    params["USE_SPLANCHNIC_PROFILE"] = args.splanchnic_amp > 0.0
    params["SPLANCHNIC_AMP"] = args.splanchnic_amp

    # Configure dissolution mode based on argument
    if args.mode == "mm":
        # Legacy MM mode (default)
        params["USE_MECH_DISSOLUTION"] = False
        params["USE_DIFFUSION_LAYER"] = False
        params["D_ua"] = None
        params["h_diff"] = None
        print("Using MM (Michaelis-Menten) dissolution mode")
    elif args.mode == "mech":
        # Mechanistic dissolution with fallback k_diss
        params["USE_MECH_DISSOLUTION"] = True
        params["USE_DIFFUSION_LAYER"] = False
        params["D_ua"] = None
        params["h_diff"] = None
        params["gamma_prec"] = 1.0
        print("Using Mech (Mechanistic k_diss) dissolution mode")
    elif args.mode == "mech_dh":
        # Mechanistic dissolution with D/h diffusion layer
        params["USE_MECH_DISSOLUTION"] = True
        params["USE_DIFFUSION_LAYER"] = True
        params["D_ua"] = 1e-6
        params["h_diff"] = 0.05
        params["gamma_prec"] = 2.0
        k_eff = (params["D_ua"] / params["h_diff"]) * 3600.0
        print(f"Using Mech (D/h) dissolution mode: D_ua={params['D_ua']}, "
              f"h_diff={params['h_diff']}, k_diss_eff={k_eff:.4f} 1/h")

    t_end = 24.0
    dt = t_end / 99  # ~100 evaluation points

    start = time.time()
    t, Y = run_simulation(params=params, t_end=t_end, dt=dt)
    elapsed = time.time() - start

    df = _build_results_dataframe(t, Y)
    ua_ng_ml, uag_ng_ml = _calculate_plasma_profiles(Y, params)
    _print_summary(t, Y, ua_ng_ml, uag_ng_ml, elapsed, params, df=df)
