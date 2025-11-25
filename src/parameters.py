# parameters.py
import numpy as np

# -------------------------------------------------------------------
# Full PBPK parameter builder
# -------------------------------------------------------------------
def get_default_parameters():
    params = {}

    # ===============================================================
    # Mechanistic dissolution switches & constants
    # ===============================================================
    params["USE_MECH_DISSOLUTION"] = False
    params["USE_DIFFUSION_LAYER"] = False

    params["k_diss"] = 5.0           # fallback first-order dissolution [1/h]
    params["k_prec"] = 1.0           # fallback precipitation [1/h]
    params["gamma_prec"] = 1.0       # supersaturation exponent
    params["S0"] = 200.0             # baseline surface area [cm^2]
    params["A0_insol"] = None
    params["D_ua"] = None
    params["h_diff"] = None

    # Stomach emptying (kept for completeness)
    params["k_ge_liq"] = 2.0
    params["k_ge_sol"] = 0.7
    params["k_diss_st"] = 1.0

    # ===============================================================
    # Legacy PBPK constants (inspired by the working v1 model)
    # ===============================================================
    params.update(_build_legacy_pbpk_constants())

    # Align newer keys with historical names
    params["parent"] = {"SOL": params["SOLUA"]}
    params["AODOSE"] = params["AODOSEUA"]

    # === Time-varying splanchnic blood flow (optional) ===
    params["USE_SPLANCHNIC_PROFILE"] = False     # OFF by default → backward compatible
    params["SPLANCHNIC_AMP"] = 0.5               # +50% peak flow increase
    params["SPLANCHNIC_T_PEAK"] = 1.0            # peak hyperemia at 1 hour

    # ---------------------------------------------------------------
    # UA / UAGluc volume dictionaries for concentration helper
    # ---------------------------------------------------------------
    params["ua_volumes"] = {
        "VASIUA": params["VSI"],
        "VALIUA": params["VLI"],
        "VUAex":  0.01,  # bookkeeping only
        "VALUA":  params["VL"],
        "VABUA":  params["VB"],
        "VAPUA":  params["VP"],
        "VAGUA":  params["VG"],
        "VABRUA": params["VBR"],
        "VASUA":  params["VS"],
        "VARUA":  params["VR"],
        "VAFUA":  params["VF"],
        "VAKUA":  params["VK"],
        "VAMUA":  params["VM"],
        "VASIUAinsol": params["VSI"],
        "VABILUA": 0.10,
    }

    params["ua_index"] = {
        "ASIUA": 0, "ALIUA": 1, "UAex": 2, "ALUA": 3,
        "ABUA": 4, "APUA": 5, "AGUA": 6, "ABRUA": 7,
        "ASUA": 8, "ARUA": 9, "AFUA":10, "AKUA":11,
        "AMUA":12, "ASIUAinsol":13, "ABILUA":14
    }

    params["uagluc_volumes"] = {
        "VALUAGluc": params["VL"],
        "VABUAGluc": params["VB"],
        "VAPUAGluc": params["VP"],
        "VAGUAGluc": params["VG"],
        "VASUAGluc": params["VS"],
        "VARUAGluc": params["VR"],
        "VAFUAGluc": params["VF"],
        "VAKUAGluc": params["VK"],
        "VAMUAGluc": params["VM"],
        "VAUCUAUr": 0.01,
    }

    params["uagluc_index"] = {
        "ALUAGluc":0, "ABUAGluc":1, "APUAGluc":2, "AGUAGluc":3,
        "ASUAGluc":4, "ARUAGluc":5, "AFUAGluc":6, "AKUAGluc":7,
        "AMUAGluc":8, "AUCUAUr":9
    }

    # ---------------------------------------------------------------
    # Attach PBPK flow implementations
    # ---------------------------------------------------------------
    from flows import (
        blood_flow_fn, gi_flow_fn,
        metabolism_flow_fn, uagluc_flow_fn
    )

    params["blood_flow_fn"] = blood_flow_fn
    params["gi_flow_fn"] = gi_flow_fn
    params["metabolism_flow_fn"] = metabolism_flow_fn
    params["uagluc_flow_fn"] = uagluc_flow_fn

    return params


def _build_legacy_pbpk_constants():
    """
    Build the physiological / kinetic constants from the validated
    PBPK implementation (mnfr4480 supplemental tables).
    """
    data = {}

    BW = 83.3
    BMI = 29.76
    height = np.sqrt(BW / BMI)
    BSA = np.sqrt((height * 100 * BW) / 3600)

    data.update({
        "BW": BW,
        "BMI": BMI,
        "height": height,
        "BSA": BSA,
    })

    # Tissue fractions → volumes
    frac = {
        "VGc": 0.0114, "VLc": 0.032, "VPc": 0.0154, "VRc": 0.0232,
        "VSc": 0.202,  "VFc": 0.284, "VMc": 0.372,  "VKc": 0.0044,
        "VBRc":0.0224, "VBc": 0.079
    }
    data.update({
        "VG": frac["VGc"] * BW,
        "VL": frac["VLc"] * BW,
        "VP": frac["VPc"] * BW,
        "VR": frac["VRc"] * BW,
        "VS": frac["VSc"] * BW,
        "VF": frac["VFc"] * BW,
        "VM": frac["VMc"] * BW,
        "VK": frac["VKc"] * BW,
        "VBR": frac["VBRc"] * BW,
        "VB": frac["VBc"] * BW,
        "VSI": 9.0,
        "VLI": 7.5,
        "VST": 0.78,
    })

    # Cardiac output and blood flows
    QC = 366.0
    data.update({
        "QC": QC,
        "QG": 0.1155 * QC,
        "QP": 0.0990 * QC,
        "QLA": 0.069 * QC,
        "QL": 0.272 * QC,
        "QR": 1.043 * QC,
        "QS": 0.091 * QC,
        "QF": 0.053 * QC,
        "QM": 0.181 * QC,
        "QK": 0.217 * QC,
        "QBR":0.128 * QC,
    })

    # GFR (L/h)
    GFRc = 125.0
    GFR = GFRc * (BSA / 1.73)
    data["GFR_Lh"] = GFR * 60 / 1000

    # Partition coefficients UA
    data.update({
        "PGUA": 1.68, "PPUA": 1.46, "PLUA": 1.44, "PRUA": 0.65,
        "PSUA": 1.81, "PFUA": 1.40, "PMUA": 1.05, "PKUA": 1.06,
        "PBRUA": 2.06,
    })

    # Partition coefficients UAGluc
    data.update({
        "PGUAGluc": 0.51, "PPUAGluc": 0.52, "PLUAGluc": 0.53,
        "PRUAGluc": 0.55, "PSUAGluc": 0.45, "PFUAGluc": 0.12,
        "PMUAGluc": 0.53, "PKUAGluc": 0.55,
    })

    # Fraction unbound
    data["FUUA"] = 0.198
    data["FUUAGluc"] = 0.307

    # Solubility / dissolution
    data["SOLUA"] = 13.0
    data["VmaxUAsol"] = 22.01
    data["KmUAsol"] = 205.7

    # GI transfer
    LogPappUA = -4.26
    PeffUA = (3600 / 10) * (10 ** (0.7524 * LogPappUA - 0.5441))
    data["PeffUA"] = PeffUA
    data["areaSI"] = 72.0
    data["areaLI"] = 47.0
    data["KaUA"] = PeffUA * data["areaSI"]
    data["KbUA"] = PeffUA * data["areaLI"]
    data["Ksi"] = 1 / 4.27
    data["Kli"] = 1 / 24.6

    # Kinetics / scaling
    data["VLS9"] = 107.3
    data["L"] = frac["VLc"] * 1000
    data["VmaxLUAGlucc"] = 1.70
    data["KmLUAGluc"] = 24.5
    data["EHR"] = 0.038
    # Bile emptying rate constant (1/h)
    # Backward compatible default: k_EHR = 1.0 reproduces original behavior (t½ ≈ 0.69 h)
    # Physiologically plausible range: 0.3-2.0 h^-1
    data["k_EHR"] = 1.0
    data["VGS9"] = 35.2
    data["G"] = frac["VGc"] * 1000
    data["VmaxGUAGlucc"] = 3.6
    data["KmGUAGluc"] = 14.4

    # Dose & MW
    MWUA = 228.203
    MWUAGluc = 404.3
    ODOSEUA = 500.0
    AODOSEUA = ODOSEUA * 1000 / MWUA

    data.update({
        "MWUA": MWUA,
        "MWUAGluc": MWUAGluc,
        "ODOSEUA": ODOSEUA,
        "AODOSEUA": AODOSEUA,
    })

    return data


def build_default_params():
    """Compatibility alias used by older scripts."""
    return get_default_parameters()