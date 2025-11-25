# flows.py
import numpy as np

# ---------------------------------------------------------------
# Unified mechanistic dissolution engine
# ---------------------------------------------------------------

def compute_dissolution_flows(params, A_insol, A_sol):
    """
    Unified dissolution / precipitation flow for SI.

    Inputs:
        A_insol : undissolved UA in SI
        A_sol   : dissolved UA in SI

    Returns dict:
        {
            "solSI": dissolution flow (insol → sol) [umol/h],
            "precSI": precipitation flow (sol → insol) [umol/h]
        }
    """

    # --- parameters ---
    k_diss = params["k_diss"]              # fallback
    k_prec = params["k_prec"]
    S0     = params["S0"]

    # concentration in SI
    Vsi = params["VSI"]
    C_sol = A_sol / Vsi                    # µmol/cm³ effectively

    # solubility of UA
    Cs = params["parent"]["SOL"]

    # --- select mechanism ---
    mech = params.get("USE_MECH_DISSOLUTION", False)
    use_Dh = params.get("USE_DIFFUSION_LAYER", False)
    gamma = params.get("gamma_prec", 1.0)

    # OPTIONAL: D/h override
    if use_Dh:
        D = params["D_ua"]
        h = params["h_diff"]
        if (D is not None) and (h is not None):
            k_diss_eff = (D / h) * 3600.0   # convert 1/s → 1/h
        else:
            k_diss_eff = k_diss
    else:
        k_diss_eff = k_diss

    # --- dissolution surface area ---
    # shrinking core model: S = S0 * (A_insol / A0_insol)^(2/3)
    A0 = params.get("A0_insol", None)
    if A0 is not None and A0 > 0:
        ratio = max(A_insol / A0, 1e-12)
        S_dyn = S0 * ratio**(2.0/3.0)
    else:
        S_dyn = S0

    # ==================================================================
    # MODE 1: MM legacy (no mechanistic dissolution)
    # ==================================================================
    if not mech:
        CSIUA = A_sol / Vsi
        CSIUAinsol = A_insol / Vsi

        if (CSIUA < Cs) and (params.get("VmaxUAsol") is not None):
            sol_rate = params["VmaxUAsol"] * CSIUAinsol / (params["KmUAsol"] + CSIUAinsol)
        else:
            sol_rate = 0.0

        return {
            "solSI": sol_rate,
            "precSI": 0.0,
        }

    # ==================================================================
    # MODE 2: mechanistic – Noyes–Whitney + precipitation
    # ==================================================================

    # Dissolution (Noyes–Whitney)
    # dA/dt = k_diss_eff * S * (Cs - C)
    driving = max(Cs - C_sol, 0.0)
    sol_rate = k_diss_eff * S_dyn * driving

    # Precipitation: k_prec * (C/Cs - 1)^gamma * A_sol
    if C_sol > Cs:
        supersat = (C_sol / Cs - 1.0)
        prec_rate = k_prec * (supersat ** gamma) * A_sol
    else:
        prec_rate = 0.0

    return {
        "solSI": sol_rate,
        "precSI": prec_rate
    }

# ---------------------------------------------------------------
# PBPK concentration helpers
# ---------------------------------------------------------------

def calculate_ua_concentrations(params, A):
    idx = params["ua_index"]

    def amt(name):
        return A[idx[name]]

    CSIUA = amt("ASIUA") / params["VSI"]
    CSIUAinsol = amt("ASIUAinsol") / params["VSI"]
    CLIUA = amt("ALIUA") / params["VLI"]

    CGUA = amt("AGUA") / params["VG"]
    CVGUA = CGUA / params["PGUA"]

    CLUA = amt("ALUA") / params["VL"]
    CVLUA = CLUA / params["PLUA"]

    CPUA = amt("APUA") / params["VP"]
    CVPUA = CPUA / params["PPUA"]

    CRUA = amt("ARUA") / params["VR"]
    CVRUA = CRUA / params["PRUA"]

    CSUA = amt("ASUA") / params["VS"]
    CVSUA = CSUA / params["PSUA"]

    CFUA = amt("AFUA") / params["VF"]
    CVFUA = CFUA / params["PFUA"]

    CMUA = amt("AMUA") / params["VM"]
    CVMUA = CMUA / params["PMUA"]

    CKUA = amt("AKUA") / params["VK"]
    CVKUA = CKUA / params["PKUA"]

    CBRUA = amt("ABRUA") / params["VBR"]
    CVBRUA = CBRUA / params["PBRUA"]

    CBUA = amt("ABUA") / params["VB"]
    CVBUA = CBUA * params["FUUA"]

    return {
        "CSIUA": CSIUA,
        "CSIUAinsol": CSIUAinsol,
        "CLIUA": CLIUA,
        "CVBUA": CVBUA,
        "CVLUA": CVLUA,
        "CVPUA": CVPUA,
        "CVGUA": CVGUA,
        "CVBRUA": CVBRUA,
        "CVRUA": CVRUA,
        "CVSUA": CVSUA,
        "CVFUA": CVFUA,
        "CVKUA": CVKUA,
        "CVMUA": CVMUA,
    }


def calculate_uagluc_concentrations(params, A):
    idx = params["uagluc_index"]

    def amt(name):
        return A[idx[name]]

    CGUAGluc = amt("AGUAGluc") / params["VG"]
    CVGUAGluc = CGUAGluc / params["PGUAGluc"]

    CLUAGluc = amt("ALUAGluc") / params["VL"]
    CVLUAGluc = CLUAGluc / params["PLUAGluc"]

    CPUAGluc = amt("APUAGluc") / params["VP"]
    CVPUAGluc = CPUAGluc / params["PPUAGluc"]

    CRUAGluc = amt("ARUAGluc") / params["VR"]
    CVRUAGluc = CRUAGluc / params["PRUAGluc"]

    CSUAGluc = amt("ASUAGluc") / params["VS"]
    CVSUAGluc = CSUAGluc / params["PSUAGluc"]

    CFUAGluc = amt("AFUAGluc") / params["VF"]
    CVFUAGluc = CFUAGluc / params["PFUAGluc"]

    CMUAGluc = amt("AMUAGluc") / params["VM"]
    CVMUAGluc = CMUAGluc / params["PMUAGluc"]

    CKUAGluc = amt("AKUAGluc") / params["VK"]
    CVKUAGluc = CKUAGluc / params["PKUAGluc"]

    CBUAGluc = amt("ABUAGluc") / params["VB"]
    CVBUAGluc = CBUAGluc * params["FUUAGluc"]

    return {
        "CVBUAGluc": CVBUAGluc,
        "CVLUAGluc": CVLUAGluc,
        "CVPUAGluc": CVPUAGluc,
        "CVGUAGluc": CVGUAGluc,
        "CVSUAGluc": CVSUAGluc,
        "CVRUAGluc": CVRUAGluc,
        "CVFUAGluc": CVFUAGluc,
        "CVKUAGluc": CVKUAGluc,
        "CVMUAGluc": CVMUAGluc,
    }


def calculate_blood_flows(params, *args):
    return params["blood_flow_fn"](params, *args)


def calculate_gi_flows(params, *args):
    return params["gi_flow_fn"](params, *args)


def calculate_metabolism_flows(params, *args):
    return params["metabolism_flow_fn"](params, *args)


def calculate_uagluc_flows(params, *args):
    return params["uagluc_flow_fn"](params, *args)


# ---------------------------------------------------------------
# PBPK FLOW IMPLEMENTATIONS
# ---------------------------------------------------------------

def splanchnic_flow_multiplier(params):
    """
    Smooth delayed postprandial hyperemia profile.

    M(t) = 1 + AMP * f(t) where
    f(t) = (1 - exp(-(t - t_onset)/tau_rise)) * exp(-(t - t_onset)/tau_decay)
    and f(t < t_onset) = 0.
    """
    if not params.get("USE_SPLANCHNIC_PROFILE", False):
        return 1.0

    t = params.get("_time", 0.0)
    AMP = params.get("SPLANCHNIC_AMP", 0.5)
    t_onset = params.get("SPLANCHNIC_ONSET", 0.5)
    tau_rise = params.get("SPLANCHNIC_TAU_RISE", 1.0)
    tau_decay = params.get("SPLANCHNIC_TAU_DECAY", 2.5)

    # Optional shape factor to slow onset/rise without changing defaults.
    # If not provided → factor=1.0 → exact previous behaviour.
    shape_factor = params.get("SPLANCHNIC_SHAPE_FACTOR", 1.0)
    if shape_factor <= 0:
        shape_factor = 1.0

    t_onset_eff = t_onset * shape_factor
    tau_rise_eff = tau_rise * shape_factor

    if t < t_onset_eff:
        return 1.0

    x = t - t_onset_eff
    f = (1.0 - np.exp(-x / tau_rise_eff)) * np.exp(-x / tau_decay)

    return 1.0 + AMP * f


def blood_flow_fn(params, CBUA, CVLUA, CVPUA, CVGUA, CVBRUA,
                  CVSUA, CVRUA, CVFUA, CVKUA, CVMUA):
    # Base flows
    QG = params["QG"]
    QP = params["QP"]
    
    # === Apply splanchnic multiplier ===
    M_spl = splanchnic_flow_multiplier(params)
    QG_eff = QG * M_spl
    QP_eff = QP * M_spl
    
    artL = params["QLA"] * CBUA
    artP = QP_eff * CBUA
    artG = QG_eff * CBUA
    artBR = params["QBR"] * CBUA
    artS = params["QS"] * CBUA
    artR = params["QR"] * CBUA
    artF = params["QF"] * CBUA
    artK = params["QK"] * CBUA
    artM = params["QM"] * CBUA

    venL = params["QL"] * CVLUA
    venP = QP_eff * CVPUA
    venG = QG_eff * CVGUA
    venBR = params["QBR"] * CVBRUA
    venS = params["QS"] * CVSUA
    venR = params["QR"] * CVRUA
    venF = params["QF"] * CVFUA
    venK = params["QK"] * CVKUA
    venM = params["QM"] * CVMUA

    return {
        "artL": artL, "artP": artP, "artG": artG, "artBR": artBR,
        "artS": artS, "artR": artR, "artF": artF, "artK": artK, "artM": artM,
        "venL": venL, "venP": venP, "venG": venG, "venBR": venBR,
        "venS": venS, "venR": venR, "venF": venF, "venK": venK, "venM": venM,
    }


def gi_flow_fn(params, ASIUA, CSIUA, ASIUAinsol, CSIUAinsol, ALIUA, CLIUA):
    transSI = ASIUA * params["Ksi"]
    transLI = ALIUA * params["Kli"]
    transSitoG = params["KaUA"] * CSIUA
    transLitoL = params["KbUA"] * CLIUA
    transtSi2 = ASIUAinsol * params["Ksi"]

    return {
        "transSI": transSI,
        "transLI": transLI,
        "transSitoG": transSitoG,
        "transtSi2": transtSi2,
        "transLitoL": transLitoL,
    }


def metabolism_flow_fn(params, CVLUA, CVGUA, ABILUA, CVBUA):
    VmaxLUAGluc = params["VmaxLUAGlucc"] / 1000 * 60 * params["VLS9"] * params["L"] * params["BW"]
    VmaxGUAGluc = params["VmaxGUAGlucc"] / 1000 * 60 * params["VGS9"] * params["G"] * params["BW"]

    FLUAGluc = VmaxLUAGluc * CVLUA / (params["KmLUAGluc"] + CVLUA)
    FGUAGluc = VmaxGUAGluc * CVGUA / (params["KmGUAGluc"] + CVGUA)
    urineUA = params["GFR_Lh"] * CVBUA
    # entRC now explicitly controlled by k_EHR but defaults to same behavior
    entRC = params["k_EHR"] * ABILUA

    # ORIGINAL: keep intrinsic gut glucuronidation independent of splanchnic multiplier
    FGUAGluc_eff = FGUAGluc

    return {
        "FLUAGluc": FLUAGluc,
        "FGUAGluc": FGUAGluc_eff,
        "urineUA": urineUA,
        "entRC": entRC,
    }


def uagluc_flow_fn(params, CVBUAGluc, CVLUAGluc, CVPUAGluc, CVGUAGluc,
                   CVSUAGluc, CVRUAGluc, CVFUAGluc, CVKUAGluc, CVMUAGluc):
    CBUAGluc_arterial = params["ABUAGluc"] / params["VB"] if params["VB"] > 0 else 0.0

    artL1 = params["QLA"] * CBUAGluc_arterial
    artP1 = params["QP"] * CBUAGluc_arterial
    artG1 = params["QG"] * CBUAGluc_arterial
    artS1 = params["QS"] * CBUAGluc_arterial
    artR1 = params["QR"] * CBUAGluc_arterial
    artF1 = params["QF"] * CBUAGluc_arterial
    artK1 = params["QK"] * CBUAGluc_arterial
    artM1 = params["QM"] * CBUAGluc_arterial

    venL1 = (1 - params["EHR"]) * params["QL"] * CVLUAGluc
    venP1 = params["QP"] * CVPUAGluc
    venG1 = params["QG"] * CVGUAGluc
    venS1 = params["QS"] * CVSUAGluc
    venR1 = params["QR"] * CVRUAGluc
    venF1 = params["QF"] * CVFUAGluc
    venK1 = params["QK"] * CVKUAGluc
    venM1 = params["QM"] * CVMUAGluc

    urineUAGluc = params["GFR_Lh"] * CVBUAGluc
    bilex = params["EHR"] * params["QL"] * CVLUAGluc

    return {
        "artL1": artL1, "artP1": artP1, "artG1": artG1, "artS1": artS1,
        "artR1": artR1, "artF1": artF1, "artK1": artK1, "artM1": artM1,
        "venL1": venL1, "venP1": venP1, "venG1": venG1, "venS1": venS1,
        "venR1": venR1, "venF1": venF1, "venK1": venK1, "venM1": venM1,
        "urineUAGluc": urineUAGluc,
        "bilex": bilex,
    }

# ---------------------------------------------------------------
# EXPORT FUNCTIONS FOR PBPK ENGINE
# ---------------------------------------------------------------
__all__ = [
    "compute_dissolution_flows",
    "calculate_ua_concentrations",
    "calculate_uagluc_concentrations",
    "calculate_blood_flows",
    "calculate_gi_flows",
    "calculate_metabolism_flows",
    "calculate_uagluc_flows",
]

