# model.py

import numpy as np
from flows import (
    calculate_ua_concentrations, calculate_uagluc_concentrations,
    calculate_blood_flows, calculate_gi_flows, calculate_metabolism_flows,
    calculate_uagluc_flows, compute_dissolution_flows
)

UA_COMPARTMENTS = [
    'ASIUA', 'ALIUA', 'UAex', 'ALUA', 'ABUA', 'APUA', 'AGUA', 'ABRUA',
    'ASUA', 'ARUA', 'AFUA', 'AKUA', 'AMUA', 'ASIUAinsol', 'ABILUA'
]

UAGluc_COMPARTMENTS = [
    'ALUAGluc','ABUAGluc','APUAGluc','AGUAGluc','ASUAGluc',
    'ARUAGluc','AFUAGluc','AKUAGluc','AMUAGluc','AUCUAUr'
]

def pbpk_ode_system(t, y, params):

    y[y < 0] = 0

    # Provide current time to flows & dissolution (maintains backward compatibility)
    params["_time"] = t

    n_ua = len(UA_COMPARTMENTS)
    A_ua = y[:n_ua]
    A_uag = y[n_ua:]

    params['ABUA'] = A_ua[4]
    params['ABUAGluc'] = A_uag[1]

    C_ua = calculate_ua_concentrations(params, A_ua)
    C_uag = calculate_uagluc_concentrations(params, A_uag)

    F_blood = calculate_blood_flows(params,
        C_ua['CVBUA'], C_ua['CVLUA'], C_ua['CVPUA'], C_ua['CVGUA'],
        C_ua['CVBRUA'], C_ua['CVSUA'], C_ua['CVRUA'], C_ua['CVFUA'],
        C_ua['CVKUA'], C_ua['CVMUA']
    )

    F_gi = calculate_gi_flows(params,
        A_ua[0], C_ua['CSIUA'], A_ua[13], C_ua['CSIUAinsol'],
        A_ua[1], C_ua['CLIUA']
    )

    F_met = calculate_metabolism_flows(params,
        C_ua['CVLUA'], C_ua['CVGUA'], A_ua[14], C_ua['CVBUA']
    )

    F_uag = calculate_uagluc_flows(params,
        C_uag['CVBUAGluc'], C_uag['CVLUAGluc'], C_uag['CVPUAGluc'],
        C_uag['CVGUAGluc'], C_uag['CVSUAGluc'], C_uag['CVRUAGluc'],
        C_uag['CVFUAGluc'], C_uag['CVKUAGluc'], C_uag['CVMUAGluc']
    )

    # ------------------------------------------------------------
    # NEW: unified dissolution/precipitation
    # ------------------------------------------------------------
    F_diss = compute_dissolution_flows(
        params,
        A_insol = A_ua[13],
        A_sol   = A_ua[0]
    )

    dA = np.zeros(n_ua)

    dA[0] = -F_gi['transSI'] - F_gi['transSitoG'] + F_diss["solSI"] - F_diss["precSI"]
    dA[1] = F_gi['transSI'] - F_gi['transLI'] - F_gi['transLitoL']
    dA[2] = F_gi['transLI'] + F_gi['transtSi2']
    dA[3] = F_blood['artL'] + F_blood['venP'] + F_blood['venG'] + F_gi['transLitoL'] - F_blood['venL'] - F_met['FLUAGluc']

    dA[4] = (F_blood['venL'] + F_blood['venBR'] + F_blood['venS'] +
             F_blood['venR'] + F_blood['venF'] + F_blood['venK'] +
             F_blood['venM']) - (
             F_blood['artL'] + F_blood['artP'] + F_blood['artG'] +
             F_blood['artS'] + F_blood['artBR'] + F_blood['artR'] +
             F_blood['artF'] + F_blood['artK'] + F_blood['artM'])

    dA[5] = F_blood['artP'] - F_blood['venP']
    dA[6] = F_gi['transSitoG'] - F_blood['venG'] + F_blood['artG'] - F_met['FGUAGluc']
    dA[7] = F_blood['artBR'] - F_blood['venBR']
    dA[8] = F_blood['artS'] - F_blood['venS']
    dA[9] = F_blood['artR'] - F_blood['venR']
    dA[10] = F_blood['artF'] - F_blood['venF']
    dA[11] = F_blood['artK'] - F_blood['venK'] - F_met['urineUA']
    dA[12] = F_blood['artM'] - F_blood['venM']

    dA[13] = F_diss["precSI"] - F_diss["solSI"] - F_gi['transtSi2']
    if A_ua[13] < 0:
        dA[13] = 0

    dA[14] = -F_met['entRC'] + F_uag['bilex']

    # -------------------------------------------------------------
    # UAGluc model unchanged
    # -------------------------------------------------------------
    du = np.zeros(len(UAGluc_COMPARTMENTS))
    du[0] = F_met['FLUAGluc'] + F_uag['venG1'] + F_uag['venP1'] + F_uag['artL1'] - F_uag['venL1'] - F_uag['bilex']
    du[1] = (F_uag['venK1'] + F_uag['venL1'] + F_uag['venS1'] + F_uag['venF1'] +
             F_uag['venR1'] + F_uag['venM1']) - (
             F_uag['artP1'] + F_uag['artS1'] + F_uag['artL1'] + F_uag['artF1'] +
             F_uag['artG1'] + F_uag['artR1'] + F_uag['artK1'] + F_uag['artM1'])

    du[2] = F_uag['artP1'] - F_uag['venP1']
    du[3] = F_uag['artG1'] - F_uag['venG1'] + F_met['FGUAGluc']
    du[4] = F_uag['artS1'] - F_uag['venS1']
    du[5] = F_uag['artR1'] - F_uag['venR1']
    du[6] = F_uag['artF1'] - F_uag['venF1']
    du[7] = F_uag['artK1'] - F_uag['venK1'] - F_uag['urineUAGluc']
    du[8] = F_uag['artM1'] - F_uag['venM1']
    du[9] = F_uag['urineUAGluc']

    return np.concatenate([dA, du])


def get_initial_conditions(params):
    INIT_ASIUA = params["parent"]["SOL"] * params["VST"]
    INIT_ASIUAinsol = params["AODOSE"] - INIT_ASIUA

    params["A0_insol"] = INIT_ASIUAinsol

    init = {comp: 0.0 for comp in UA_COMPARTMENTS + UAGluc_COMPARTMENTS}
    init["ASIUA"] = INIT_ASIUA
    init["ASIUAinsol"] = INIT_ASIUAinsol

    return np.array([init[c] for c in UA_COMPARTMENTS + UAGluc_COMPARTMENTS])
