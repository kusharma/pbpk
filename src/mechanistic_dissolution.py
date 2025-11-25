"""
Mechanistic dissolution model for Urolithin A in the small intestine.

Implements a Noyes–Whitney–style dissolution rate plus a supersaturation-
based precipitation rate, with a shrinking-core surface area model.

Dissolution (Noyes–Whitney):
    R_diss = k_diss_eff * S(t) * max(SOL - C, 0)

    - If D_ua and h_diff are provided in params, k_diss_eff is derived as:
          k_diss_eff = (D_ua / h_diff) * 3600   [1/h]
      otherwise it falls back to params["k_diss"] [1/h].

    - S(t) is a shrinking-core surface area:
          S(t) = S0 * (A_insol / A0_insol)^(2/3)

Precipitation:
    Uses a supersaturation ratio with exponent gamma:

        C       = A_diss / VSI
        S_ratio = C / SOL

    For S_ratio > 1,
        R_prec = k_prec * (S_ratio^gamma - 1) * VSI

All rates are in µmol/h.
"""

from __future__ import annotations


class MechanisticDissolution:
    def __init__(self, params: dict):
        # Medium / geometry
        self.VSI = params["VSI"]                  # SI volume [L]
        self.SOL = params["parent"]["SOL"]        # solubility [µmol/L]
        self.S0  = params["S0"]                   # baseline surface area [cm²]

        # Initial undissolved load for shrinking-core area scaling
        # (this is set in get_initial_conditions)
        self.A0_insol = params.get("A0_insol", None)

        # Dissolution: D/h or fallback k_diss
        D = params.get("D_ua", None)              # [cm²/s]
        h = params.get("h_diff", None)            # [cm]

        if D is not None and h is not None and h > 0:
            # (D/h)*3600 gives an effective 1/h rate
            self.k_diss = (D / h) * 3600.0        # [1/h]
        else:
            self.k_diss = params["k_diss"]        # [1/h] lumped

        # Precipitation parameters
        self.k_prec = params["k_prec"]            # [1/h]
        self.gamma_prec = params.get("gamma_prec", 1.0)

    # ------------------------------------------------------------------
    # Helper: shrinking-core surface area
    # ------------------------------------------------------------------
    def surface_area(self, A_insol: float) -> float:
        """
        Shrinking-core surface area S(t).

        If A0_insol is missing or numerically tiny, we simply use S0.
        """
        if self.A0_insol is None or self.A0_insol <= 1e-9:
            return self.S0
        if A_insol <= 0.0:
            return 0.0
        return self.S0 * (A_insol / self.A0_insol) ** (2.0 / 3.0)

    # ------------------------------------------------------------------
    # Dissolution rate
    # ------------------------------------------------------------------
    def dissolution_rate(self, A_insol: float, A_diss: float) -> float:
        """
        Dissolution rate in µmol/h from solid (undissolved) to dissolved.

        Uses Noyes–Whitney-like kinetics:
            R_diss = k_diss * S(t) * max(SOL - C, 0)
        """
        if A_insol <= 0.0:
            return 0.0

        C = A_diss / self.VSI            # [µmol/L]
        S = self.surface_area(A_insol)   # [cm²]

        driving = max(self.SOL - C, 0.0)
        if driving <= 0.0 or S <= 0.0:
            return 0.0

        return self.k_diss * S * driving

    # ------------------------------------------------------------------
    # Precipitation rate
    # ------------------------------------------------------------------
    def precipitation_rate(self, A_diss: float) -> float:
        """
        Precipitation rate in µmol/h from dissolved back to solid.

        Uses supersaturation ratio:
            S_ratio = C / SOL
        with:
            R_prec = k_prec * (S_ratio^gamma - 1)_+ * VSI
        """
        if A_diss <= 0.0:
            return 0.0
        if self.SOL <= 0.0:
            return 0.0

        C = A_diss / self.VSI             # [µmol/L]
        S_ratio = C / self.SOL
        if S_ratio <= 1.0:
            return 0.0

        ss_term = (S_ratio ** self.gamma_prec) - 1.0
        if ss_term <= 0.0:
            return 0.0

        return self.k_prec * ss_term * self.VSI


def dissolution_ode(t, y, params):
    """
    Simple wrapper calling the unified dissolution engine.
    y = [A_insol, A_sol]
    """
    A_insol, A_sol = y
    flows = compute_dissolution_flows(params, A_insol, A_sol)
    dA_insol = -flows["solSI"] + flows["precSI"]
    dA_sol   =  flows["solSI"] - flows["precSI"]
    return [dA_insol, dA_sol]

__all__ = ["dissolution_ode"]

