"""
Shared sweep parameters used across the various sweep scripts.

This single configuration module centralizes the AMP presets (fasted, light meal, etc.)
and any reusable hyperemia/kinectic tuning knobs so we avoid duplicating the same magic
numbers in multiple scripts.
"""

from typing import List, Dict


AMP_MEAL_PRESETS: List[Dict[str, object]] = [
    {
        "amp": 0.0,
        "label": "No Meal",
        "notes": "Resting baseline; no postprandial hyperemia",
    },
    {
        "amp": 0.5,
        "label": "Light Meal",
        "notes": "Small carbohydrate load with mild flow increase",
    },
    {
        "amp": 1.0,
        "label": "Normal Meal",
        "notes": "Typical mixed meal with moderate postprandial hyperemia",
    },
    {
        "amp": 1.5,
        "label": "Heavy-Fat Meal",
        "notes": "Large/high-fat meal with strong, prolonged hyperemia",
    },
]


AMP_VALUES = [entry["amp"] for entry in AMP_MEAL_PRESETS]
AMP_LABELS = {entry["amp"]: entry["label"] for entry in AMP_MEAL_PRESETS}


# Shared hyperemia shaping knobs (used in sweeps that want to slow the rise)
HYPEREMIA_SHAPE = {
    "onset": 0.5,      # hours until hyperemia starts
    "tau_rise": 2.0,   # hours for the flow ramp to build
    "tau_decay": 2.5,  # hours for the decay phase after the peak
    "ramp_gamma": 2.0, # >1 flattens the early rise (shape parameter)
}

K_EHR_PRESETS = [
    {"k_ehr": 0.2, "label": "Very slow/fasted", "notes": "Fasting gallbladder; rare bile pulses"},
    {"k_ehr": 0.5, "label": "Slow/fasted",      "notes": "Fasting or only mild prandial stimulation"},
    {"k_ehr": 1.0, "label": "Normal/postprandial", "notes": "Typical post-meal emptying"},
    {"k_ehr": 2.0, "label": "Rapid/stimulated", "notes": "Maximally stimulated (CCK, fatty meal, or drugs)"},
]


