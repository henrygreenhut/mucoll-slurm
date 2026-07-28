#!/usr/bin/env python3
"""Canonical PFO feature transformation for the RECO PFN study."""

import numpy as np


RAW_FEATURES = (
    "pt", "eta", "phi", "energy", "mass", "charge", "pdg", "px", "py", "pz",
)
RAW = {name: i for i, name in enumerate(RAW_FEATURES)}
FEATURES = (
    "log_pt", "eta", "sin_phi", "cos_phi", "log_energy", "charge",
)
FEATURE_DEFINITIONS = {
    "log_pt": "ln(pt / GeV)",
    "eta": "asinh(pz / pt)",
    "sin_phi": "sin(atan2(py, px))",
    "cos_phi": "cos(atan2(py, px))",
    "log_energy": "ln(energy / GeV)",
    "charge": "charge / e",
}


def pfn_features(raw):
    """Transform padded raw PFO arrays; zero-pT entries remain zero padding."""
    mask = raw[:, :, RAW["pt"]] > 0
    out = np.zeros((len(raw), raw.shape[1], len(FEATURES)), dtype=np.float32)
    pt = raw[:, :, RAW["pt"]]
    eta = raw[:, :, RAW["eta"]]
    phi = raw[:, :, RAW["phi"]]
    energy = raw[:, :, RAW["energy"]]
    charge = raw[:, :, RAW["charge"]]

    if np.any(mask & (energy <= 0)):
        raise ValueError("real PFOs must have positive energy for log(E)")
    log_pt = np.zeros_like(pt, dtype=np.float32)
    log_energy = np.zeros_like(energy, dtype=np.float32)
    np.log(pt, out=log_pt, where=mask)
    np.log(energy, out=log_energy, where=mask)

    values = (
        log_pt,
        eta,
        np.sin(phi),
        np.cos(phi),
        log_energy,
        charge,
    )
    for index, value in enumerate(values):
        out[:, :, index][mask] = value[mask]
    return out
