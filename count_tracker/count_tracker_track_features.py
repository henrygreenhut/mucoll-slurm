#!/usr/bin/env python3
"""Kinematic-7 track features for the SIM-vs-COUNT tracker PFN.

Defines the raw per-track values stored in the HDF5 stores and the transform to
the seven model features. No clipping or learned normalization is applied,
matching the RECO PFO study's feature philosophy. A padded track (pt == 0) stays
zero in every feature so the PFN mask drops it before the latent sum.
"""

import numpy as np


CURVATURE_TO_PT = 0.00015  # pt [GeV] = CURVATURE_TO_PT / |omega|, MAIA field

RAW_FEATURES = ("pt", "eta", "phi", "d0", "z0", "charge")
RAW = {name: index for index, name in enumerate(RAW_FEATURES)}

FEATURES = ("log_pt", "eta", "sin_phi", "cos_phi", "d0", "z0", "charge")
FEATURE_DEFINITIONS = {
    "log_pt": "ln(pt / GeV), pt = 0.00015 / |omega|",
    "eta": "asinh(tanLambda)",
    "sin_phi": "sin(phi) at the chosen track state",
    "cos_phi": "cos(phi) at the chosen track state",
    "d0": "transverse impact parameter D0 [mm]",
    "z0": "longitudinal impact parameter Z0 [mm]",
    "charge": "sign(omega)",
}


def track_row(phi, omega, tan_lambda, d0, z0):
    """One raw RAW_FEATURES row from a track state; None if not finite."""
    if not np.all(np.isfinite([phi, omega, tan_lambda, d0, z0])):
        return None
    pt = CURVATURE_TO_PT / abs(omega) if abs(omega) > 1e-12 else 0.0
    return [pt, float(np.arcsinh(tan_lambda)), float(phi),
            float(d0), float(z0), float(np.sign(omega))]


def pfn_features(raw):
    """Transform padded raw track arrays; zero-pt entries remain zero padding."""
    raw = np.asarray(raw, dtype=np.float32)
    mask = raw[:, :, RAW["pt"]] > 0
    out = np.zeros((len(raw), raw.shape[1], len(FEATURES)), dtype=np.float32)
    pt = raw[:, :, RAW["pt"]]
    phi = raw[:, :, RAW["phi"]]
    log_pt = np.zeros_like(pt, dtype=np.float32)
    np.log(pt, out=log_pt, where=mask)
    values = (
        log_pt,
        raw[:, :, RAW["eta"]],
        np.sin(phi),
        np.cos(phi),
        raw[:, :, RAW["d0"]],
        raw[:, :, RAW["z0"]],
        raw[:, :, RAW["charge"]],
    )
    for index, value in enumerate(values):
        out[:, :, index][mask] = value[mask]
    return out
