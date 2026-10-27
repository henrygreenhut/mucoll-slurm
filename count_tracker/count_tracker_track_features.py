#!/usr/bin/env python3
"""Paper-aligned fitted-track observables for the SIM-versus-COUNT PFN.

The store keeps physical observables and the original signed curvature. The
network sees pT, eta, phi, d0, and z0; it does not use curvature or charge.
"""

import numpy as np


FIELD_T = 5.0
# EDM4hep omega is signed curvature in 1/mm. pT [GeV] = 0.3 B[T] R[m].
CURVATURE_TO_PT = 0.3 * FIELD_T / 1000.0

RAW_FEATURES = ("pt", "eta", "phi", "d0", "z0", "omega")
RAW_FEATURES_WITH_CHI2_NDF = RAW_FEATURES + ("chi2_ndf",)
SUPPORTED_RAW_FEATURES = (RAW_FEATURES, RAW_FEATURES_WITH_CHI2_NDF)
RAW = {name: index for index, name in enumerate(RAW_FEATURES)}

FEATURES = ("log_pt", "eta", "sin_phi", "cos_phi", "d0", "z0")
FEATURES_WITH_CHI2_NDF = FEATURES + ("chi2_ndf",)
FEATURE_DEFINITIONS = {
    "log_pt": "ln(pT / GeV), pT [GeV] = 0.0015 / |omega [1/mm]| for B = 5 T",
    "eta": "asinh(tanLambda) at the AtIP track state",
    "sin_phi": "sin(phi) at the AtIP track state",
    "cos_phi": "cos(phi) at the AtIP track state",
    "d0": "transverse impact parameter D0 [mm] at AtIP",
    "z0": "longitudinal impact parameter Z0 [mm] at AtIP",
    "chi2_ndf": "track-fit chi-squared divided by the number of degrees of freedom",
}


def track_row(phi, omega, tan_lambda, d0, z0):
    """Physical observables and signed curvature from one valid AtIP state."""
    values = np.asarray([phi, omega, tan_lambda, d0, z0], dtype=np.float64)
    if not np.all(np.isfinite(values)) or omega == 0:
        raise ValueError("AtIP track parameters must be finite with nonzero omega")
    pt = CURVATURE_TO_PT / abs(omega)
    if not np.isfinite(pt) or pt <= 0:
        raise ValueError("AtIP track curvature does not yield finite positive pT")
    return [float(pt), float(np.arcsinh(tan_lambda)), float(phi),
            float(d0), float(z0), float(omega)]


def pfn_feature_names(raw_features=RAW_FEATURES):
    """Network feature names for one supported raw-store schema."""
    raw_features = tuple(raw_features)
    if raw_features == RAW_FEATURES:
        return FEATURES
    if raw_features == RAW_FEATURES_WITH_CHI2_NDF:
        return FEATURES_WITH_CHI2_NDF
    raise ValueError(f"unsupported raw track feature schema: {raw_features}")


def pfn_features(raw, n_tracks, raw_features=RAW_FEATURES):
    """Transform fitted tracks; n_tracks, rather than pT, identifies padding."""
    raw = np.asarray(raw, dtype=np.float32)
    n_tracks = np.asarray(n_tracks, dtype=np.int64)
    raw_features = tuple(raw_features)
    network_features = pfn_feature_names(raw_features)
    if raw.ndim != 3 or raw.shape[2] != len(raw_features):
        raise ValueError("raw tracks have an unexpected feature shape")
    if n_tracks.shape != (len(raw),) or np.any(n_tracks < 0) or np.any(n_tracks > raw.shape[1]):
        raise ValueError("n_tracks must give the valid prefix length of each event")
    mask = np.arange(raw.shape[1])[None, :] < n_tracks[:, None]
    raw_index = {name: index for index, name in enumerate(raw_features)}
    pt = raw[:, :, raw_index["pt"]]
    if np.any(~np.isfinite(raw[mask])) or np.any(pt[mask] <= 0):
        raise ValueError("valid fitted tracks must have finite features and positive pT")
    out = np.zeros((len(raw), raw.shape[1], len(network_features)), dtype=np.float32)
    phi = raw[:, :, raw_index["phi"]]
    log_pt = np.zeros_like(pt)
    np.log(pt, out=log_pt, where=mask)
    values = (
        log_pt,
        raw[:, :, raw_index["eta"]],
        np.sin(phi),
        np.cos(phi),
        raw[:, :, raw_index["d0"]],
        raw[:, :, raw_index["z0"]],
    )
    for index, value in enumerate(values):
        out[:, :, index][mask] = value[mask]
    if "chi2_ndf" in raw_index:
        reduced_chi2 = raw[:, :, raw_index["chi2_ndf"]]
        if np.any(reduced_chi2[mask] < 0):
            raise ValueError("valid fitted tracks must have nonnegative chi2/ndf")
        out[:, :, len(FEATURES)][mask] = reduced_chi2[mask]
    return out
