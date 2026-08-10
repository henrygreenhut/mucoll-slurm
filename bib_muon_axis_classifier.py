#!/usr/bin/env python3
"""The simplest possible native-vs-synthetic-K=1 classifier.

Rule: count the energetic muons pointing near the horizontal axis in each
pseudo-crossing; whichever crossing has more is called "native".

On GEN truth this reaches AUC ~ 0.93 -- i.e. given one native and one rotated
crossing, the native one has the larger count ~93% of the time. That single
hand-built number beats the trained PFN (0.82), because it keeps exactly the
high-signal objects (energetic muons on the beam-bending axis) that a
mean-pooled network dilutes.

Run where the mother bank lives (Perlmutter):
    module load python
    python bib_muon_axis_classifier.py
"""
import numpy as np
import h5py

BANK = "/pscratch/sd/h/hgreen/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5"
NFILES = 420                    # source cycles pooled into one pseudo-crossing
NCON = 2000                     # crossings per class
EMIN = 5.0                      # GeV: "energetic"
HALFWIN = np.radians(30)        # +-30 deg window around the axis
AXIS = [np.pi]                  # -x lobe: for MU+ the dipole piles the excess
                                # here (AUC 0.93). Use [0.0, np.pi] for the full
                                # horizontal axis (AUC 0.90).

# --- load GEN particles, with their mother and cycle grouping --------------
with h5py.File(BANK) as f:
    g = f["particles"]
    px, py = g["px"][:], g["py"][:]
    E, pdg = g["E"][:], np.abs(g["pdg"][:])
    mother_off = f["mother_offsets"][:]        # particle ranges per mother
    cycle_off = f["cycle_offsets"][:]          # mother ranges per cycle
phi = np.arctan2(py, px)
owner = np.repeat(np.arange(len(mother_off) - 1), np.diff(mother_off))  # mother of each particle
cyc_pbound = mother_off[cycle_off]             # particle boundary per cycle
ncyc = len(cycle_off) - 1

# --- the "rotated" class: one random azimuth per mother (= synthetic K=1) ---
rng = np.random.default_rng(1701)
phi_rot = phi + rng.uniform(0, 2 * np.pi, owner.max() + 1)[owner]

# --- the observable: energetic muons within HALFWIN of the axis ------------
def near_axis(angle):
    dist = np.min([np.abs(((angle - c + np.pi) % (2 * np.pi)) - np.pi) for c in AXIS], axis=0)
    return dist < HALFWIN

tag_nat = (pdg == 13) & (E > EMIN) & near_axis(phi)
tag_rot = (pdg == 13) & (E > EMIN) & near_axis(phi_rot)

# The count is additive over cycles, so a crossing's count is just the sum of
# its cycles' counts -- precompute per cycle, then sample crossings cheaply.
count_nat = np.add.reduceat(tag_nat.astype(np.int64), cyc_pbound[:-1])
count_rot = np.add.reduceat(tag_rot.astype(np.int64), cyc_pbound[:-1])

# --- build crossings and score each with the rule --------------------------
pick = np.random.default_rng(7)
score_nat = np.array([count_nat[pick.choice(ncyc, NFILES, replace=False)].sum()
                      for _ in range(NCON)])
score_rot = np.array([count_rot[pick.choice(ncyc, NFILES, replace=False)].sum()
                      for _ in range(NCON)])

# --- AUC = P(native count > rotated count) + 1/2 P(tie), over all pairs -----
sr = np.sort(score_rot)
less = np.searchsorted(sr, score_nat, side="left")     # rotated crossings with a smaller count
tied = np.searchsorted(sr, score_nat, side="right") - less
auc = (less + 0.5 * tied).sum() / (NCON * NCON)

print(f"mean energetic muons near the axis:  native {score_nat.mean():.1f}   "
      f"rotated {score_rot.mean():.1f}")
print(f"AUC = P(native has more) + 1/2 P(tie) = {auc:.3f}")
print(f"(i.e. pick the crossing with more energetic axis-muons -> right {auc*100:.0f}% of the time)")
