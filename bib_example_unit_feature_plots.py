"""Plot all 9 PFN input features (logpt, theta, cosphi, sinphi, and the 5
PDG one-hot categories), pooled from real training units, class 0 (norm1,
standard) vs class 1 (norm42, rotated/cloned).

Uses the SAME unit-construction code as bib_example_unit_plots.py (Store /
common_positions / split_indices / UnitSampler, imported from the real
trainer) and the SAME feature builder the network actually sees
(libtest_common.build_features) -- raw (pre-normalization) feature values,
not the z-scored network input, since raw values are what's interpretable
here.

Usage (Perlmutter, on a login node, after module load tensorflow):
    python bib_example_unit_feature_plots.py

Usage (OSCAR, no GPU/container needed -- UnitSampler/Store have no
tensorflow import):
    module load python/3.11.11-5e66
    source ~/envs/mucoll/bin/activate
    python bib_example_unit_feature_plots.py

Store paths/n_files below match whichever OSCAR training job is under
investigation (currently: the n42 reference config, oscar_n42_paper_rawsum,
which showed epoch-0-then-collapse-to-chance -- checking whether the class
separation the network is trained on looks sane before suspecting the
training loop itself).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

import libtest_common as lc
from pfn_libtest_train import UnitSampler

NORM1_STORE = os.path.expanduser("~/mucoll/stores/gen_norm1_reconstructed_MUPLUS.h5")
NORM42_STORE = f"/oscar/scratch/{os.environ.get('USER', '')}/mucoll/stores/gen_norm42_MUPLUS.h5"
N_FILES = 42
CLONE_FACTOR = 42
SPLIT_FRACS = (0.50, 0.25, 0.25)
SPLIT = "train"

N_EXAMPLES = 20
SEED = 11
OUT_PNG = "plots/oscar_n42_example_unit_feature_plots.png"

LABEL0 = "class 0 (norm1, standard)"
LABEL1 = "class 1 (norm42, rotated/cloned)"
COLORS = {LABEL0: "#1f77b4", LABEL1: "#d62728"}

CONTINUOUS_FEATURES = ["logpt", "theta", "cosphi", "sinphi"]
PDG_FEATURES = lc.PDG_ONEHOT  # ["pdg_gamma", "pdg_n", "pdg_e", "pdg_mu", "pdg_other"]


def build_samplers():
    store1 = lc.Store(NORM1_STORE)
    store_b = lc.Store(NORM42_STORE)
    common, pos1, pos_b = lc.common_positions(store1, store_b)
    splits = lc.split_indices(len(common), SPLIT_FRACS)
    idx = splits[SPLIT]

    files_b = N_FILES // CLONE_FACTOR
    sampler0 = UnitSampler(store1, {SPLIT: pos1[idx]}, N_FILES)
    sampler1 = UnitSampler(store_b, {SPLIT: pos_b[idx]}, files_b)
    return store1, store_b, sampler0, sampler1


def example_unit_features(store, sampler, rng, n_examples):
    """Draw n_examples real training units; return pooled (N, 9) raw
    (pre-normalization) feature array, same builder the network uses."""
    feats = []
    for _ in range(n_examples):
        positions = sampler.random_unit(rng, SPLIT)
        raw = store.file_arrays(positions)
        feats.append(lc.build_features(raw))
    return np.concatenate(feats, axis=0)


def main():
    store1, store_b, sampler0, sampler1 = build_samplers()
    rng = np.random.default_rng(SEED)

    feats0 = example_unit_features(store1, sampler0, rng, N_EXAMPLES)
    feats1 = example_unit_features(store_b, sampler1, rng, N_EXAMPLES)
    print(f"{LABEL0}: {N_EXAMPLES} example units -> {len(feats0)} particles pooled")
    print(f"{LABEL1}: {N_EXAMPLES} example units -> {len(feats1)} particles pooled")

    col = {name: i for i, name in enumerate(lc.FEATURE_NAMES)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(CONTINUOUS_FEATURES):
        ax = axes[i]
        c0 = feats0[:, col[name]]
        c1 = feats1[:, col[name]]
        both = np.concatenate([c0, c1])
        bins = np.linspace(both.min(), both.max(), 80)
        ax.hist(c0, bins=bins, histtype="step", density=True, linewidth=1.8,
                color=COLORS[LABEL0], label=LABEL0)
        ax.hist(c1, bins=bins, histtype="step", density=True, linewidth=1.8,
                color=COLORS[LABEL1], label=LABEL1)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"{name} distribution")
        ax.legend(fontsize=8)

    ax = axes[4]
    fractions0 = [feats0[:, col[name]].mean() for name in PDG_FEATURES]
    fractions1 = [feats1[:, col[name]].mean() for name in PDG_FEATURES]
    x = np.arange(len(PDG_FEATURES))
    width = 0.35
    ax.bar(x - width / 2, fractions0, width, color=COLORS[LABEL0], label=LABEL0)
    ax.bar(x + width / 2, fractions1, width, color=COLORS[LABEL1], label=LABEL1)
    ax.set_xticks(x)
    ax.set_xticklabels(PDG_FEATURES, rotation=20)
    ax.set_ylabel("fraction of particles")
    ax.set_title("PDG one-hot composition")
    ax.legend(fontsize=8)

    axes[5].axis("off")
    axes[5].text(
        0.0, 0.9, f"{N_EXAMPLES} example training units/class\n"
                 f"({LABEL0}: {sampler0.files_per_unit} files/unit)\n"
                 f"({LABEL1}: {sampler1.files_per_unit} files/unit)\n"
                 f"pooled: {len(feats0)} vs {len(feats1)} particles",
        transform=axes[5].transAxes, fontsize=10, va="top")

    fig.suptitle("PFN input features: real training units, norm1 vs norm42",
                 fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
