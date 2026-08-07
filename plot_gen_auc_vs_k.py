#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def test_auc(path):
    if not path.exists():
        return None
    with path.open() as handle:
        return float(json.load(handle)["test_auc"])


def variable_summary(label):
    return ROOT / "variable_k_results" / label / "summary.json"


def library_summary(label):
    return ROOT / "pfn_results" / label / "auc_summary.json"


def available(points):
    return [(k, auc) for k, auc in points if auc is not None]


def coordinates(points):
    return [item[0] for item in points], [item[1] for item in points]


variable_runs = {
    10: "n420_k1_vs_k10_scaled_lr1e-4_mseed1",
    21: "n420_k1_vs_k21_scaled_lr1e-4_mseed1_min80",
}

variable_main = available([
    (k, test_auc(variable_summary(label)))
    for k, label in variable_runs.items()
])
fixed_store_runs = {
    5: "n420_fixed_k1_vs_k5_scaled_lr1e-4_decay80_mseed1_pointonly",
    7: "n420_fixed_k1_vs_k7_scaled_lr1e-4_decay80_mseed1_pointonly",
}
fixed_store_main = available([
    (k, test_auc(library_summary(label)))
    for k, label in fixed_store_runs.items()
])
library_main = available([(
    42,
    test_auc(library_summary(
        "n420_recipe_bs4_expanded_scaled_lr1e-4_mseed1_pointonly"
    )),
)])
perlmutter_runs = {
    5: "pm_n420_k1_vs_k5_synthetic_scaled_lr1e-4_decay80_mseed1_v1",
    42: "pm_n420_k1_vs_k42_synthetic_scaled_lr1e-4_decay80_mseed1_v1",
}
perlmutter_main = available([
    (k, test_auc(library_summary(label)))
    for k, label in perlmutter_runs.items()
])
if (
    not variable_main
    or not fixed_store_main
    or not library_main
    or len(perlmutter_main) != len(perlmutter_runs)
):
    raise SystemExit("missing required GEN result summaries")

main_points = sorted(variable_main + fixed_store_main + library_main)
x, y = coordinates(main_points)

plt.rcParams["font.family"] = "serif"
fig, axis = plt.subplots(figsize=(6.2, 4.4))
axis.plot(
    x,
    y,
    marker="o",
    markersize=7,
    linewidth=2,
    color="#0072B2",
    label="K=1 vs K",
)
perlmutter_x, perlmutter_y = coordinates(sorted(perlmutter_main))
axis.scatter(
    perlmutter_x,
    perlmutter_y,
    s=64,
    color="#D55E00",
    label="Perlmutter reruns",
    zorder=3,
)
axis.axhline(0.5, color="0.45", linewidth=1, linestyle="--")
axis.set_xticks(x)
axis.invert_xaxis()
axis.set_xlabel("K")
axis.set_ylabel("Test AUC")
axis.set_title("GEN PFN AUC versus mother-reuse factor")
axis.set_ylim(0.48, 1.01)
axis.grid(axis="y", alpha=0.25, linewidth=0.5)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False)

fig.tight_layout()
output = ROOT / "plots" / "gen_n420_auc_vs_k"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
plt.close(fig)

for k, auc in main_points:
    print(f"K={k}: test AUC {auc:.6f}")
for k, auc in sorted(perlmutter_main):
    print(f"Perlmutter K={k}: test AUC {auc:.6f}")
print(f"wrote {output.with_suffix('.pdf')}")
print(f"wrote {output.with_suffix('.png')}")
