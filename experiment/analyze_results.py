"""Statistical analysis of the 6-configuration ensemble comparison:
Friedman rank test (blocked by seed) -> Nemenyi post-hoc -> Vargha-Delaney
effect sizes for the comparisons that matter for the paper's argument.
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scikit_posthocs as sp
    HAS_SP = True
except ImportError:
    HAS_SP = False

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = "--tuned" in sys.argv
SUF = "_tpe" if TUNED else ""
df = pd.read_csv(os.path.join(HERE, f"results_mx{SUF}.csv"))
CONFIGS = ["MLP", "LSTM", "GRU", "Average", "StaticStacking", "BoostingStacking"]
N_SEEDS = df["seed"].nunique()
print(f"Seeds: {N_SEEDS}  Configs: {CONFIGS}")

# ---------- summary table ----------
summary = df.groupby("config")[["mae", "rmse", "r2"]].agg(["mean", "std"])
summary = summary.reindex(CONFIGS)
summary.to_csv(os.path.join(HERE, f"summary{SUF}.csv"))
print("\n=== Summary (mean +/- std) ===")
for cfg in CONFIGS:
    m = summary.loc[cfg]
    print(f"{cfg:18s} MAE={m[('mae','mean')]:.2f}+-{m[('mae','std')]:.2f}  "
          f"RMSE={m[('rmse','mean')]:.2f}+-{m[('rmse','std')]:.2f}  "
          f"R2={m[('r2','mean')]:.4f}+-{m[('r2','std')]:.4f}")

# ---------- Friedman test on test MAE, blocked by seed ----------
wide = df.pivot(index="seed", columns="config", values="mae")[CONFIGS]
stat, p = stats.friedmanchisquare(*[wide[c].values for c in CONFIGS])
print(f"\n=== Friedman test (test MAE) ===\nchi2={stat:.4f}  p={p:.6f}  "
      f"{'SIGNIFICANT' if p < 0.05 else 'not significant'} at alpha=0.05")

mean_ranks = wide.rank(axis=1, ascending=True).mean(axis=0).sort_values()
print("\nMean ranks (lower = better MAE):")
print(mean_ranks.to_string())

# ---------- Nemenyi post-hoc ----------
if HAS_SP and p < 0.05:
    nemenyi = sp.posthoc_nemenyi_friedman(wide[CONFIGS].values)
    nemenyi.index = CONFIGS
    nemenyi.columns = CONFIGS
    nemenyi.to_csv(os.path.join(HERE, f"nemenyi_pvalues{SUF}.csv"))
    print("\n=== Nemenyi post-hoc p-values ===")
    print(nemenyi.round(4).to_string())
elif not HAS_SP:
    print("\nscikit-posthocs not available, skipping Nemenyi.")
else:
    print("\nFriedman not significant; Nemenyi post-hoc not formally warranted.")


# ---------- Vargha-Delaney A12 ----------
def vd_a12(x, y):
    m, n = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))
    rx = r[:m].sum()
    a12 = (rx / m - (m + 1) / 2) / n
    return a12


def effect_label(a12):
    d = abs(a12 - 0.5)
    if d > 0.21:
        return "large"
    if d > 0.14:
        return "medium"
    if d > 0.06:
        return "small"
    return "negligible"


print("\n=== Key pairwise comparisons (test MAE): Mann-Whitney U + Vargha-Delaney A12 ===")
best_single = mean_ranks.index[0] if mean_ranks.index[0] in ["MLP", "LSTM", "GRU"] else \
    [c for c in mean_ranks.index if c in ["MLP", "LSTM", "GRU"]][0]
pairs = [
    ("BoostingStacking", "StaticStacking"),
    ("BoostingStacking", "Average"),
    ("BoostingStacking", best_single),
    ("StaticStacking", best_single),
    ("Average", best_single),
]
pairwise_rows = []
for a, b in pairs:
    x, y = wide[a].values, wide[b].values
    u, pw = stats.mannwhitneyu(x, y, alternative="two-sided")
    a12 = vd_a12(x, y)
    row = dict(pair=f"{a} vs {b}", U=u, p=pw, A12=a12, effect=effect_label(a12),
               mean_a=x.mean(), mean_b=y.mean())
    pairwise_rows.append(row)
    print(f"{a:18s} vs {b:18s}: U={u:.1f} p={pw:.4f} A12={a12:.3f} ({effect_label(a12)})  "
          f"mean_MAE: {x.mean():.2f} vs {y.mean():.2f}")

pd.DataFrame(pairwise_rows).to_csv(os.path.join(HERE, f"pairwise_tests{SUF}.csv"), index=False)

# ---------- figure ----------
# Okabe-Ito colorblind-safe palette, matching figures/make_pipeline_figure.py
BOX_COLORS = ["#5D5D5D", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(7, 3.6))
data = [wide[c].values for c in CONFIGS]
bp = ax.boxplot(data, tick_labels=CONFIGS, showmeans=True, patch_artist=True,
                 medianprops=dict(color="black", linewidth=1.3),
                 meanprops=dict(marker="D", markerfacecolor="white",
                                markeredgecolor="black", markersize=5))
for patch, color in zip(bp["boxes"], BOX_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_ylabel("Test MAE (W/m$^2$)")
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(HERE, f"fig_mae_boxplot{SUF}.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, f"fig_mae_boxplot{SUF}.pdf"), bbox_inches="tight")
print(f"\nSaved fig_mae_boxplot{SUF}.png/.pdf, summary{SUF}.csv, pairwise_tests{SUF}.csv")

# Fixed vs. TPE-tuned hyperparameters, per configuration (paired by seed:
# same seed -> same initialisation stream, only the hyperparameters differ).
if TUNED and os.path.exists(os.path.join(HERE, "results_mx.csv")):
    fixed = pd.read_csv(os.path.join(HERE, "results_mx.csv")).pivot(
        index="seed", columns="config", values="mae")
    common = wide.index.intersection(fixed.index)
    print(f"\n=== Fixed vs. TPE-tuned (test MAE, {len(common)} paired seeds): Wilcoxon + A12 ===")
    rows = []
    for cfg in CONFIGS:
        x, y = wide.loc[common, cfg].values, fixed.loc[common, cfg].values
        w, pw = stats.wilcoxon(x, y)
        a12 = vd_a12(x, y)
        rows.append(dict(config=cfg, mae_fixed=y.mean(), mae_tuned=x.mean(),
                         delta=x.mean() - y.mean(), rel_change=(x.mean() - y.mean()) / y.mean(),
                         W=w, p=pw, A12=a12, effect=effect_label(a12)))
        print(f"{cfg:18s} fixed={y.mean():.2f}  tuned={x.mean():.2f}  "
              f"delta={x.mean()-y.mean():+.2f} ({100*(x.mean()-y.mean())/y.mean():+.1f}%)  "
              f"p={pw:.2e}  A12={a12:.3f} ({effect_label(a12)})")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "fixed_vs_tuned.csv"), index=False)
