"""Joint analysis of the fixed-hyperparameter and TPE-tuned protocols.

Run after `analyze_results.py` and `analyze_results.py --tuned`. Produces:
  * pairwise_all_fixed.csv / pairwise_all_tpe.csv -- all 15 within-protocol
    pairs: Nemenyi p (from the 6-config Friedman) + Vargha-Delaney A12.
  * nemenyi_joint.csv, ranks_joint.csv -- all 12 configurations (6 fixed +
    6 tuned) in one Friedman/Nemenyi analysis, blocked by seed.
  * figures/fig_mae_fixed_vs_tuned.pdf -- paired boxplot per configuration.
  * figures/fig_nemenyi_heatmaps.pdf -- all-vs-all Nemenyi p-values, both protocols.
  * figures/fig_cd_joint.pdf -- critical-difference diagram, 12 configurations.
  * figures/fig_tpe_convergence.pdf -- best-so-far validation MAE per TPE trial.
"""
import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
CONFIGS = ["MLP", "LSTM", "GRU", "Average", "StaticStacking", "BoostingStacking"]
SHORT = {"MLP": "MLP", "LSTM": "LSTM", "GRU": "GRU", "Average": "Avg",
         "StaticStacking": "Stack", "BoostingStacking": "Boost"}
C_FIXED, C_TUNED = "#E69F00", "#0072B2"
ARCH_COLORS = {"MLP": "#0072B2", "LSTM": "#E69F00", "GRU": "#009E73"}  # same as pipeline figure
INK, MUTED = "#222222", "#6b6b6b"

plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6})


def vd_a12(x, y):
    m, n = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))
    return (r[:m].sum() / m - (m + 1) / 2) / n


def effect_label(a12):
    d = abs(a12 - 0.5)
    return "large" if d > 0.21 else "medium" if d > 0.14 else "small" if d > 0.06 else "negligible"


def wide(path):
    return pd.read_csv(os.path.join(HERE, path)).pivot(index="seed", columns="config",
                                                       values="mae")[CONFIGS]


fixed, tuned = wide("results_mx.csv"), wide("results_mx_tpe.csv")
seeds = fixed.index.intersection(tuned.index)
fixed, tuned = fixed.loc[seeds], tuned.loc[seeds]

# ---- all 15 pairs within each protocol --------------------------------------
nem = {}
for tag, w in [("fixed", fixed), ("tpe", tuned)]:
    p = sp.posthoc_nemenyi_friedman(w.values)
    p.index = p.columns = CONFIGS
    nem[tag] = p
    rows = []
    for a, b in itertools.combinations(CONFIGS, 2):
        a12 = vd_a12(w[a].values, w[b].values)
        rows.append(dict(a=a, b=b, mean_a=w[a].mean(), mean_b=w[b].mean(),
                         nemenyi_p=p.loc[a, b], A12=a12, effect=effect_label(a12)))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"pairwise_all_{tag}.csv"), index=False)
    print(f"\n=== All pairs ({tag}) ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---- joint 12-configuration analysis ----------------------------------------
joint = pd.concat([fixed.add_suffix(" (F)"), tuned.add_suffix(" (T)")], axis=1)
chi2, pj = stats.friedmanchisquare(*[joint[c].values for c in joint.columns])
ranks = joint.rank(axis=1).mean().sort_values()
pjoint = sp.posthoc_nemenyi_friedman(joint.values)
pjoint.index = pjoint.columns = joint.columns
pjoint.to_csv(os.path.join(HERE, "nemenyi_joint.csv"))
ranks.to_csv(os.path.join(HERE, "ranks_joint.csv"), header=["mean_rank"])
k, n = joint.shape[1], joint.shape[0]
cd = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
print(f"\n=== Joint Friedman (k={k}, N={n}) chi2={chi2:.3f} p={pj:.2e}  CD={cd:.3f}")
print(ranks.round(3).to_string())

# ---- Fig: paired boxplot fixed vs tuned -------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 2.5))
pos = np.arange(len(CONFIGS))
for off, w, col, lab, hatch in [(-0.19, fixed, C_FIXED, "Fixed", "///"),
                                 (0.19, tuned, C_TUNED, "TPE-tuned", None)]:
    bp = ax.boxplot([w[c].values for c in CONFIGS], positions=pos + off, widths=0.32,
                    patch_artist=True, showmeans=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.1),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=INK, markersize=4),
                    whiskerprops=dict(color=MUTED, linewidth=0.8),
                    capprops=dict(color=MUTED, linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=col,
                                    markeredgecolor="white", markeredgewidth=0.5))
    for b in bp["boxes"]:
        b.set(facecolor=col, alpha=0.85, edgecolor=INK, linewidth=0.7)
        if hatch:
            b.set_hatch(hatch)
    bp["boxes"][0].set_label(lab)
for i, c in enumerate(CONFIGS):
    p = stats.wilcoxon(tuned[c], fixed[c]).pvalue
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    top = max(fixed[c].max(), tuned[c].max())
    ax.text(i, top + 0.6, star, ha="center", va="bottom", color=INK, fontsize=8)
ax.set_xticks(pos, [SHORT[c] for c in CONFIGS])
ax.set_ylabel("Test MAE (W/m$^2$)")
ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
ax.set_ylim(top=ax.get_ylim()[1] + 1.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_mae_fixed_vs_tuned.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_mae_fixed_vs_tuned.png"), dpi=300, bbox_inches="tight")

# ---- Fig: all-vs-all Nemenyi heatmaps ---------------------------------------
cmap = ListedColormap(["#08306b", "#2171b5", "#9ecae1", "#f2f2f2"])
norm = BoundaryNorm([0, 0.001, 0.01, 0.05, 1.0001], cmap.N)
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
for ax, (tag, title, w) in zip(axes, [("fixed", "Fixed hyperparameters", fixed),
                                      ("tpe", "TPE-tuned hyperparameters", tuned)]):
    P = nem[tag].values.copy()
    mask = np.triu(np.ones_like(P, dtype=bool))
    Pm = np.ma.array(P, mask=mask)
    im = ax.imshow(Pm, cmap=cmap, norm=norm)
    for i, j in itertools.product(range(6), range(6)):
        if i > j:
            a12 = vd_a12(w[CONFIGS[i]].values, w[CONFIGS[j]].values)
            txt = f"{P[i, j]:.3f}" if P[i, j] >= 0.001 else "<.001"
            dark = P[i, j] < 0.01
            ax.text(j, i - 0.13, txt, ha="center", va="center", fontsize=5.5,
                    color="white" if dark else INK)
            ax.text(j, i + 0.22, f"A={a12:.2f}", ha="center", va="center", fontsize=4.6,
                    color="white" if dark else MUTED)
    ax.set_xticks(range(6), [SHORT[c] for c in CONFIGS], rotation=0)
    ax.set_yticks(range(6), [SHORT[c] for c in CONFIGS])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(5.5, 0.5)
    ax.set_title(title, fontsize=7, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
cb = fig.colorbar(im, ax=axes, ticks=[0.0005, 0.0055, 0.03, 0.5], fraction=0.025, pad=0.02)
cb.ax.set_yticklabels(["<0.001", "<0.01", "<0.05", "n.s."])
cb.set_label("Nemenyi p (row vs. column)")
cb.outline.set_visible(False)
fig.savefig(os.path.join(FIG, "fig_nemenyi_heatmaps.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_nemenyi_heatmaps.png"), dpi=300, bbox_inches="tight")

# ---- Fig: critical-difference diagram, 12 configurations --------------------
fig, ax = plt.subplots(figsize=(4.8, 2.3))
sp.critical_difference_diagram(
    ranks, pjoint, ax=ax, label_fmt_left="{label} ({rank:.2f})  ",
    label_fmt_right="  ({rank:.2f}) {label}",
    color_palette={c: (C_FIXED if c.endswith("(F)") else C_TUNED) for c in joint.columns},
    label_props={"fontsize": 6.5},
    crossbar_props={"color": INK, "linewidth": 1.4},
    elbow_props={"linewidth": 0.7})
fig.savefig(os.path.join(FIG, "fig_cd_joint.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_cd_joint.png"), dpi=300, bbox_inches="tight")

# ---- Fig: TPE convergence ---------------------------------------------------
# Top: best-so-far validation MAE (completed trials). Bottom: trial outcome
# strip -- filled = trained to early stopping, hollow = pruned by the median rule.
trials = pd.read_csv(os.path.join(HERE, "tpe_trials.csv"))
fig, (ax, axs) = plt.subplots(2, 1, figsize=(4.8, 2.7), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.08))
label_y = {}
for arch, col in ARCH_COLORS.items():
    t = trials[trials.model == arch].sort_values("number")
    done = t[t.state == "COMPLETE"]
    curve = t["value"].where(t.state == "COMPLETE").cummin().ffill()
    ax.step(t["number"] + 1, curve, where="post", color=col, linewidth=2, label=arch)
    ax.scatter(done["number"] + 1, done["value"], s=14, color=col, edgecolor="white",
               linewidth=0.5, zorder=3)
    label_y[arch] = curve.iloc[-1]
for (arch, y), dy in zip(sorted(label_y.items(), key=lambda kv: kv[1]), [-1.2, 0.2, 1.6]):
    ax.text(30.8, y + dy, f"{arch} {y:.1f}", va="center", fontsize=6.5, color=INK)
ax.set_ylabel("Validation MAE\n(W/m$^2$)")
ax.set_ylim(38, 60)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7)
for row, (arch, col) in enumerate(ARCH_COLORS.items()):
    t = trials[trials.model == arch]
    for state, fc in [("COMPLETE", col), ("PRUNED", "none")]:
        tt = t[t.state == state]
        axs.scatter(tt["number"] + 1, np.full(len(tt), row), s=16, facecolor=fc,
                    edgecolor=col, linewidth=0.8)
axs.set_yticks(range(3), list(ARCH_COLORS))
axs.set_ylim(2.6, -0.6)
axs.set_xlim(0.3, 33.5)
axs.set_xlabel("TPE trial (filled = completed, hollow = pruned)")
axs.spines[["top", "right", "left"]].set_visible(False)
axs.tick_params(axis="y", length=0)
fig.savefig(os.path.join(FIG, "fig_tpe_convergence.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_tpe_convergence.png"), dpi=300, bbox_inches="tight")

# Convergence summary: trial at which the final best was first reached.
for arch in ARCH_COLORS:
    t = trials[(trials.model == arch) & (trials.state == "COMPLETE")]
    bi = int(t.loc[t["value"].idxmin(), "number"]) + 1
    dur = trials[trials.model == arch]
    print(f"{arch}: best at trial {bi}/30; pruned {int((dur.state=='PRUNED').sum())}; "
          f"mean dur complete={dur[dur.state=='COMPLETE'].duration.mean():.1f}s "
          f"pruned={dur[dur.state=='PRUNED'].duration.mean():.1f}s; "
          f"total={dur.duration.sum():.0f}s")
print("Saved figures to", os.path.abspath(FIG))
