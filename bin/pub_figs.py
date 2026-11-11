"""Publication-grade Results figures (vector PDF) for the lightweight-classifier paper."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'results' / '2026-11-11_paper_assets'
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

P = ASSETS / "data"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS / "figures"; OUT.mkdir(parents=True, exist_ok=True)
F = json.loads((ASSETS / "facts.json").read_text())
S, M, K, D = [pd.read_csv(P / f"{n}.csv") for n in ("S", "M", "K", "D")]
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "lines.linewidth": 1.0, "savefig.dpi": 600})
INK, MUTED, GRID = "#1A1A1A", "#6B6B6B", "#E6E6E6"
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C6D00", "#000000", "#882255", "#44AA99", "#999933"]
RF_C, GBM_C, LR_C, ACC_C, MCC_C, SENS_C = "#D55E00", "#0072B2", "#7F7F7F", "#9E9E9E", "#009E73", "#E69F00"
ORDER = ["sepsis_survival", "diabetes_130", "physionet_2019_AtoB", "support2", "physionet_2012", "sepsis_sirs", "ckd_uae",
         "depression_heart_failure", "cardiac_arrest_spain", "heart_failure_chicco", "lung_cancer_icu"]
NAME = {"sepsis_survival": "Sepsis survival", "diabetes_130": "Diabetes 130", "physionet_2019_AtoB": "PhysioNet 2019 A→B",
        "support2": "SUPPORT2", "physionet_2012": "PhysioNet 2012", "sepsis_sirs": "Sepsis/SIRS ICU", "ckd_uae": "CKD (UAE)",
        "depression_heart_failure": "Depression & heart failure", "cardiac_arrest_spain": "Cardiac arrest (Spain)",
        "heart_failure_chicco": "Heart failure", "lung_cancer_icu": "Lung cancer ICU"}
COL = dict(zip(ORDER, OI))
EVENTS = {d: F["meta_data"][d]["events"] for d in ORDER}
LARGE = ["diabetes_130", "sepsis_survival", "physionet_2019_AtoB", "support2", "physionet_2012"]

def style(ax, grid_x=True):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if grid_x: ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True); ax.tick_params(colors=INK)

def letter(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right", color=INK)

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)

# ================= Figure 1: accuracy conceals differences that MCC reveals
acc = F["accmcc"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.08))
y = np.arange(len(ORDER))[::-1]
for yi, d in zip(y, ORDER):
    r = acc[d]; a.plot([r["acc_range"], r["mcc_range"]], [yi, yi], color="#C8C8C8", lw=1.6, zorder=1, solid_capstyle="round")
    a.scatter(r["acc_range"], yi, s=26, color=ACC_C, zorder=3, edgecolor="white", lw=0.4)
    a.scatter(r["mcc_range"], yi, s=26, color=MCC_C, zorder=3, edgecolor="white", lw=0.4)
a.set_yticks(y); a.set_yticklabels([NAME[d] for d in ORDER]); a.set_xlim(-0.004, 0.18); a.set_ylim(-0.7, len(ORDER) - 0.3)
a.set_xlabel("Range across the seven classifiers\n(maximum − minimum of the median over splits)")
a.legend(handles=[Line2D([], [], marker="o", ls="", color=ACC_C, label="Accuracy"), Line2D([], [], marker="o", ls="", color=MCC_C, label="MCC")],
         frameon=False, loc="lower right", handletextpad=0.2)
style(a); letter(a, "a", x=-0.36)
M10 = M[(M.budget == 0.1) & ~M.model.str.contains("retrain")].copy()
M10["accuracy"] = (M10.TP + M10.TN) / (M10.TP + M10.TN + M10.FP + M10.FN)
med = M10.groupby(["dataset", "model"])[["accuracy", "mcc"]].median()
for yi, d in zip(y, ORDER):
    g = med.loc[d]
    if g.accuracy.nunique() == 1:
        b.text(0.02, yi, "accuracy identical across classifiers", va="center", fontsize=6.3, color=MUTED, style="italic")
        continue
    tau = stats.kendalltau(g.accuracy, g.mcc).statistic
    b.barh(yi, tau, height=0.55, color=MCC_C if tau > 0 else "#B03A2E", alpha=0.85, zorder=2)
    if acc[d]["best_acc"] != acc[d]["best_mcc"]:
        b.scatter(1.06, yi, marker="D", s=14, color=INK, clip_on=False, zorder=3)
b.axvline(0, color=INK, lw=0.6); b.set_xlim(-1, 1.12); b.set_xticks([-1, -0.5, 0, 0.5, 1])
b.set_yticks(y); b.set_yticklabels([]); b.set_ylim(a.get_ylim())
b.set_xlabel("Kendall's τ between the classifier rankings\nby accuracy and by MCC")
b.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=3.5, label="best classifier differs\n(accuracy vs MCC)")],
         frameon=False, loc="lower left", handletextpad=0.2, fontsize=6.3)
style(b); letter(b, "b", x=0.0)
save(fig, "fig1_accuracy_mcc")

# ================= Figure 2: excess over retraining noise, random-effects meta-analysis
def forest(ax, cand, title, xlim, text=True):
    e = S[(S.budget == 0.1) & (S.candidate == cand)].set_index("dataset").loc[ORDER]
    se = (e.excess_newly_missed_per100_ci_high - e.excess_newly_missed_per100_ci_low) / (2 * stats.t.ppf(.975, e.splits - 1))
    r = F[f"meta|0.1|{cand}|excess_newly_missed_per100"]
    w = 1 / (se**2 + r["tau2"]); w = w / w.sum()
    yy = np.arange(len(ORDER))[::-1] + 2.0
    for yi, (d, row), wi in zip(yy, e.iterrows(), w):
        ax.plot([row.excess_newly_missed_per100_ci_low, row.excess_newly_missed_per100_ci_high], [yi, yi], color=INK, lw=0.8, zorder=2)
        ax.scatter(row.excess_newly_missed_per100, yi, marker="s", s=10 + 160 * wi, color=COL[d] if False else (RF_C if cand.startswith("rf") else GBM_C),
                   edgecolor="white", lw=0.3, zorder=3)
        lo, hi = row.excess_newly_missed_per100_ci_low, row.excess_newly_missed_per100_ci_high
        if hi > xlim[1]: ax.annotate("", xy=(xlim[1], yi), xytext=(xlim[1] - 0.8, yi), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=6))
        if text:
            ax.text(1.03, yi, f"{row.excess_newly_missed_per100:+.1f} [{lo:+.1f}, {hi:+.1f}]", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6.4, color=INK)
    yd = 0.6; h = 0.42
    diamond = Polygon([[r["lo"], yd], [r["pooled"], yd + h], [r["hi"], yd], [r["pooled"], yd - h]], closed=True, color=INK, zorder=3)
    ax.add_patch(diamond); ax.plot([r["pi_lo"], r["pi_hi"]], [yd, yd], color=INK, lw=0.6, ls=(0, (1.5, 1.2)), zorder=2)
    if text:
        ax.text(1.03, yd, f"{r['pooled']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=6.4, color=INK, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(*xlim); ax.set_ylim(-0.2, len(ORDER) + 2.7)
    ax.set_yticks(list(yy) + [yd]); ax.set_title(title, loc="left", color=INK, pad=4)
    ax.set_xlabel("Excess newly missed positives per 100 events")
    style(ax)
    return yy, yd, r

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.9), gridspec_kw=dict(wspace=0.95))
yy, yd, ra = forest(a, "rf_thinned", "Random forest, 128 → 16 trees (post-training thinning)", (-6, 20))
a.set_yticklabels([f"{NAME[d]} ({EVENTS[d]:,})" for d in ORDER] + [f"Pooled, random effects (I² = {100 * ra['I2']:.0f}%)"])
_, _, rb = forest(b, "gbm_truncated", "Gradient boosting, 400 → 100 rounds (truncation)", (-6, 20))
b.set_yticklabels([])
for ax, s in ((a, "a"), (b, "b")): letter(ax, s, x=-0.02 if s == "b" else -0.62)
a.text(-0.03, len(ORDER) + 2.1, "Dataset (events)", transform=a.get_yaxis_transform(), fontsize=6.4, fontweight="bold", va="center", ha="right")
a.text(1.03, len(ORDER) + 2.1, "Excess [95% CI]", transform=a.get_yaxis_transform(), fontsize=6.4, fontweight="bold", va="center")
b.text(1.03, len(ORDER) + 2.1, "Excess [95% CI]", transform=b.get_yaxis_transform(), fontsize=6.4, fontweight="bold", va="center")
save(fig, "fig2_excess_meta")

# ================= Figure 3: dose–response and mechanism
fig = plt.figure(figsize=(7.0, 5.3))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.34)
a, b, c, d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
Db = D[D.budget == 0.1]
trees = [64, 32, 16, 8, 4]
for ds in [x for x in ORDER if x in LARGE]:
    e = Db[Db.dataset == ds].set_index("trees").loc[trees]
    xs = np.log2(trees)
    a.errorbar(xs, e.excess_missed_per100, yerr=[e.excess_missed_per100 - e.ci_low, e.ci_high - e.excess_missed_per100],
               color=COL[ds], marker="o", ms=3, capsize=1.5, lw=1.0, elinewidth=0.6, label=NAME[ds])
a.axhline(0, color=MUTED, lw=0.6, ls="--"); a.set_xlim(6.3, 1.7)
a.set_xticks(np.log2(trees)); a.set_xticklabels([str(t) for t in trees])
a.set_xlabel("Trees kept from the 128-tree reference forest"); a.set_ylabel("Excess newly missed positives\nper 100 events")
top = a.secondary_xaxis("top"); top.set_xticks(np.log2(trees)); top.set_xticklabels(["50%", "25%", "12.5%", "6.25%", "3.1%"])
top.set_xlabel("Storage relative to the reference", labelpad=3); top.tick_params(labelsize=6.5)
a.legend(frameon=False, loc="upper left", fontsize=6.2, handlelength=1.4)
style(a, grid_x=False); a.grid(True, which="major", color=GRID, lw=0.5); letter(a, "a", x=-0.2, y=1.17)
Kb = K[K.budget == 0.1]
for ds in ORDER:
    e = Kb[Kb.dataset == ds]
    b.scatter(e.expected_newly_missed.clip(lower=0.4), e.observed_newly_missed.clip(lower=0.4), s=7, color=COL[ds], alpha=0.75, lw=0, label=NAME[ds])
hi = 400
b.plot([0.4, hi], [0.4, hi], color=MUTED, lw=0.6, ls="--"); b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(0.4, hi); b.set_ylim(0.4, hi)
b.set_xlabel("Expected (tree-sampling model)"); b.set_ylabel("Observed newly missed positives")
ma = F["mech_all"]
b.text(0.04, 0.96, f"r = {ma['r']:.2f} across {ma['n']} splits\nexpected {ma['exp']:,.0f} · observed {ma['obs']:,.0f}", transform=b.transAxes,
       va="top", fontsize=6.4, color=INK)
style(b, grid_x=False); b.grid(True, which="major", color=GRID, lw=0.5); letter(b, "b", x=-0.2, y=1.04)
q = F["quartiles"]; xq = np.arange(4)
c.bar(xq - 0.2, q["retrained"], 0.38, color="#BDBDBD", label="Retrained reference (mean of 3)")
c.bar(xq + 0.2, q["thinned"], 0.38, color=RF_C, label="Thinned forest (16 of 128 trees)")
for xi, v in zip(xq - 0.2, q["retrained"]): c.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.2, color=INK)
for xi, v in zip(xq + 0.2, q["thinned"]): c.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.2, color=INK)
c.set_xticks(xq); c.set_xticklabels(["Q1\nclosest to\nthreshold", "Q2", "Q3", "Q4\nfurthest above\nthreshold"])
c.set_ylim(0, 60); c.set_ylabel("Reference true-positive alerts lost (%)")
c.set_xlabel("Quartile of the reference score margin")
c.legend(frameon=False, loc="upper right", fontsize=6.2)
style(c, grid_x=False); c.grid(True, color=GRID, lw=0.5); letter(c, "c", x=-0.2, y=1.04)
yd_ = np.arange(len(ORDER))[::-1]
for yi, ds in zip(yd_, ORDER):
    e = Kb[Kb.dataset == ds].auroc_predicting_changes.dropna()
    d.plot([e.quantile(0.1), e.quantile(0.9)], [yi, yi], color=COL[ds], lw=1.4, solid_capstyle="round")
    d.scatter(e.median(), yi, s=16, color=COL[ds], edgecolor="white", lw=0.4, zorder=3)
d.set_yticks(yd_); d.set_yticklabels([NAME[x] for x in ORDER], fontsize=6.5); d.set_xlim(0.5, 1.0)
d.axvline(0.5, color=MUTED, lw=0.6)
d.set_xlabel("ROC AUC of the predicted flip probability\n(median and 10th–90th percentile over splits)")
style(d); d.grid(True, color=GRID, lw=0.5); letter(d, "d", x=-0.52, y=1.04)
save(fig, "fig3_dose_mechanism")

# ================= Supplementary Figure S1: metric suite on [0, 1]
NM = {"rf_parent": "RF 128 (reference)", "rf_thinned": "RF 128→16 thinned", "rf_direct_small": "RF 16 direct",
      "gbm_parent": "GBM 400 (reference)", "gbm_truncated": "GBM 400→100 truncated", "gbm_direct_small": "GBM small direct", "logistic": "Logistic regression"}
M10["sens"] = M10.sensitivity
md = M10.groupby(["dataset", "model"])[["accuracy", "mcc", "sens"]].median()
fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.2), sharex=True)
for ax, ds in zip(axes.flat, ORDER):
    mods = [m for m in NM if (ds, m) in md.index]; yy = np.arange(len(mods))[::-1]
    for key, colr, mk in (("accuracy", ACC_C, "s"), ("mcc", MCC_C, "o"), ("sens", SENS_C, "^")):
        ax.scatter([md.loc[(ds, m), key] for m in mods], yy, color=colr, marker=mk, s=12, zorder=3, lw=0)
    ax.set_yticks(yy); ax.set_yticklabels([NM[m] for m in mods], fontsize=5.6); ax.set_xlim(0, 1); ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_title(NAME[ds], loc="left", fontsize=7, pad=2); style(ax)
axes.flat[8].tick_params(labelbottom=True)
ax = axes.flat[-1]; ax.axis("off")
ax.legend(handles=[Line2D([], [], marker="s", ls="", color=ACC_C, label="Accuracy"), Line2D([], [], marker="o", ls="", color=MCC_C, label="MCC"),
                   Line2D([], [], marker="^", ls="", color=SENS_C, label="Sensitivity")], frameon=False, loc="center", fontsize=7)
fig.tight_layout(h_pad=0.8, w_pad=0.6); save(fig, "figS1_metrics_unit_interval")

# ================= Supplementary Figure S2: all compressed variants, per dataset
cands = [("rf_thinned", "RF 128→16 thinned", RF_C), ("rf_direct_small", "RF 16 direct", "#E8A07A"),
         ("gbm_truncated", "GBM 400→100 truncated", GBM_C), ("gbm_direct_small", "GBM small direct", "#7FB3D8")]
fig, ax = plt.subplots(figsize=(7.0, 5.2))
base = np.arange(len(ORDER))[::-1] * 1.0
for j, (cand, lab, colr) in enumerate(cands):
    e = S[(S.budget == 0.1) & (S.candidate == cand)].set_index("dataset").loc[ORDER]
    off = (1.5 - j) * 0.18
    ax.errorbar(e.excess_newly_missed_per100, base + off, xerr=[e.excess_newly_missed_per100 - e.excess_newly_missed_per100_ci_low,
                e.excess_newly_missed_per100_ci_high - e.excess_newly_missed_per100], fmt="o", ms=2.6, color=colr, elinewidth=0.7, capsize=0, label=lab)
ax.axvline(0, color=MUTED, ls="--", lw=0.6); ax.set_yticks(base); ax.set_yticklabels([NAME[d] for d in ORDER])
ax.set_xlabel("Excess newly missed positives per 100 events (> 0: harm beyond retraining noise)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.07), fontsize=6.4)
style(ax); fig.tight_layout(); save(fig, "figS2_all_variants")
print("figures ->", OUT)
