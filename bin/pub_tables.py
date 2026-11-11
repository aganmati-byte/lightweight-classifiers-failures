"""LaTeX (booktabs) tables for the Results section."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'results' / '2026-11-11_paper_assets'
import numpy as np, pandas as pd
from scipy import stats

P = ASSETS / "data"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ASSETS / "tables"; OUT.mkdir(parents=True, exist_ok=True)
F = json.loads((ASSETS / "facts.json").read_text())
S, M, D, Z = [pd.read_csv(P / f"{n}.csv") for n in ("S", "M", "D", "Z")]
ORDER = ["sepsis_survival", "diabetes_130", "physionet_2019_AtoB", "support2", "physionet_2012", "sepsis_sirs", "ckd_uae",
         "depression_heart_failure", "cardiac_arrest_spain", "heart_failure_chicco", "lung_cancer_icu"]
NAME = {"sepsis_survival": "Sepsis survival", "diabetes_130": "Diabetes 130", "physionet_2019_AtoB": r"PhysioNet 2019 A$\rightarrow$B",
        "support2": "SUPPORT2", "physionet_2012": "PhysioNet 2012", "sepsis_sirs": "Sepsis/SIRS ICU", "ckd_uae": "CKD (UAE)",
        "depression_heart_failure": r"Depression \& heart failure", "cardiac_arrest_spain": "Cardiac arrest (Spain)",
        "heart_failure_chicco": "Heart failure", "lung_cancer_icu": "Lung cancer ICU"}
CAND = [("rf_thinned", r"RF 128$\rightarrow$16, thinned"), ("rf_direct_small", "RF 16, trained directly"),
        ("gbm_truncated", r"GBM 400$\rightarrow$100, truncated"), ("gbm_direct_small", "GBM small, trained directly")]
MODELS = {"rf_parent": "RF 128 (reference)", "rf_thinned": r"RF 128$\rightarrow$16 thinned", "rf_direct_small": "RF 16 direct",
          "gbm_parent": "GBM 400 (reference)", "gbm_truncated": r"GBM 400$\rightarrow$100 truncated", "gbm_direct_small": "GBM small direct",
          "logistic": "Logistic regression"}
num = lambda x, d=2: (f"{x:+.{d}f}").replace("-", "$-$") if np.isfinite(x) else "--"
ci = lambda m, lo, hi, d=2: f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"
thou = lambda n: f"{int(n):,}".replace(",", "{,}")

# storage relative to reference (Diabetes 130, first split)
z = Z[Z.dataset == "diabetes_130"].set_index("model").bytes
REL = {"rf_thinned": z["rf_thinned"] / z["rf_parent"], "rf_direct_small": z["rf_direct_small"] / z["rf_parent"],
       "gbm_truncated": z["gbm_truncated"] / z["gbm_parent"], "gbm_direct_small": z["gbm_direct_small"] / z["gbm_parent"]}

def stacked(caption, label, blocks, size="\\small"):
    """One table, blocks stacked vertically; each block is (colspec, header_lines, rows) at full text width."""
    out = ["\\begin{table}[!htb]", "\\centering", "\\caption{" + caption + "}", "\\label{" + label + "}", "{" + size]
    for i, (spec, head, rows) in enumerate(blocks):
        if i: out.append("\\par\\vspace{1.2em}")
        out += ["\\begin{tabular}{" + spec + "}", "\\toprule", head, "\\midrule",
                "\n".join(rows), "\\bottomrule", "\\end{tabular}"]
    out += ["}", "\\end{table}", ""]
    return "\n".join(out)

def meta_table(budget, fname, label, caption):
    top, bot = [], []
    for c, lab in CAND:
        r = F[f"meta|{budget}|{c}|excess_newly_missed_per100"]; n = F[f"meta|{budget}|{c}|excess_net_tp_lost_per100"]
        m = F[f"meta|{budget}|{c}|excess_delta_mcc"]
        p = r["sign_p"]; ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        top.append(f"{lab} & {100 * REL[c]:.1f}\\% & {ci(r['pooled'], r['lo'], r['hi'])} & [{num(r['pi_lo'])}, {num(r['pi_hi'])}] & "
                   f"{100 * r['I2']:.0f}\\% & {r['npos']}/{r['k']} ({ps}) \\\\")
        bot.append(f"{lab} & {ci(n['pooled'], n['lo'], n['hi'])} & {ci(m['pooled'], m['lo'], m['hi'], 3)} \\\\")
    h1 = (" & & \\multicolumn{4}{c}{Excess newly missed positives per 100 events} \\\\\n\\cmidrule(l){3-6}\n"
          "Compressed classifier & Storage & Pooled [95\\% CI] & 95\\% PI & $I^2$ & Datasets $>0$ ($p$) \\\\")
    h2 = "Compressed classifier & Excess net TP lost [95\\% CI] & Excess $\\Delta$MCC [95\\% CI] \\\\"
    (OUT / fname).write_text(stacked(caption, label, [("lccccc", h1, top), ("lcc", h2, bot)]))

meta_table(0.1, "tab1_meta.tex", "tab:META",
           "\\textbf{Excess harm of compression beyond ordinary retraining noise, pooled over the 11 datasets.} "
           "Excess: change of the compressed classifier relative to its reference, minus the mean change of three independently retrained "
           "copies of the reference, on the same test patients; 10\\% validation-selected alert budget. Pooled: DerSimonian--Laird random effects "
           "over per-dataset means (Nadeau--Bengio standard errors). PI: prediction interval for a new dataset. Datasets $>0$: datasets with a "
           "positive excess of newly missed positives and one-sided sign-test $p$-value. Net TP lost: newly missed minus recovered positives. "
           "Storage: serialized size relative to the reference (Diabetes 130). Positive excess $\\Delta$MCC values indicate a better MCC than retraining.")
meta_table(0.2, "tabS4_meta_budget20.tex", "tab:META-20",
           "\\textbf{Sensitivity analysis: 20\\% alert budget.} Same quantities as \\autoref{tab:META}, with every classifier alerting the 20\\% "
           "of patients with the highest validation risk scores.")

# ---------- Table 2: per dataset (RF thinning and GBM truncation)
top, bot = [], []
for d in ORDER:
    md = F["meta_data"][d]; rf = S[(S.dataset == d) & (S.budget == 0.1) & (S.candidate == "rf_thinned")].iloc[0]
    gb = S[(S.dataset == d) & (S.budget == 0.1) & (S.candidate == "gbm_truncated")].iloc[0]
    noise = F["noise_rf_per100"][d]; k = int(rf.splits)
    top.append(f"{NAME[d]} & {thou(md['n'])} & {thou(md['events'])} & {100 * md['prevalence']:.1f}\\% & {k} & {noise:.1f} \\\\")
    bot.append(f"{NAME[d]} & "
               f"{ci(rf.excess_newly_missed_per100, rf.excess_newly_missed_per100_ci_low, rf.excess_newly_missed_per100_ci_high, 1)} & "
               f"{int(round(rf.share_splits_excess_missed_gt0 * k))}/{k} & "
               f"{ci(gb.excess_newly_missed_per100, gb.excess_newly_missed_per100_ci_low, gb.excess_newly_missed_per100_ci_high, 1)} & "
               f"{int(round(gb.share_splits_excess_missed_gt0 * k))}/{k} \\\\")
cap2 = ("\\textbf{Excess newly missed positives per dataset} (per 100 positive test patients, 10\\% alert budget). "
        "Records: patients or hospital stays. Splits: patient-grouped train/validation/test partitions. Retraining noise: positives newly missed by a retrained "
        "copy of the 128-tree reference forest. Excess: mean over splits [Nadeau--Bengio 95\\% confidence interval]. $>0$: splits with a positive excess.")
h1 = "Dataset & Records & Events & Prevalence & Splits & Retraining noise \\\\"
h2 = (" & \\multicolumn{2}{c}{RF 128$\\rightarrow$16, thinned} & \\multicolumn{2}{c}{GBM 400$\\rightarrow$100, truncated} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
      "Dataset & Excess [95\\% CI] & $>0$ & Excess [95\\% CI] & $>0$ \\\\")
(OUT / "tab2_per_dataset.tex").write_text(stacked(cap2, "tab:PER-DATASET", [("lrrrrr", h1, top), ("lcccc", h2, bot)]))

# ---------- Table S1: complete metric suite (longtable)
Mb = M[(M.budget == 0.1) & ~M.model.str.contains("retrain")].copy()
tp, fp, fn, tn = [Mb[c].astype(float) for c in ("TP", "FP", "FN", "TN")]
Mb["accuracy"] = (tp + tn) / (tp + fp + fn + tn); Mb["precision"] = tp / (tp + fp).replace(0, np.nan)
Mb["npv"] = tn / (tn + fn).replace(0, np.nan); Mb["f1"] = 2 * tp / (2 * tp + fp + fn).replace(0, np.nan)
med = Mb.groupby(["dataset", "model"]).median(numeric_only=True)
cols = [("mcc", "MCC"), ("f1", "F$_1$"), ("accuracy", "Acc."), ("sensitivity", "Sens."), ("specificity", "Spec."),
        ("precision", "Prec."), ("npv", "NPV"), ("brier", "Brier"), ("auroc", "ROC AUC"), ("auprc", "PR AUC")]
body = []
for d in ORDER:
    body.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textbf{{{NAME[d]}}}}} \\\\*")
    mods = [m for m in MODELS if (d, m) in med.index]; best = max(med.loc[(d, m), "mcc"] for m in mods)
    for m in mods:
        v = [f"{med.loc[(d, m), c]:.3f}" for c, _ in cols]
        if np.isclose(med.loc[(d, m), "mcc"], best): v[0] = f"\\textbf{{{v[0]}}}"
        body.append(f"\\quad {MODELS[m]} & " + " & ".join(v) + " \\\\")
    body.append("\\midrule")
body[-1] = "\\bottomrule"
head = "Classifier & " + " & ".join(h for _, h in cols) + " \\\\"
(OUT / "tabS1_metric_suite.tex").write_text(
    "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{l" + "r" * len(cols) + "}\n"
    "\\caption{\\textbf{Complete set of classification metrics.} Median over patient-grouped splits; every classifier alerts the 10\\% of patients "
    "with the highest validation risk score. Highest MCC per dataset in bold. F$_1$: F$_1$ score. Acc.: accuracy. Sens.: sensitivity. Spec.: specificity. Prec.: precision. NPV: negative predictive value. Brier: Brier score (lower is better).}"
    "\\label{tab:METRIC-SUITE}\\\\\n\\toprule\n" + head + "\n\\midrule\n\\endfirsthead\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n"
    + "\n".join(body) + "\n\\end{longtable}}\n")

# ---------- Table S2: all compressed variants per dataset (longtable)
body = []
for d in ORDER:
    body.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{NAME[d]}}}}} \\\\*")
    for c, lab in CAND:
        r = S[(S.dataset == d) & (S.budget == 0.1) & (S.candidate == c)].iloc[0]; k = int(r.splits)
        body.append(f"\\quad {lab} & {ci(r.excess_newly_missed_per100, r.excess_newly_missed_per100_ci_low, r.excess_newly_missed_per100_ci_high)} & "
                    f"{ci(r.excess_net_tp_lost_per100, r.excess_net_tp_lost_per100_ci_low, r.excess_net_tp_lost_per100_ci_high)} & "
                    f"{ci(r.excess_delta_mcc, r.excess_delta_mcc_ci_low, r.excess_delta_mcc_ci_high, 3)} & {int(round(r.share_splits_excess_missed_gt0 * k))}/{k} \\\\")
    body.append("\\midrule")
body[-1] = "\\bottomrule"
head = "Compressed classifier & \\shortstack{Excess newly missed\\\\per 100 events} & \\shortstack{Excess net TP lost\\\\per 100 events} & Excess $\\Delta$MCC & Splits $>0$ \\\\"
(OUT / "tabS2_all_variants.tex").write_text(
    "{\\scriptsize\n\\begin{longtable}{lcccc}\n\\caption{\\textbf{Excess over retraining noise for every compressed classifier and dataset} "
    "(mean over splits [Nadeau--Bengio 95\\% confidence interval], 10\\% alert budget).}\\label{tab:ALL-VARIANTS}\\\\\n\\toprule\n" + head +
    "\n\\midrule\n\\endfirsthead\n\\toprule\n" + head + "\n\\midrule\n\\endhead\n" + "\n".join(body) + "\n\\end{longtable}}\n")

# ---------- Table S3: dose–response
Db = D[D.budget == 0.1]; sizes = {(k.split("'")[1], int(k.split(", ")[1].rstrip(")"))): v for k, v in F["sizes"].items()}
body = []
for d in [x for x in ORDER if x in set(Db.dataset)]:
    body.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{NAME[d]}}}}} \\\\*")
    for t in (64, 32, 16, 8, 4):
        r = Db[(Db.dataset == d) & (Db.trees == t)].iloc[0]
        body.append(f"\\quad {t} trees ({100 * sizes[(d, t)] / sizes[(d, 128)]:.1f}\\%) & {ci(r.excess_missed_per100, r.ci_low, r.ci_high)} & "
                    f"{ci(r.excess_delta_mcc, r.mcc_ci_low, r.mcc_ci_high, 3)} & {r.expected_missed:.1f} / {r.observed_missed:.1f} & {r.auroc_flip:.3f} \\\\")
    body.append("\\midrule")
body[-1] = "\\bottomrule"
(OUT / "tabS3_dose_response.tex").write_text(
    "\\begin{table}[!htb]\n\\centering\n\\caption{\\textbf{Dose--response of forest thinning} (30 patient-grouped splits, 10\\% alert budget). "
    "Trees kept from the 128-tree reference (storage relative to the reference). Expected / observed: newly missed positives per split predicted by the "
    "tree-sampling model and observed. ROC AUC: discrimination of the predicted per-patient flip probability for the patients whose decision changed.}\n"
    "\\label{tab:DOSE}\n{\\scriptsize\n\\begin{tabular}{lcccc}\n\\toprule\n"
    "Thinned forest & Excess newly missed per 100 events & Excess $\\Delta$MCC & Expected / observed & ROC AUC \\\\\n\\midrule\n"
    + "\n".join(body) + "\n\\end{tabular}}\n\\end{table}\n")
print("tables ->", OUT)
