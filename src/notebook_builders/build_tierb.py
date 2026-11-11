"""Builds CHICCO_TIER_B_DAVIDE.ipynb: Davide's clinical cohorts + random-effects meta-analysis across all datasets."""
import json, sys, os, tempfile

def grab(builder, nout):
    ns = {}; argv = sys.argv; tmp = tempfile.mkdtemp()
    sys.argv = ['x', os.path.join(tmp, 'a.ipynb'), os.path.join(tmp, 'a.py')]
    exec(open(builder).read(), ns); sys.argv = argv
    return [s for k, s in ns['cells'] if k == 'code']

base = grab('/home/claude/nb/build_nb.py', 0)
data = grab('/home/claude/nb/build_data.py', 0)
MODELS, METRICS = base[2], base[3]
RUN_SPLIT = base[4].rsplit('\nfor name in DATASETS:', 1)[0]
AGG = base[5]
DL_FUNCS = data[2].split('status = {}')[0]          # figshare / uci / cran downloaders
READER = data[3].split('inventory = []')[0]         # read_tables()

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Lightweight Classifiers — **Tier B: Davide Chicco's clinical cohorts** + **meta-analysis across all datasets**

**En français.** Même protocole que la réplication (excès de patients oubliés au-delà du hasard du ré-entraînement), appliqué aux cohortes cliniques publiques des articles de Davide. Principe : **peu de tests, beaucoup de datasets**. À la fin, une **méta-analyse à effets aléatoires** combine tous les datasets disponibles (réplication, extension, Tier B).

**Règles fixées avant de voir les résultats**
| Règle | Définition |
|---|---|
| Inclusion | issue binaire ; ≥ 250 patients ; ≥ 50 patients dans la classe minoritaire ; variables tabulaires |
| Cible | issue clinique défavorable (décès, maladie) = classe positive |
| Anti-fuite | exclusion des identifiants, dates, durées de suivi et de toute autre variable d'issue |
| Protocole | 50 partitions, 3 copies ré-entraînées, RF 128→16, GBM 400→100, budgets 10 % et 20 %, IC de Nadeau–Bengio |
| Méta-analyse | DerSimonian–Laird sur l'excès de patients oubliés pour 100 événements, budget 10 % |

⚠️ **Les cibles sont détectées automatiquement quand elles ne sont pas encore fixées.** Le fichier `TARGETS_USED.md` liste, pour chaque dataset, la cible, la classe positive et les variables utilisées ou exclues : il doit être vérifié avant que ces résultats entrent dans l'article.

**Run (Colab):** *Runtime → Run all*. `QUICK_TEST = True` first (minutes), then `False`.
""")

md("### 1 · Configuration et Drive")
code(r"""
import os, sys, io, json, re, time, zipfile, tarfile, hashlib, pickle, platform, warnings, gc, subprocess, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
warnings.filterwarnings('ignore')

# ======================= USER CONFIGURATION =======================
QUICK_TEST    = True
RUN_NAME      = 'EXCESS_TIERB_DAVIDE_20261110_R3'
N_SPLITS      = 50
N_RETRAIN     = 3
ALERT_BUDGETS = [0.10, 0.20]
RF_PARENT, RF_SMALL   = 128, 16
GBM_PARENT, GBM_SMALL = 400, 100
BOOTSTRAP_DRAWS = 500
MIN_PATIENTS, MIN_MINORITY = 250, 50
OTHER_RUNS = ['EXCESS_REPLICATION_20261110_R1', 'EXCESS_EXTENSION_20261110_R2']   # pooled in the meta-analysis
# ==================================================================
if QUICK_TEST:
    N_SPLITS, N_RETRAIN = 2, 2
    RF_PARENT, RF_SMALL, GBM_PARENT, GBM_SMALL, BOOTSTRAP_DRAWS = 32, 8, 100, 25, 100
    RUN_NAME += '_QUICKTEST'; OTHER_RUNS = [r + '_QUICKTEST' for r in OTHER_RUNS]
PRIMARY_BUDGET = ALERT_BUDGETS[0]

try:
    from google.colab import drive
    drive.mount('/content/drive'); PROJECT_HOME = Path('/content/drive/MyDrive/CHICCO_MEGA_RESEARCH'); MODE = 'google_colab_drive'
except ImportError:
    PROJECT_HOME = Path(os.environ['CHICCO_LOCAL_DRIVE_ROOT']); MODE = 'explicit_local_drive_mirror'
DAV = PROJECT_HOME / 'data' / 'raw' / 'davide'; DAV.mkdir(parents=True, exist_ok=True)
OUT = PROJECT_HOME / 'results' / RUN_NAME
for d in [OUT, OUT / 'splits', OUT / 'figures']: d.mkdir(parents=True, exist_ok=True)
for pkg in ['openpyxl', 'xlrd', 'pyreadr', 'ucimlrepo']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=False)
try:
    import lightgbm as lgb; GBM_BACKEND = 'lightgbm ' + lgb.__version__
except ImportError:
    lgb = None; GBM_BACKEND = 'sklearn HistGradientBoosting ' + sklearn.__version__
try:
    from IPython.display import display
except ImportError:
    display = lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)
UA = {'User-Agent': 'Mozilla/5.0 (research notebook; Chicco-Ganmati project)'}
def http(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=dict(UA, **(headers or {})))
    with urllib.request.urlopen(req, timeout=180) as r: return r.read()
print('MODE:', MODE, '| GBM:', GBM_BACKEND, '| results ->', OUT)
""")

md("### 2 · Téléchargement (réutilise les fichiers déjà présents dans `data/raw/davide/`)")
code(r"""
SOURCES = {
  'cardiac_arrest_spain':      ('figshare_id', 4876247),
  'depression_heart_failure':  ('figshare_id', 3916224),
  'lung_cancer_icu':           ('figshare_id', 7713893),
  'clinical_data_14201600':    ('figshare_id', 14201600),
  'sepsis_icu_90day':          ('figshare_id', 28873667),
  'sepsis_sirs':               ('figshare_title', 'C-Reactive Protein and Hemogram Parameters for the Non-Sepsis Systemic Inflammatory Response Syndrome and Sepsis: What Do They Mean?'),
  'ckd_uae':                   ('figshare_title', 'Chronic kidney disease in patients at high risk of cardiovascular disease in the United Arab Emirates: A population-based study'),
  'mesothelioma':              ('uci', 351),
  'heart_failure_chicco':      ('uci', 519),
  'neuroblastoma_shanghai':    ('manual', 'PeerJ 6:e5665 supplemental files, put them in data/raw/davide/neuroblastoma_shanghai/'),
}
""" + "\n" + DL_FUNCS + r"""
fetch_status = {}
for name, (kind, ref) in SOURCES.items():
    if kind == 'manual':
        (DAV / name).mkdir(parents=True, exist_ok=True); fetch_status[name] = 'manual: ' + ref
    else:
        try: fetch_status[name] = fetch(name, kind, ref)
        except Exception as e: fetch_status[name] = f'FAILED: {type(e).__name__}: {e}'
    print(f'{name:26s} -> {fetch_status[name]}')
""")

md("""
### 3 · Définition des cibles (à vérifier) et contrôle anti-fuite
`DATASET_SPECS` fixe la cible quand elle est connue. Sinon, la cible est **proposée automatiquement** parmi les colonnes binaires dont le nom évoque l'issue (décès, mortalité, exitus, classe…). Toutes les décisions sont écrites dans `TARGETS_USED.md`.
""")
code(READER + r"""
DATASET_SPECS = {   # verified 10 Nov 2026 from TARGETS_USED.md of the quick run
  # target column, positive (adverse) value, drop = extra exclusions, keep = override the leak guard, use = explicit predictor list
  'cardiac_arrest_spain': dict(target='Exitus', positive=1, keep=['Functional_status', 'Time_min'],
                               note='death before hospital arrival (Requena-Morales 2017); prior functional status and arrest-to-EMS time are pre-outcome predictors used in the original article'),
  'depression_heart_failure': dict(target='Death (1=Yes, 0=No)', positive=1, drop=['Hospitalized (1=Yes, 0=No)'],
                               note='2-year death (Jani 2016); hospitalization during follow-up is a second outcome: excluded'),
  'lung_cancer_icu': dict(target='Hospital_death', positive=1,
                          use=['sex', 'age', 'Pre_PS', 'PoorPS', 'Histology_coding', 'Stage', 'Distatntmets', 'Surgery', 'WBRT', 'pallRT',
                               'prev_op', 'prev_rtx', 'prev_gks', 'prev_wbrt', 'prev_ctx', 'prev_ccrt', 'line', 'Refractory', 'Txoff', 'NewlyDx',
                               'cancer_status', 'Underlying_CV', 'Underlying_stroke', 'underlying_pulmo', 'pulmocomorbid', 'ILD', 'Underlying_DM',
                               'Underlying_etc', 'other_malignancy', 'smk_hx', 'Pneumonia', 'Sepsis', 'respfailure', 'hemodynamic', 'AKI',
                               'neurology', 'cardiovasc', 'prevCPR', 'Planned', 'preICUDNR', 'admission_via', 'PrevICUadm', 'Intensivist', 'post2011',
                               'neutropenia', 'ANC', 'WBC', 'anemia', 'Hb', 'thrombocytopenia', 'PLT', 'hypoalbuminemia', 'Albumin', 'BUN', 'Cr',
                               'highCr', 'Na', 'K', 'PT_percent', 'aPTT', 'procalcitonin', 'PaO2FiO2', 'SAPS_II', 'APACH_II', 'SOFA_D1'],
                          note='hospital death (Song 2019, primary outcome); only variables known at ICU admission (no ICU treatments, no post-ICU or discharge variables)'),
  'sepsis_sirs':     dict(target='Mortality', positive=1, note='ICU mortality (Gucyetmez & Atalan 2016)'),
  'ckd_uae':         dict(target='EventCKD35', positive=1, drop=['StudyID'], note='CKD stage 3-5 event (Al-Shamsi 2018); StudyID is an identifier'),
  'heart_failure_chicco': dict(target='DEATH_EVENT', positive=1, drop=['time'], note='death (Chicco & Jurman 2020); time = follow-up length (outcome-related): excluded'),
  'mesothelioma':    dict(target=None, positive=None, drop=['diagnosis method'], note='diagnosis method can reveal the class: excluded'),
}
TARGET_WORDS = ['death', 'dead', 'died', 'deceas', 'mortal', 'exitus', 'fallec', 'muert', 'surviv', 'outcome', 'class', 'status', 'event', 'sepsis', 'ckd', 'diagnos']
LEAK = re.compile(r'(time|days?\b|month|follow|^fu_|los\b|length|stay|death|dead|died|deceas|mortal|surviv|exitus|outcome|discharge|readmi|event|censor|date|_os\b|^os\b|pfs)', re.I)
IDLIKE = re.compile(r'(^id$|_id$|^id_|studyid|serial|patient|record|^n$|^no\.?$|^nr|number|code$)', re.I)
ADVERSE = re.compile(r'(dead|death|died|deceas|exitus|fallec|muert|^yes$|^y$|^s[ií]$|sepsis|ckd|mesothel|positive|^true$)', re.I)
ALIVE_NAME = re.compile(r'(alive|surviv)', re.I)

def choose_table(name, tables, spec):
    items = [(k, v) for k, v in tables.items() if isinstance(v, pd.DataFrame) and v.shape[0] >= 20]
    if spec.get('table'): items = [(k, v) for k, v in items if spec['table'] in k]
    if not items: return None, None
    return max(items, key=lambda kv: kv[1].shape[0] * kv[1].shape[1])

def auto_target(df):
    best = None
    for rank, w in enumerate(TARGET_WORDS):
        for c in df.columns:
            if w in str(c).lower() and df[c].nunique(dropna=True) == 2:
                return c
    return best

def positive_value(col, values):
    vals = list(values)
    num = pd.to_numeric(pd.Series(vals), errors='coerce')
    if num.notna().all() and set(num.astype(float)) == {0.0, 1.0}:
        return vals[list(num.astype(float)).index(0.0 if ALIVE_NAME.search(str(col)) else 1.0)], 'numeric 0/1 (1 = event unless the name says alive/survival)'
    hits = [v for v in vals if ADVERSE.search(str(v).strip())]
    if len(hits) == 1: return hits[0], 'value name indicates the adverse outcome'
    return None, 'UNCLEAR'

def build_dataset(name):
    spec = DATASET_SPECS.get(name, {})
    files = [p for p in (DAV / name).rglob('*') if p.is_file() and not p.name.startswith('figshare_')]
    tables = {}
    for p in files: tables.update(read_tables(p))
    key, df = choose_table(name, tables, spec)
    if df is None: return None, dict(dataset=name, status='no readable table')
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    lower = {c.lower(): c for c in df.columns}
    target = spec.get('target'); target = lower.get(str(target).lower(), target) if target else auto_target(df)
    if target is None or target not in df.columns:
        return None, dict(dataset=name, table=key, status='no target found', columns=list(df.columns))
    y_raw = df[target]; ok = y_raw.notna(); vals = y_raw[ok].unique()
    if len(vals) != 2: return None, dict(dataset=name, table=key, target=target, status=f'target has {len(vals)} values')
    if spec.get('positive') is not None: pos, why = spec['positive'], 'fixed in DATASET_SPECS'
    else: pos, why = positive_value(target, vals)
    if pos is None:
        counts = y_raw[ok].value_counts(); pos, why = counts.index[-1], 'UNCLEAR -> minority class used (CHECK)'
    num = pd.to_numeric(y_raw[ok], errors='coerce'); pnum = pd.to_numeric(pd.Series([pos]), errors='coerce').iloc[0]
    if num.notna().all() and pd.notna(pnum):
        y = (num.astype(float) == float(pnum)).astype(int).to_numpy()
    else:
        y = (y_raw[ok].astype(str).str.strip() == str(pos).strip()).astype(int).to_numpy()
    X = df.loc[ok].drop(columns=[target]).reset_index(drop=True)
    dropped = {}
    if spec.get('use'):
        missing = [c for c in spec['use'] if c not in X.columns]
        if missing: print(f'  {name}: predictors not found (skipped): {missing}')
        dropped.update({c: 'not in pre-declared predictor list' for c in X.columns if c not in spec['use']})
        X = X[[c for c in spec['use'] if c in X.columns]]
    keep = [k.lower() for k in spec.get('keep', [])]
    for c in list(X.columns):
        reason = None
        if c.lower() in keep and X[c].nunique(dropna=True) > 1: continue
        if c in spec.get('drop', []) or c.lower() in [d.lower() for d in spec.get('drop', [])]: reason = 'specified'
        elif LEAK.search(c): reason = 'outcome/time-related name'
        elif IDLIKE.search(c) and X[c].nunique() > 0.9 * len(X): reason = 'identifier'
        elif pd.api.types.is_datetime64_any_dtype(X[c]): reason = 'date'
        elif X[c].nunique(dropna=True) <= 1: reason = 'constant'
        elif not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.5 * len(X): reason = 'free text / identifier'
        if reason: dropped[c] = reason; X = X.drop(columns=c)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            num = pd.to_numeric(X[c], errors='coerce')
            if num.notna().mean() > 0.95 and X[c].notna().mean() > 0: X[c] = num
            else: X[c] = X[c].astype('object').where(X[c].notna(), 'Missing').astype(str)
    n, ev = len(y), int(y.sum()); minority = min(ev, n - ev)
    status = 'included' if (n >= MIN_PATIENTS and minority >= MIN_MINORITY and X.shape[1] >= 2) else \
             f'excluded (n={n}, minority={minority}, features={X.shape[1]})'
    info = dict(dataset=name, table=key, target=target, positive=str(pos), positive_rule=why, n=n, events=ev,
                prevalence=round(ev / n, 4), features_used=list(X.columns), features_dropped=dropped, status=status,
                note=spec.get('note', ''))
    groups = np.array([f'{name}_{i}' for i in range(n)])
    return ((X, y, groups, dict(dataset=name, task=f'{target} = {pos}', n=n, events=ev, prevalence=round(ev / n, 4),
                                n_groups=n, group='one row per patient')) if status == 'included' else None), info

DATA, REPORT = {}, []
for name in SOURCES:
    try:
        d, info = build_dataset(name)
    except Exception as e:
        d, info = None, dict(dataset=name, status=f'error: {type(e).__name__}: {e}')
    REPORT.append(info)
    if d is not None: DATA[name] = d
    print(f"{name:26s} {info.get('status')}  target={info.get('target')}  positive={info.get('positive')}  "
          f"n={info.get('n')} events={info.get('events')}  rule={info.get('positive_rule')}")
lines = ['# Targets used (VERIFY before using results in the paper)\n']
for r in REPORT:
    lines.append(f"## {r['dataset']} — {r.get('status')}\n" + '\n'.join(f'- {k}: {v}' for k, v in r.items() if k not in ('dataset', 'status')) + '\n')
(OUT / 'TARGETS_USED.md').write_text('\n'.join(lines))
DATASETS = list(DATA)
CONFIG = dict(run_name=RUN_NAME, datasets=DATASETS, n_splits=N_SPLITS, n_retrain=N_RETRAIN, alert_budgets=ALERT_BUDGETS,
              rf_parent=RF_PARENT, rf_small=RF_SMALL, gbm_parent=GBM_PARENT, gbm_small=GBM_SMALL, bootstrap=BOOTSTRAP_DRAWS,
              gbm_backend=GBM_BACKEND.split()[0], targets={r['dataset']: (r.get('target'), r.get('positive')) for r in REPORT if r['dataset'] in DATA},
              protocol_version='2026-11-10_tierB_v1')
SIGNATURE = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=str).encode()).hexdigest()[:16]
(OUT / 'PROTOCOL.json').write_text(json.dumps(CONFIG, indent=2, default=str))
print('\nINCLUDED:', DATASETS, '| signature', SIGNATURE, '\nCheck:', OUT / 'TARGETS_USED.md')
""")

md("### 4 · Modèles et mesures (identiques à la réplication)")
code("from sklearn.model_selection import GroupShuffleSplit\n" + MODELS + "\nSPLITTERS = {}")
code(METRICS)

md("### 5 · Analyse par dataset (checkpoint Drive après chaque partition)")
code(RUN_SPLIT.replace('tr, va, te = grouped_split(y, groups, seed)', 'tr, va, te = SPLITTERS.get(name, grouped_split)(y, groups, seed)') + r"""

for name in DATASETS:
    for s in range(N_SPLITS):
        try:
            status = run_split(name, s)
        except AssertionError as e:
            status = f'skipped ({e})'
        print(f'{name} split {s+1:02d}/{N_SPLITS}: {status}', flush=True)
print('TIER B COMPLETE ->', OUT / 'splits')
""")

md("### 6 · Résumé Tier B")
code(AGG)

md("""
### 7 · Méta-analyse à effets aléatoires sur **tous** les datasets disponibles
Pour chaque dataset : excès moyen et erreur-type déduite de l'intervalle de Nadeau–Bengio. Combinaison par DerSimonian–Laird, avec intervalle de prédiction (où tomberait l'effet sur un nouveau dataset).
""")
code(r"""
import matplotlib.pyplot as plt
frames = []
for run in OTHER_RUNS:
    f = PROJECT_HOME / 'results' / run / 'SUMMARY_excess_over_retraining.csv'
    if f.exists(): frames.append(pd.read_csv(f).assign(source=run))
    else: print('not found (skipped):', f)
frames.append(pd.read_csv(OUT / 'SUMMARY_excess_over_retraining.csv').assign(source=RUN_NAME))
ALLS = pd.concat(frames).drop_duplicates(['dataset', 'budget', 'family', 'candidate'], keep='last')
ALLS.to_csv(OUT / 'SUMMARY_all_datasets.csv', index=False)

def dersimonian_laird(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float); k = len(est)
    w = 1 / se**2; fe = np.sum(w * est) / np.sum(w); Q = np.sum(w * (est - fe)**2)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w))) if k > 1 else 0.0
    wr = 1 / (se**2 + tau2); mu = np.sum(wr * est) / np.sum(wr); se_mu = np.sqrt(1 / np.sum(wr))
    t = stats.t.ppf(0.975, max(1, k - 2)); pi = (mu - t * np.sqrt(tau2 + se_mu**2), mu + t * np.sqrt(tau2 + se_mu**2))
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(k=k, pooled=mu, ci_low=mu - 1.96 * se_mu, ci_high=mu + 1.96 * se_mu, pi_low=pi[0], pi_high=pi[1], tau2=tau2, I2=i2)

meta_rows = []; fig_data = {}
for cand in ['rf_thinned', 'rf_direct_small', 'gbm_truncated', 'gbm_direct_small']:
    e = ALLS[(ALLS.budget == PRIMARY_BUDGET) & (ALLS.candidate == cand)].dropna(subset=['excess_newly_missed_per100_ci_low'])
    if len(e) < 2: continue
    tq = stats.t.ppf(0.975, e.splits - 1)
    se = (e.excess_newly_missed_per100_ci_high - e.excess_newly_missed_per100_ci_low) / (2 * tq)
    r = dersimonian_laird(e.excess_newly_missed_per100, se); r['candidate'] = cand; meta_rows.append(r)
    fig_data[cand] = (e.assign(se=se), r)
META = pd.DataFrame(meta_rows); META.to_csv(OUT / 'META_ANALYSIS_excess.csv', index=False)
print('RANDOM-EFFECTS META-ANALYSIS · excess newly missed per 100 events · budget', PRIMARY_BUDGET)
display(META[['candidate', 'k', 'pooled', 'ci_low', 'ci_high', 'pi_low', 'pi_high', 'I2']].round(3))

if 'rf_thinned' in fig_data:
    e, r = fig_data['rf_thinned']; e = e.sort_values('excess_newly_missed_per100')
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(e) + 1.6), dpi=150); y = np.arange(len(e))[::-1] + 1
    ax.errorbar(e.excess_newly_missed_per100, y, xerr=[e.excess_newly_missed_per100 - e.excess_newly_missed_per100_ci_low,
                e.excess_newly_missed_per100_ci_high - e.excess_newly_missed_per100], fmt='s', color='#178F8A', ms=4, capsize=2)
    ax.errorbar(r['pooled'], 0, xerr=[[r['pooled'] - r['ci_low']], [r['ci_high'] - r['pooled']]], fmt='D', color='#17324D', ms=7, capsize=3)
    ax.plot([r['pi_low'], r['pi_high']], [0, 0], color='#17324D', lw=0.8, ls=':')
    ax.axvline(0, color='#5E7486', ls='--', lw=.8)
    ax.set_yticks(list(y) + [0]); ax.set_yticklabels(list(e.dataset) + ['Pooled (random effects)'])
    ax.set_xlabel('Excess newly missed positives per 100 events, RF 128→16 thinning (> 0: worse than retraining)')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(OUT / 'figures' / f'fig_meta_rf_thinned.{ext}', dpi=200, facecolor='white')
    plt.show(); plt.close(fig)

share = PROJECT_HOME / 'results' / (RUN_NAME + '_SHAREABLE.zip')
with zipfile.ZipFile(share, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ['PROTOCOL.json', 'TARGETS_USED.md', 'SUMMARY_excess_over_retraining.csv', 'SUMMARY_all_datasets.csv', 'META_ANALYSIS_excess.csv',
              'comparisons_per_split.csv', 'metrics_per_split.csv', 'mechanism_per_split.csv', 'model_sizes_first_split.csv']:
        if (OUT / f).exists(): zf.write(OUT / f, arcname=f)
    for f in sorted((OUT / 'figures').glob('*')): zf.write(f, arcname='figures/' + f.name)
(OUT / 'MANIFEST.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
        signature=SIGNATURE, data={n: DATA[n][3] for n in DATA}, fetch_status={k: (v if isinstance(v, str) else list(v)) for k, v in fetch_status.items()}), indent=2, default=str))
with zipfile.ZipFile(share, 'a') as zf: zf.write(OUT / 'MANIFEST.json', arcname='MANIFEST.json')
print('Shareable ZIP (no patient-level data):', share)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "cells": []}
for i, (k, s) in enumerate(cells):
    c = {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True), "id": f"t{i:02d}"}
    if k == "code": c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
open(sys.argv[2], "w").write("\n\n".join(s for k, s in cells if k == "code"))
print("cells", len(cells))
