"""Builds CHICCO_EXCESS_REPLICATION.ipynb (code + markdown cells) and a flat .py for local testing."""
import json, sys

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Lightweight Classifiers and Their Failures
## Replication notebook · *Is compression harm larger than ordinary retraining noise?*

**Abdelilah Ganmati — collaboration with Dr. Davide Chicco · prepared 10 November 2026**

**En français.** Ce notebook teste l'idée centrale découverte le 8 novembre : la compression d'un modèle fait-elle manquer **plus de patients positifs** que le simple **ré-entraînement** du même modèle ? Il répète l'analyse sur **30 partitions groupées par patient**, avec **plusieurs graines de ré-entraînement**, pour **deux familles** (random forest et gradient boosting), sur **deux tâches EHR indépendantes** : PhysioNet 2012 (mortalité hospitalière) et UCI Diabetes 130 (réadmission < 30 jours). Tout est sauvegardé sur Google Drive après **chaque partition** : si Colab se déconnecte, relancez *Run all* et le calcul reprend où il s'était arrêté.

### Pre-specified analysis (written before running)
| Item | Definition |
|---|---|
| **Excess newly missed** | (positives alerted by the parent but missed by the compressed model) − mean over retrained copies of the same quantity |
| **Primary endpoint** | Excess newly missed positives **per 100 test events**, RF 128 → 16 thinning, 10 % validation-selected alert budget |
| **Primary success criterion** | Nadeau–Bengio corrected 95 % CI over the 30 splits excludes 0 **on both datasets** |
| **Secondary** | Excess net true positives lost; excess ΔMCC; GBM truncation; direct small models; 20 % alert budget |
| **Mechanism test** | Thinning = sampling k of M trees ⇒ per-patient flip probability predicted from the parent's tree spread; compare expected vs observed flips |
| **Negative controls** | Retrained-vs-parent changes (noise floor); logistic regression as efficient-by-design comparator |

**How to run (Colab):** *File → Upload notebook* → *Runtime → Run all* → authorise Google Drive.
Set `QUICK_TEST = True` first (≈10 min) to check everything works, then `False` for the full run (several hours; resumable).

> Not a clinical device. Repeated splits re-use one cohort per dataset; they are not independent external validations.
""")

md("### 1 · Configuration et Google Drive")
code(r"""
import os, sys, io, json, time, zipfile, hashlib, pickle, platform, warnings, gc, urllib.request
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
warnings.filterwarnings('ignore', category=UserWarning)

# ======================= USER CONFIGURATION =======================
QUICK_TEST    = True          # True: 2 splits, small models (pipeline check). False: full study.
RUN_NAME      = 'EXCESS_REPLICATION_20261110_R1'
DATASETS      = ['physionet_2012', 'diabetes_130']   # PhysioNet first: smaller, gives results sooner
N_SPLITS      = 30            # patient-grouped repeated splits per dataset
N_RETRAIN     = 3             # independent retrained copies of each parent (noise floor)
ALERT_BUDGETS = [0.10, 0.20]  # fraction alerted, chosen on VALIDATION only; 0.10 is primary
RF_PARENT, RF_SMALL   = 128, 16
GBM_PARENT, GBM_SMALL = 400, 100   # boosting rounds: parent vs truncated / small
BOOTSTRAP_DRAWS = 500         # patient-cluster bootstrap inside each split
P2012_SETS    = ['a', 'b', 'c']    # PhysioNet 2012 sets with public outcomes; falls back to those available
# ==================================================================

if QUICK_TEST:
    N_SPLITS, N_RETRAIN = 2, 2
    RF_PARENT, RF_SMALL, GBM_PARENT, GBM_SMALL, BOOTSTRAP_DRAWS = 32, 8, 100, 25, 100
    RUN_NAME += '_QUICKTEST'
PRIMARY_BUDGET = ALERT_BUDGETS[0]

try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_HOME = Path('/content/drive/MyDrive/CHICCO_MEGA_RESEARCH'); MODE = 'google_colab_drive'
except ImportError:
    local = os.environ.get('CHICCO_LOCAL_DRIVE_ROOT')
    if not local:
        raise RuntimeError('Open this notebook in Google Colab (Drive persistence). '
                           'For a local mirror, set CHICCO_LOCAL_DRIVE_ROOT.')
    PROJECT_HOME = Path(local).expanduser().resolve(); MODE = 'explicit_local_drive_mirror'

RAW  = PROJECT_HOME / 'data' / 'raw'
PREP = PROJECT_HOME / 'data' / 'prepared'
OUT  = PROJECT_HOME / 'results' / RUN_NAME
for d in [RAW, PREP, OUT, OUT/'splits', OUT/'figures']:
    d.mkdir(parents=True, exist_ok=True)

try:
    import lightgbm as lgb
    GBM_BACKEND = 'lightgbm ' + lgb.__version__
except ImportError:
    lgb = None
    GBM_BACKEND = 'sklearn HistGradientBoosting ' + sklearn.__version__

CONFIG = dict(run_name=RUN_NAME, quick_test=QUICK_TEST, datasets=DATASETS, n_splits=N_SPLITS,
              n_retrain=N_RETRAIN, alert_budgets=ALERT_BUDGETS, rf_parent=RF_PARENT, rf_small=RF_SMALL,
              gbm_parent=GBM_PARENT, gbm_small=GBM_SMALL, bootstrap=BOOTSTRAP_DRAWS,
              p2012_sets=P2012_SETS, gbm_backend=GBM_BACKEND.split()[0],
              split='patient-grouped train 65% / validation 15% / test 20%',
              protocol_version='2026-11-10_excess_v1')
lock = OUT / 'PROTOCOL_LOCK.json'
if lock.exists() and json.loads(lock.read_text()) != CONFIG:
    raise RuntimeError(f'{OUT} already holds a run with another protocol. Change RUN_NAME.')
lock.write_text(json.dumps(CONFIG, indent=2))
SIGNATURE = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]

try:
    from IPython.display import display
except ImportError:
    display = lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)

print('MODE:', MODE, '| GBM backend:', GBM_BACKEND)
print('RESULTS ->', OUT)
print(f'{len(DATASETS)} datasets × {N_SPLITS} splits × (RF + GBM + LR), {N_RETRAIN} retrains per parent')
""")

md("""
### 2 · Données (mises en cache sur Drive)
**Diabetes 130** : mêmes 18 variables et mêmes exclusions (sorties décès/hospice) que le notebook du 8 novembre ; groupe = `patient_nbr`.
**PhysioNet 2012** : 48 premières heures en réanimation, dernière valeur et moyenne de chaque variable ; un dossier par patient ; `RecordID` n'est **jamais** utilisé comme prédicteur (correction par rapport à la version précédente).
""")
code(r"""
def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 100:
        return path
    tmp = path.with_suffix(path.suffix + '.part')
    print('DOWNLOAD', url, flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research notebook)'})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk)
    if tmp.stat().st_size < 100: raise IOError(f'Incomplete download: {url}')
    tmp.replace(path)
    return path

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

D130_NUMERIC = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
                'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
D130_CATEGORICAL = ['race','gender','age','admission_type_id','admission_source_id',
                    'A1Cresult','max_glu_serum','insulin','diabetesMed','change']

def load_diabetes_130():
    csv = RAW / 'diabetic_data.csv'
    if not csv.exists():
        z = download('https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip',
                     RAW / 'uci_296_source.zip')
        with zipfile.ZipFile(z) as zz:
            name = [n for n in zz.namelist() if Path(n).name == 'diabetic_data.csv'][0]
            csv.write_bytes(zz.read(name))
    df = pd.read_csv(csv, na_values=['?', 'Unknown/Invalid'], low_memory=False, dtype={'patient_nbr': 'string'})
    disp = pd.to_numeric(df['discharge_disposition_id'], errors='coerce')
    df = df.loc[(~disp.isin([11, 13, 14, 19, 20, 21])) & df.patient_nbr.notna()
                & df.readmitted.isin(['<30', '>30', 'NO'])].copy()
    X = df[D130_NUMERIC + D130_CATEGORICAL].copy()
    for c in D130_NUMERIC: X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in D130_CATEGORICAL: X[c] = X[c].astype('object').where(X[c].notna(), 'Missing').astype(str)
    y = (df.readmitted == '<30').astype(int).to_numpy()
    groups = df.patient_nbr.astype(str).to_numpy()
    meta = dict(dataset='UCI Diabetes 130-US Hospitals (296)', task='30-day readmission at discharge',
                doi='10.24432/C5230J', data_sha256=sha256(csv), group='patient_nbr')
    return X.reset_index(drop=True), y, groups, meta

P2012_BASE = 'https://physionet.org/files/challenge-2012/1.0.0/'

def _parse_p2012_set(s):
    cache = PREP / f'physionet2012_set_{s}_48h_v2.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    z = download(P2012_BASE + f'set-{s}.zip', RAW / f'physionet2012_set_{s}.zip')
    o = download(P2012_BASE + f'Outcomes-{s}.txt', RAW / f'physionet2012_outcomes_{s}.txt')
    outcome = pd.read_csv(o).set_index('RecordID')['In-hospital_death']
    rows = []
    with zipfile.ZipFile(z) as zz:
        names = [n for n in zz.namelist() if n.endswith('.txt') and Path(n).stem.isdigit()]
        for i, name in enumerate(names):
            rid = int(Path(name).stem)
            if rid not in outcome.index: continue
            f = pd.read_csv(io.BytesIO(zz.read(name)))
            f['Value'] = pd.to_numeric(f['Value'], errors='coerce').replace(-1, np.nan)
            r = {'record_id': rid, 'set': s}
            for p, g in f.groupby('Parameter'):
                if p == 'RecordID': continue                     # identifier, never a predictor
                v = g['Value'].dropna()
                if len(v):
                    r[f'{p}__last'] = float(v.iloc[-1]); r[f'{p}__mean'] = float(v.mean())
            rows.append(r)
            if (i + 1) % 2000 == 0: print(f'  set {s}: parsed {i+1}/{len(names)} records', flush=True)
    d = pd.DataFrame(rows).set_index('record_id').join(outcome.rename('target'), how='inner')
    d.to_pickle(cache)
    return d

def load_physionet_2012():
    parts, used = [], []
    for s in P2012_SETS:
        try:
            parts.append(_parse_p2012_set(s)); used.append(s)
        except Exception as e:
            print(f'WARNING: PhysioNet 2012 set {s} unavailable ({type(e).__name__}: {e}); continuing without it')
    if not parts: raise RuntimeError('No PhysioNet 2012 set could be loaded.')
    d = pd.concat(parts, axis=0)
    y = d.pop('target').astype(int).to_numpy()
    groups = np.array([f'{s}_{r}' for s, r in zip(d['set'], d.index)])
    X = d.drop(columns=['set'])
    for c in ['ICUType__mean', 'Gender__mean']:          # constants per stay; keep the __last copy only
        if c in X: X = X.drop(columns=c)
    if 'ICUType__last' in X: X['ICUType__last'] = X['ICUType__last'].astype('Int64').astype(str)
    meta = dict(dataset='PhysioNet/CinC Challenge 2012', task='in-hospital mortality from first 48 h in ICU',
                sets_used=used, n_features=X.shape[1], group='one ICU stay per record')
    return X.reset_index(drop=True), y, groups, meta

LOADERS = {'diabetes_130': load_diabetes_130, 'physionet_2012': load_physionet_2012}
DATA = {}
for name in DATASETS:
    X, y, g, meta = LOADERS[name]()
    meta.update(n=len(y), events=int(y.sum()), prevalence=round(float(y.mean()), 4), n_groups=len(np.unique(g)))
    DATA[name] = (X, y, g, meta)
    (OUT / f'DATA_{name}.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'{name}: {meta["n"]} rows, {meta["n_groups"]} patients, {meta["events"]} events '
          f'({100*meta["prevalence"]:.1f}%), {X.shape[1]} raw features')
""")

md("""
### 3 · Modèles et partitions
Le préprocesseur est appris **une seule fois par partition, sur l'entraînement**, puis partagé par tous les modèles, pour que les comparaisons ne diffèrent que par le modèle.
- **RF** : parent 128 arbres ; *thinned* = les 16 premiers arbres **du même parent** (aucun ré-entraînement) ; *direct small* = 16 arbres entraînés directement ; *retrain k* = 128 arbres, autre graine.
- **GBM** : parent 400 itérations ; *truncated* = les 100 premières itérations du même parent ; *direct small* = 100 itérations, arbres plus petits, autre graine ; *retrain k* = même configuration, autre graine (sous-échantillonnage des lignes et colonnes).
- **LR** : régression logistique, comparateur léger par construction.
""")
code(r"""
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def grouped_split(y, groups, seed):
    idx = np.arange(len(y))
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(idx, y, groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=seed + 1).split(dev, y[dev], groups[dev]))
    train, val = dev[a], dev[b]
    for p, q in [(train, val), (train, test), (val, test)]:
        assert set(groups[p]).isdisjoint(set(groups[q])), 'PATIENT LEAKAGE'
    for part in (train, val, test):
        assert len(np.unique(y[part])) == 2, 'a partition has a single class'
    return train, val, test

def make_preprocessor(X):
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [c for c in X.columns if c not in num]
    tr = []
    if num:
        tr.append(('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                                    ('sc', StandardScaler())]), num))
    if cat:
        tr.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                                    ('oh', OneHotEncoder(handle_unknown='ignore', min_frequency=0.002,
                                                         sparse_output=False))]), cat))
    return ColumnTransformer(tr, sparse_threshold=0)

def fit_rf(Xt, yt, n_trees, seed):
    return RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2, max_features='sqrt',
                                  n_jobs=-1, random_state=seed).fit(Xt, yt)

def tree_probabilities(rf, X):
    # One row per tree; the forest score is exactly the mean over rows.
    return np.stack([t.predict_proba(X)[:, 1] for t in rf.estimators_]).astype(np.float32)

def fit_gbm(Xt, yt, n_iter, seed, small=False):
    leaves = 7 if small else 31
    if lgb is not None:
        m = lgb.LGBMClassifier(n_estimators=n_iter, learning_rate=0.05, num_leaves=leaves,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               min_child_samples=50, random_state=seed, n_jobs=-1, verbose=-1)
    else:
        m = HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.05, max_leaf_nodes=leaves,
                                           min_samples_leaf=50, max_features=0.8, early_stopping=False,
                                           random_state=seed)
    return m.fit(Xt, yt)

def gbm_scores(m, X, n_iter=None):
    if lgb is not None:
        return m.predict_proba(X, num_iteration=n_iter)[:, 1]
    if n_iter is None:
        return m.predict_proba(X)[:, 1]
    for i, p in enumerate(m.staged_predict_proba(X), start=1):
        if i == n_iter: return p[:, 1]
    raise ValueError('n_iter larger than fitted iterations')

def fit_lr(Xt, yt):
    return LogisticRegression(solver='liblinear', max_iter=2000).fit(Xt, yt)

def serialized_bytes(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
""")

md("""
### 4 · Mesures : décisions, excès au-delà du ré-entraînement, mécanisme
Pour chaque modèle, le seuil est le quantile `1 − budget` des scores de **validation** ; le test n'est jamais utilisé pour régler quoi que ce soit.

**Prédiction du mécanisme (RF).** Garder k arbres sur M revient à tirer k probabilités d'arbres sans remise. Pour chaque patient, le score réduit a donc pour moyenne le score du parent et pour variance σ²/k · (M−k)/(M−1). La probabilité qu'il passe sous le seuil du modèle réduit est Φ((t − μ)/sd). On compare le nombre de bascules **attendu** au nombre **observé**.
""")
code(r"""
from scipy.stats import norm

def counts(y, alert):
    y = y.astype(bool); a = alert.astype(bool)
    return dict(TP=int((y & a).sum()), FP=int((~y & a).sum()), FN=int((y & ~a).sum()), TN=int((~y & ~a).sum()))

def mcc_from(tp, tn, fp, fn):
    tp, tn, fp, fn = [np.asarray(v, float) for v in (tp, tn, fp, fn)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)

def calibration_slope(y, p):
    pc = np.clip(p, 1e-3, 1 - 1e-3)   # forests give many exact 0/1 scores; slope is indicative only for RF
    z = np.log(pc / (1 - pc)).reshape(-1, 1)
    try:
        return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])
    except Exception:
        return float('nan')

def model_metrics(y, p, alert):
    c = counts(y, alert)
    sens = c['TP'] / max(1, c['TP'] + c['FN']); spec = c['TN'] / max(1, c['TN'] + c['FP'])
    return dict(**c, mcc=float(mcc_from(c['TP'], c['TN'], c['FP'], c['FN'])), sensitivity=sens, specificity=spec,
                alert_rate=float(alert.mean()), auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)),
                calibration_slope=calibration_slope(y, p), observed_over_expected=float(y.mean() / max(1e-9, p.mean())))

def change_vectors(y, ref, cand):
    y = y.astype(bool); r = ref.astype(bool); c = cand.astype(bool)
    return dict(newly_missed=(y & r & ~c), recovered=(y & ~r & c),
                new_false_alarms=(~y & ~r & c), removed_false_alarms=(~y & r & ~c))

def excess_comparison(y, groups_inv, n_groups, boot_idx, dec, ref, cand, retrains):
    # Candidate change vs the parent, minus the average change produced by retraining alone.
    cv = change_vectors(y, dec[ref], dec[cand])
    rv = [change_vectors(y, dec[ref], dec[r]) for r in retrains]
    out = {}
    for k in cv:
        out[f'{k}_candidate'] = int(cv[k].sum())
        out[f'{k}_retrain_mean'] = float(np.mean([v[k].sum() for v in rv]))
        out[f'excess_{k}'] = out[f'{k}_candidate'] - out[f'{k}_retrain_mean']
    net_c = cv['newly_missed'].astype(float) - cv['recovered']
    net_r = np.mean([v['newly_missed'].astype(float) - v['recovered'] for v in rv], axis=0)
    per_patient = {'excess_newly_missed': cv['newly_missed'] - np.mean([v['newly_missed'] for v in rv], axis=0),
                   'excess_net_tp_lost': net_c - net_r}
    out['excess_net_tp_lost'] = float(per_patient['excess_net_tp_lost'].sum())
    for k, vec in per_patient.items():
        s = np.bincount(groups_inv, weights=vec, minlength=n_groups)
        b = s[boot_idx].sum(axis=1)
        out[f'{k}_boot_low'], out[f'{k}_boot_high'] = [float(v) for v in np.percentile(b, [2.5, 97.5])]
    mc = lambda a: float(mcc_from(**{k.lower(): v for k, v in counts(y, a).items()}))
    out['mcc_candidate'] = mc(dec[cand]); out['mcc_parent'] = mc(dec[ref])
    out['mcc_retrain_mean'] = float(np.mean([mc(dec[r]) for r in retrains]))
    out['excess_delta_mcc'] = out['mcc_candidate'] - out['mcc_retrain_mean']
    return out

def thinning_mechanism(y, tree_p, thr_parent, thr_thin, k, alert_parent, alert_thin):
    M = tree_p.shape[0]
    mu = tree_p.mean(0).astype(float); var = tree_p.var(0).astype(float)
    sd = np.sqrt(var / k * (M - k) / max(1, M - 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        p_alert = np.where(sd > 0, 1 - norm.cdf((thr_thin - mu) / np.where(sd > 0, sd, 1)), (mu >= thr_thin).astype(float))
    p_change = np.where(alert_parent, 1 - p_alert, p_alert)
    obs_change = alert_parent != alert_thin
    pos_alerted = (y == 1) & alert_parent
    margin = mu[pos_alerted] - thr_parent
    q = np.quantile(margin, [0.25, 0.5, 0.75]) if pos_alerted.sum() >= 8 else np.array([0, 0, 0])
    quart = np.digitize(margin, q)
    lost = ~alert_thin[pos_alerted]
    return dict(expected_newly_missed=float((1 - p_alert)[pos_alerted].sum()),
                observed_newly_missed=int(lost.sum()),
                expected_changes=float(p_change.sum()), observed_changes=int(obs_change.sum()),
                auroc_predicting_changes=float(roc_auc_score(obs_change, p_change)) if 0 < obs_change.sum() < len(obs_change) else float('nan'),
                margin_quartile_n=[int((quart == i).sum()) for i in range(4)],
                margin_quartile_lost_thinned=[int(lost[quart == i].sum()) for i in range(4)],
                _quart=quart, _pos_alerted=pos_alerted)
""")

md("""
### 5 · Boucle principale (checkpoint Drive après chaque partition)
Relancer la cellule reprend automatiquement : les partitions déjà sauvegardées avec la même signature de protocole sont sautées. La durée est affichée après chaque partition.
""")
code(r"""
def run_split(name, s):
    path = OUT / 'splits' / f'{name}_split{s:02d}.json'
    if path.exists():
        try:
            if json.loads(path.read_text()).get('signature') == SIGNATURE:
                return 'resumed'
        except Exception:
            pass
    t0 = time.time()
    X, y, groups, meta = DATA[name]
    seed = 1000 + 37 * s
    tr, va, te = grouped_split(y, groups, seed)
    prep = make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
    Xtr, Xva, Xte = [prep.transform(X.iloc[i]).astype(np.float32) for i in (tr, va, te)]
    ytr, yva, yte = y[tr], y[va], y[te]
    S = {}                      # name -> (validation scores, test scores)
    res = []                    # resources (first split only)

    # ---- Random forest family
    rf = fit_rf(Xtr, ytr, RF_PARENT, seed)
    Tva, Tte = tree_probabilities(rf, Xva), tree_probabilities(rf, Xte)
    S['rf_parent'] = (Tva.mean(0), Tte.mean(0))
    S['rf_thinned'] = (Tva[:RF_SMALL].mean(0), Tte[:RF_SMALL].mean(0))
    if s == 0:
        thin = pickle.loads(pickle.dumps(rf)); thin.estimators_ = thin.estimators_[:RF_SMALL]; thin.n_estimators = RF_SMALL
        res += [dict(model='rf_parent', bytes=serialized_bytes(rf)), dict(model='rf_thinned', bytes=serialized_bytes(thin))]
        del thin
    del rf; gc.collect()
    m = fit_rf(Xtr, ytr, RF_SMALL, seed + 7)
    S['rf_direct_small'] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
    if s == 0: res.append(dict(model='rf_direct_small', bytes=serialized_bytes(m)))
    del m
    for k in range(1, N_RETRAIN + 1):
        m = fit_rf(Xtr, ytr, RF_PARENT, seed + 100 * k)
        S[f'rf_retrain_{k}'] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]); del m; gc.collect()

    # ---- Gradient boosting family
    g = fit_gbm(Xtr, ytr, GBM_PARENT, seed)
    S['gbm_parent'] = (gbm_scores(g, Xva), gbm_scores(g, Xte))
    S['gbm_truncated'] = (gbm_scores(g, Xva, GBM_SMALL), gbm_scores(g, Xte, GBM_SMALL))
    if s == 0:
        full_b = serialized_bytes(g)
        trunc_b = (len(g.booster_.model_to_string(num_iteration=GBM_SMALL)) if lgb is not None
                   else int(full_b * GBM_SMALL / GBM_PARENT))
        res += [dict(model='gbm_parent', bytes=full_b),
                dict(model='gbm_truncated', bytes=trunc_b, note='text model size' if lgb is not None else 'approximate (proportional)')]
    del g
    m = fit_gbm(Xtr, ytr, GBM_SMALL, seed + 7, small=True)
    S['gbm_direct_small'] = (gbm_scores(m, Xva), gbm_scores(m, Xte))
    if s == 0: res.append(dict(model='gbm_direct_small', bytes=serialized_bytes(m)))
    del m
    for k in range(1, N_RETRAIN + 1):
        m = fit_gbm(Xtr, ytr, GBM_PARENT, seed + 100 * k)
        S[f'gbm_retrain_{k}'] = (gbm_scores(m, Xva), gbm_scores(m, Xte)); del m

    # ---- Logistic regression
    m = fit_lr(Xtr, ytr)
    S['logistic'] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
    if s == 0: res.append(dict(model='logistic', bytes=serialized_bytes(m)))
    del m

    # ---- Decisions, metrics, excess comparisons
    g_unique, g_inv = np.unique(groups[te], return_inverse=True)
    boot_idx = np.random.default_rng(seed + 5).integers(0, len(g_unique), size=(BOOTSTRAP_DRAWS, len(g_unique)))
    rf_re = [f'rf_retrain_{k}' for k in range(1, N_RETRAIN + 1)]
    gb_re = [f'gbm_retrain_{k}' for k in range(1, N_RETRAIN + 1)]
    plan = [('rf', 'rf_parent', rf_re, ['rf_thinned', 'rf_direct_small', 'logistic']),
            ('gbm', 'gbm_parent', gb_re, ['gbm_truncated', 'gbm_direct_small', 'logistic'])]
    metrics, comparisons, mechanism = [], [], []
    for b in ALERT_BUDGETS:
        thr = {n: float(np.quantile(v, 1 - b)) for n, (v, t) in S.items()}
        dec = {n: S[n][1] >= thr[n] for n in S}
        for n in S:
            metrics.append(dict(budget=b, model=n, threshold=thr[n],
                                validation_alert_rate=float((S[n][0] >= thr[n]).mean()), **model_metrics(yte, S[n][1], dec[n])))
        for fam, ref, retr, cands in plan:
            for c in cands:
                comparisons.append(dict(budget=b, family=fam, reference=ref, candidate=c,
                                        **excess_comparison(yte, g_inv, len(g_unique), boot_idx, dec, ref, c, retr)))
            # noise floor: retrain 1 vs retrain 2 (should be ~0 excess by construction)
            if len(retr) >= 2:
                cv = change_vectors(yte, dec[retr[0]], dec[retr[1]])
                comparisons.append(dict(budget=b, family=fam, reference=retr[0], candidate=retr[1] + ' (noise floor)',
                                        newly_missed_candidate=int(cv['newly_missed'].sum()),
                                        recovered_candidate=int(cv['recovered'].sum())))
        mech = thinning_mechanism(yte, Tte, thr['rf_parent'], thr['rf_thinned'], RF_SMALL, dec['rf_parent'], dec['rf_thinned'])
        quart, posal = mech.pop('_quart'), mech.pop('_pos_alerted')
        mech['margin_quartile_lost_retrain_mean'] = [float(np.mean([(~dec[r][posal])[quart == i].sum() for r in rf_re])) for i in range(4)]
        mechanism.append(dict(budget=b, **mech))

    out = dict(signature=SIGNATURE, dataset=name, split=s, seed=seed,
               n_train=len(tr), n_val=len(va), n_test=len(te), events_test=int(yte.sum()),
               n_test_patients=len(g_unique), n_features_after_preprocessing=int(Xtr.shape[1]),
               metrics=metrics, comparisons=comparisons, mechanism=mechanism, resources=res,
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(OUT / 'splits' / f'{name}_split{s:02d}_scores.npz', y_test=yte, test_index=te,
                        **{f'test_{n}': S[n][1].astype(np.float32) for n in S})
    tmp = path.with_suffix('.part'); tmp.write_text(json.dumps(out, default=float)); tmp.replace(path)
    return f"saved in {out['seconds']} s"

for name in DATASETS:
    for s in range(N_SPLITS):
        status = run_split(name, s)
        print(f'{name} split {s+1:02d}/{N_SPLITS}: {status}', flush=True)
print('ALL SPLITS COMPLETE ->', OUT / 'splits')
""")

md("""
### 6 · Agrégation sur les partitions (peut être lancée à tout moment)
Les partitions répétées réutilisent la même cohorte : leurs résultats sont **corrélés**. L'intervalle utilise donc la **correction de Nadeau & Bengio (2003)** pour les ré-échantillonnages répétés : variance × (1/J + n_test/n_train), au lieu de la simple erreur standard qui serait trop optimiste.
""")
code(r"""
def nadeau_bengio(x, n_test, n_train):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; J = len(x)
    if J < 2: return (float(x.mean()) if J else np.nan, np.nan, np.nan)
    se = np.sqrt((1 / J + n_test / n_train) * x.var(ddof=1)); t = stats.t.ppf(0.975, J - 1)
    return float(x.mean()), float(x.mean() - t * se), float(x.mean() + t * se)

splits = [json.loads(p.read_text()) for p in sorted((OUT / 'splits').glob('*_split??.json'))]
splits = [r for r in splits if r.get('signature') == SIGNATURE]
assert splits, 'No completed split yet.'
comp = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], events_test=r['events_test'],
                          n_test=r['n_test'], n_train=r['n_train'], **c) for r in splits for c in r['comparisons']])
mets = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], **m) for r in splits for m in r['metrics']])
mech = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], **m) for r in splits for m in r['mechanism']])
comp.to_csv(OUT / 'comparisons_per_split.csv', index=False)
mets.to_csv(OUT / 'metrics_per_split.csv', index=False)
mech.to_csv(OUT / 'mechanism_per_split.csv', index=False)

main = comp[~comp.candidate.str.contains('noise floor')].copy()
main['excess_newly_missed_per100'] = 100 * main.excess_newly_missed / main.events_test
main['excess_net_tp_lost_per100'] = 100 * main.excess_net_tp_lost / main.events_test
rows = []
for (ds, b, fam, cand), g in main.groupby(['dataset', 'budget', 'family', 'candidate']):
    nt, ntr = g.n_test.mean(), g.n_train.mean()
    r = dict(dataset=ds, budget=b, family=fam, candidate=cand, splits=len(g),
             newly_missed=g.newly_missed_candidate.mean(), retrain_noise_newly_missed=g.newly_missed_retrain_mean.mean())
    for col in ['excess_newly_missed_per100', 'excess_net_tp_lost_per100', 'excess_delta_mcc']:
        m, lo, hi = nadeau_bengio(g[col], nt, ntr)
        r[col] = m; r[col + '_ci_low'] = lo; r[col + '_ci_high'] = hi
    r['share_splits_excess_missed_gt0'] = float((g.excess_newly_missed > 0).mean())
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'SUMMARY_excess_over_retraining.csv', index=False)

fmt = lambda m, lo, hi, d=2: f'{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]'
view = summary.assign(
    excess_missed_per100=[fmt(a, b, c) for a, b, c in zip(summary.excess_newly_missed_per100, summary.excess_newly_missed_per100_ci_low, summary.excess_newly_missed_per100_ci_high)],
    excess_net_lost_per100=[fmt(a, b, c) for a, b, c in zip(summary.excess_net_tp_lost_per100, summary.excess_net_tp_lost_per100_ci_low, summary.excess_net_tp_lost_per100_ci_high)],
    excess_dMCC=[fmt(a, b, c, 3) for a, b, c in zip(summary.excess_delta_mcc, summary.excess_delta_mcc_ci_low, summary.excess_delta_mcc_ci_high)],
)[['dataset', 'budget', 'family', 'candidate', 'splits', 'excess_missed_per100', 'excess_net_lost_per100', 'excess_dMCC', 'share_splits_excess_missed_gt0']]
print('EXCESS OVER RETRAINING  ·  mean [Nadeau–Bengio 95% CI] across splits  ·  per 100 test events')
display(view)

prim = summary[(summary.budget == PRIMARY_BUDGET) & (summary.candidate == 'rf_thinned')]
print('\nPRIMARY ENDPOINT (RF 128→16 thinning, budget', PRIMARY_BUDGET, ')')
for _, r in prim.iterrows():
    ok = r.excess_newly_missed_per100_ci_low > 0
    print(f"  {r.dataset:15s} {fmt(r.excess_newly_missed_per100, r.excess_newly_missed_per100_ci_low, r.excess_newly_missed_per100_ci_high)}"
          f"  -> {'CI excludes 0' if ok else 'CI includes 0 (not confirmed)'}")
print('  Pre-specified success = CI excludes 0 on BOTH datasets:',
      bool(len(prim) == len(DATASETS) and (prim.excess_newly_missed_per100_ci_low > 0).all()))

mk = mech.groupby(['dataset', 'budget'])[['expected_newly_missed', 'observed_newly_missed',
                                          'expected_changes', 'observed_changes', 'auroc_predicting_changes']].mean().round(2)
print('\nMECHANISM TEST (RF thinning): expected vs observed, mean per split')
display(mk)

model_view = mets[mets.budget == PRIMARY_BUDGET].groupby(['dataset', 'model'])[
    ['mcc', 'sensitivity', 'auroc', 'auprc', 'brier', 'calibration_slope', 'alert_rate']].median().round(3)
print('\nPER-MODEL METRICS (median across splits, primary budget)')
display(model_view)
""")

md("### 7 · Figures (sauvegardées dans `figures/`)")
code(r"""
import matplotlib; import matplotlib.pyplot as plt
NAVY, TEAL, ORANGE, GREY, SLATE = '#17324D', '#178F8A', '#D98C3F', '#AFC3CF', '#5E7486'
LABEL = {'rf_thinned': 'RF 128→16 thinned', 'rf_direct_small': 'RF 16 direct', 'gbm_truncated': 'GBM 400→100 truncated',
         'gbm_direct_small': 'GBM small direct', 'logistic': 'Logistic regression'}
DSCOL = {'diabetes_130': TEAL, 'physionet_2012': ORANGE}

def forest_plot(metric, xlabel, fname, budget=PRIMARY_BUDGET):
    d = summary[summary.budget == budget].copy()
    d['label'] = d.family.str.upper() + ' · ' + d.candidate.map(LABEL)
    labels = list(dict.fromkeys(d.sort_values(['family', 'candidate']).label))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.6), dpi=150)
    off = {ds: (i - (len(DATASETS) - 1) / 2) * 0.22 for i, ds in enumerate(DATASETS)}
    for ds in DATASETS:
        e = d[d.dataset == ds].set_index('label')
        ys = [len(labels) - 1 - labels.index(l) + off[ds] for l in e.index]
        ax.errorbar(e[metric], ys, xerr=[e[metric] - e[metric + '_ci_low'], e[metric + '_ci_high'] - e[metric]],
                    fmt='o', color=DSCOL.get(ds, NAVY), capsize=3, label=ds.replace('_', ' '))
    ax.axvline(0, color=SLATE, lw=1, ls='--')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel); ax.grid(axis='x', alpha=.25); ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=len(DATASETS))
    ax.set_title(f'Excess over retraining noise  ·  alert budget {int(budget*100)}%  ·  mean and Nadeau–Bengio 95% CI',
                 fontsize=10, color=NAVY, loc='left')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'figures' / fname, dpi=200, facecolor='white'); plt.show(); plt.close(fig)

forest_plot('excess_newly_missed_per100', 'Excess newly missed positives per 100 test events (>0 = worse than retraining)', 'fig1_excess_missed.png')
forest_plot('excess_delta_mcc', 'Excess ΔMCC vs retrained copies (<0 = worse than retraining)', 'fig2_excess_mcc.png')

# Mechanism: expected vs observed newly missed (one point per split)
mm = mech[mech.budget == PRIMARY_BUDGET]
fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
for ds in DATASETS:
    e = mm[mm.dataset == ds]; ax.scatter(e.expected_newly_missed, e.observed_newly_missed, color=DSCOL.get(ds, NAVY), label=ds.replace('_', ' '), alpha=.8)
lim = [0, 1.08 * max(1, mm[['expected_newly_missed', 'observed_newly_missed']].max().max())]
ax.plot(lim, lim, ls='--', color=SLATE, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Expected (tree-sampling model)'); ax.set_ylabel('Observed after thinning')
ax.set_title('Newly missed positives per split', fontsize=10, color=NAVY, loc='left'); ax.legend(frameon=False)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'fig3_mechanism_expected_vs_observed.png', dpi=200, facecolor='white'); plt.show(); plt.close(fig)

# Margin quartiles pooled over splits
fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.2 * len(DATASETS), 4), dpi=150, squeeze=False)
for ax, ds in zip(axes[0], DATASETS):
    e = mm[mm.dataset == ds]
    n = np.sum(np.vstack(e.margin_quartile_n), axis=0)
    thin = 100 * np.sum(np.vstack(e.margin_quartile_lost_thinned), axis=0) / np.maximum(1, n)
    ret = 100 * np.sum(np.vstack(e.margin_quartile_lost_retrain_mean), axis=0) / np.maximum(1, n)
    x = np.arange(4); ax.bar(x - .2, ret, .4, color=GREY, label='Retrained RF (mean)'); ax.bar(x + .2, thin, .4, color=ORANGE, label='Thinned RF')
    ax.set_xticks(x, ['Q1\nnear threshold', 'Q2', 'Q3', 'Q4\nmost confident']); ax.set_ylim(0, 100)
    ax.set_ylabel('Parent true-positive alerts lost (%)'); ax.set_title(ds.replace('_', ' '), fontsize=10, color=NAVY, loc='left')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.25)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'fig4_margin_quartiles.png', dpi=200, facecolor='white'); plt.show(); plt.close(fig)
print('Figures saved ->', OUT / 'figures')
""")

md("### 8 · Manifeste et archive partageable (sans données individuelles)")
code(r"""
res = [dict(dataset=r['dataset'], **x) for r in splits if r['split'] == 0 for x in r['resources']]
pd.DataFrame(res).to_csv(OUT / 'model_sizes_first_split.csv', index=False)
manifest = dict(status='complete' if len(splits) == len(DATASETS) * N_SPLITS else f'partial: {len(splits)} splits',
                completed_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG, signature=SIGNATURE,
                python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, platform=platform.platform(),
                data={n: DATA[n][3] for n in DATASETS},
                limits=['Repeated splits re-use one cohort per dataset (correlated); NB-corrected intervals used',
                        'No external-site or temporal validation in this run',
                        'Alert budgets are illustrative, not clinically chosen',
                        'GBM truncated size is exact for LightGBM, proportional approximation otherwise'])
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, default=str))
share = PROJECT_HOME / 'results' / (RUN_NAME + '_SHAREABLE.zip')
with zipfile.ZipFile(share, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in ['MANIFEST.json', 'PROTOCOL_LOCK.json', 'SUMMARY_excess_over_retraining.csv', 'comparisons_per_split.csv',
              'metrics_per_split.csv', 'mechanism_per_split.csv', 'model_sizes_first_split.csv']:
        if (OUT / f).exists(): z.write(OUT / f, arcname=f)
    for f in sorted((OUT / 'figures').glob('*.png')): z.write(f, arcname='figures/' + f.name)
print('Manifest:', OUT / 'MANIFEST.json'); print('Shareable ZIP (no patient-level data):', share)
""")

# ---------- write notebook + flat script ----------
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    lines = src.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines, "id": f"c{i:02d}"}
    if kind == "code": c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)
out_nb, out_py = sys.argv[1], sys.argv[2]
json.dump(nb, open(out_nb, "w"), indent=1, ensure_ascii=False)
open(out_py, "w").write("\n\n".join(src for kind, src in cells if kind == "code"))
print("cells:", len(cells))
