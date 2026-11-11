import os, sys, io, json, time, zipfile, hashlib, pickle, platform, warnings, gc, urllib.request, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
warnings.filterwarnings('ignore', category=UserWarning)

# ======================= USER CONFIGURATION =======================
QUICK_TEST   = True
NEW_DATASETS = ['support2', 'sepsis_survival', 'physionet_2019_AtoB']
OLD_DATASETS = ['physionet_2012', 'diabetes_130']        # read from the finished replication, not retrained
DOSE_TREES   = [64, 32, 16, 8, 4]                        # trees kept from the 128-tree parent
BASE_RUN     = 'EXCESS_REPLICATION_20261110_R1'
RUN_NAME     = 'EXCESS_EXTENSION_20261110_R2'
# ==================================================================
if QUICK_TEST:
    BASE_RUN += '_QUICKTEST'; RUN_NAME += '_QUICKTEST'

try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_HOME = Path('/content/drive/MyDrive/CHICCO_MEGA_RESEARCH'); MODE = 'google_colab_drive'
except ImportError:
    local = os.environ.get('CHICCO_LOCAL_DRIVE_ROOT')
    if not local: raise RuntimeError('Open in Google Colab, or set CHICCO_LOCAL_DRIVE_ROOT for a local mirror.')
    PROJECT_HOME = Path(local).expanduser().resolve(); MODE = 'explicit_local_drive_mirror'

RAW, PREP = PROJECT_HOME / 'data' / 'raw', PROJECT_HOME / 'data' / 'prepared'
BASE = PROJECT_HOME / 'results' / BASE_RUN
OUT  = PROJECT_HOME / 'results' / RUN_NAME
for d in [RAW, PREP, OUT, OUT / 'splits', OUT / 'dose', OUT / 'figures']:
    d.mkdir(parents=True, exist_ok=True)
if not (BASE / 'PROTOCOL_LOCK.json').exists():
    raise FileNotFoundError(f'Finished replication not found: {BASE}. Run the replication notebook first.')

# Re-use EXACTLY the base protocol (same trees, splits, seeds, budgets) so results are comparable.
B = json.loads((BASE / 'PROTOCOL_LOCK.json').read_text())
N_SPLITS, N_RETRAIN, ALERT_BUDGETS = B['n_splits'], B['n_retrain'], B['alert_budgets']
RF_PARENT, RF_SMALL, GBM_PARENT, GBM_SMALL, BOOTSTRAP_DRAWS = B['rf_parent'], B['rf_small'], B['gbm_parent'], B['gbm_small'], B['bootstrap']
P2012_SETS = B['p2012_sets']
PRIMARY_BUDGET = ALERT_BUDGETS[0]
DOSE_TREES = [k for k in DOSE_TREES if k < RF_PARENT] or [RF_PARENT // 2, RF_PARENT // 4]
DATASETS = NEW_DATASETS            # used by the reused aggregation code for the new run

try:
    import lightgbm as lgb
    GBM_BACKEND = 'lightgbm ' + lgb.__version__
except ImportError:
    lgb = None; GBM_BACKEND = 'sklearn HistGradientBoosting ' + sklearn.__version__
if B['gbm_backend'] != GBM_BACKEND.split()[0]:
    print('WARNING: GBM backend differs from the base run:', B['gbm_backend'], 'vs', GBM_BACKEND)

CONFIG = dict(run_name=RUN_NAME, base_run=BASE_RUN, base_signature=json.loads((BASE / 'MANIFEST.json').read_text()).get('signature') if (BASE / 'MANIFEST.json').exists() else None,
              new_datasets=NEW_DATASETS, old_datasets=OLD_DATASETS, dose_trees=DOSE_TREES, n_splits=N_SPLITS,
              n_retrain=N_RETRAIN, alert_budgets=ALERT_BUDGETS, rf_parent=RF_PARENT, rf_small=RF_SMALL,
              gbm_parent=GBM_PARENT, gbm_small=GBM_SMALL, bootstrap=BOOTSTRAP_DRAWS, gbm_backend=GBM_BACKEND.split()[0],
              protocol_version='2026-11-10_extension_v1')
lock = OUT / 'PROTOCOL_LOCK.json'
if lock.exists() and json.loads(lock.read_text()) != CONFIG:
    raise RuntimeError(f'{OUT} holds a run with another protocol. Change RUN_NAME.')
lock.write_text(json.dumps(CONFIG, indent=2))
SIGNATURE = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
try:
    from IPython.display import display
except ImportError:
    display = lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)
print('MODE:', MODE, '| GBM:', GBM_BACKEND, '| base run:', BASE.name)
print('New run ->', OUT); print('Protocol re-used:', N_SPLITS, 'splits,', N_RETRAIN, 'retrains, RF', RF_PARENT, '→', RF_SMALL, '| dose:', DOSE_TREES)

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



def _norm(c): return str(c).lower().replace('.', '').replace('_', '').replace(' ', '')

def _uci_frame(uci_id, cache_name):
    cache = RAW / cache_name
    if cache.exists():
        return pd.read_csv(cache)
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'ucimlrepo'], check=True)
        from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=uci_id)
    df = getattr(d.data, 'original', None)
    if df is None:
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df.to_csv(cache, index=False)
    return df

SUPPORT_PREDICTORS = ['age', 'sex', 'race', 'edu', 'income', 'num.co', 'diabetes', 'dementia', 'ca', 'dzgroup', 'dzclass',
                      'hday', 'scoma', 'sps', 'aps', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb', 'bili',
                      'crea', 'sod', 'ph', 'glucose', 'bun', 'urine']
# Never used: death, d.time, sfdm2, slos, charges, totcst, totmcst, avtisst, dnr, dnrday, surv2m, surv6m, prg2m, prg6m, adl*

def load_support2():
    df = _uci_frame(880, 'support2_uci880.csv')
    cols = {_norm(c): c for c in df.columns}
    if 'hospdead' not in cols: raise ValueError(f'hospdead not found. Columns: {list(df.columns)}')
    keep = [cols[_norm(k)] for k in SUPPORT_PREDICTORS if _norm(k) in cols]
    missing = [k for k in SUPPORT_PREDICTORS if _norm(k) not in cols]
    if missing: print('SUPPORT2: pre-declared predictors not found (skipped):', missing)
    if len(keep) < 15: raise ValueError(f'Too few SUPPORT2 predictors matched. Columns: {list(df.columns)}')
    y = pd.to_numeric(df[cols['hospdead']], errors='coerce'); ok = y.notna().to_numpy()
    X = df.loc[ok, keep].copy().reset_index(drop=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('object').where(X[c].notna(), 'Missing').astype(str)
    y = y[ok].astype(int).to_numpy()
    groups = np.array([f's{i}' for i in range(len(y))])
    meta = dict(dataset='SUPPORT2 (UCI 880)', task='in-hospital death; predictors from study entry / day 3',
                predictors=keep, group='one record per patient')
    return X, y, groups, meta

def load_sepsis_survival():
    df = _uci_frame(827, 'sepsis_survival_uci827.csv')
    out = [c for c in df.columns if 'outcome' in str(c).lower()]
    if not out: raise ValueError(f'outcome column not found. Columns: {list(df.columns)}')
    alive = pd.to_numeric(df[out[0]], errors='coerce'); ok = alive.notna().to_numpy()
    feats = [c for c in df.columns if c != out[0] and 'id' not in _norm(c)]
    X = df.loc[ok, feats].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
    y = (1 - alive[ok]).astype(int).to_numpy()                 # 1 = died
    groups = np.array([f'e{i}' for i in range(len(y))])        # no patient identifier available
    meta = dict(dataset='Sepsis Survival Minimal Clinical Records (UCI 827; Chicco & Jurman 2020)',
                task='death during hospital stay from age, sex and episode number', predictors=feats,
                group='episode (NO patient identifier: episodes of one patient may be split)')
    return X, y, groups, meta

SEPSIS_SOURCE = 's3://physionet-open/challenge-2019/1.0.0/training'
def _p2019(cohort):
    for d in [PROJECT_HOME / 'data' / 'v3_prepared', PREP]:
        f = d / f'physionet2019_{cohort}_landmark12_v3.pkl'
        if f.exists():
            return pd.read_pickle(f)
    folder = RAW / f'physionet2019_{cohort}'
    if len(list(folder.glob('*.psv'))) < 19000:
        if not shutil.which('aws'):
            subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'awscli'], check=True)
        folder.mkdir(parents=True, exist_ok=True)
        print('SYNC PhysioNet 2019 cohort', cohort, '(public mirror, resumable)', flush=True)
        subprocess.run(['aws', 's3', 'sync', '--no-sign-request', f'{SEPSIS_SOURCE}/training_set{cohort}/', str(folder),
                        '--only-show-errors'], check=True, timeout=5400)
    rows = []
    for file in sorted(folder.glob('*.psv')):
        d0 = pd.read_csv(file, sep='|', na_values=['NaN'])
        if len(d0) < 12 or 'SepsisLabel' not in d0: continue
        prefix = d0.iloc[:12]                                  # first 12 ICU hours only
        r = {'patient_file': file.name, 'target': int(pd.to_numeric(prefix.SepsisLabel, errors='coerce').iloc[-1] > 0)}
        for col in prefix.columns:
            if col in {'SepsisLabel', 'ICULOS'}: continue
            v = pd.to_numeric(prefix[col], errors='coerce')
            r[col + '__last'] = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else np.nan
            r[col + '__mean'] = float(v.mean()) if v.notna().any() else np.nan
        rows.append(r)
    d = pd.DataFrame(rows); d.to_pickle(PREP / f'physionet2019_{cohort}_landmark12_v3.pkl')
    return d

COHORT_B = {}
def load_physionet_2019_AtoB():
    a, b = _p2019('A'), _p2019('B')
    cols = [c for c in a.columns if c not in ('patient_file', 'target') and c in b.columns]
    X = pd.concat([a[cols], b[cols]], ignore_index=True)
    y = np.r_[a['target'].to_numpy(), b['target'].to_numpy()].astype(int)
    groups = np.r_[('A_' + a['patient_file'].astype(str)).to_numpy(), ('B_' + b['patient_file'].astype(str)).to_numpy()]
    COHORT_B['physionet_2019_AtoB'] = np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)]
    meta = dict(dataset='PhysioNet/CinC Challenge 2019', task='sepsis label at fixed ICU hour 12; train on hospital A, test on hospital B',
                n_A=len(a), events_A=int(a['target'].sum()), n_B=len(b), events_B=int(b['target'].sum()),
                group='one record per patient', note='fixed external test set B; NB interval conservative')
    return X, y, groups, meta

def split_A_to_B(y, groups, seed):
    isB = COHORT_B['physionet_2019_AtoB']; A = np.where(~isB)[0]; te = np.where(isB)[0]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=seed + 1).split(A, y[A], groups[A]))
    return A[a], A[b], te

LOADERS = {'diabetes_130': load_diabetes_130, 'physionet_2012': load_physionet_2012, 'support2': load_support2,
           'sepsis_survival': load_sepsis_survival, 'physionet_2019_AtoB': load_physionet_2019_AtoB}
DATA = {}
for name in OLD_DATASETS + NEW_DATASETS:
    X, y, g, meta = LOADERS[name]()
    meta.update(n=len(y), events=int(y.sum()), prevalence=round(float(y.mean()), 4), n_groups=len(np.unique(g)))
    DATA[name] = (X, y, g, meta)
    (OUT / f'DATA_{name}.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'{name}: {meta["n"]} rows, {meta["n_groups"]} groups, {meta["events"]} events ({100*meta["prevalence"]:.1f}%), {X.shape[1]} features')

from sklearn.model_selection import GroupShuffleSplit
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
SPLITTERS = {'physionet_2019_AtoB': split_A_to_B}

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
    tr, va, te = SPLITTERS.get(name, grouped_split)(y, groups, seed)
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


for name in NEW_DATASETS:
    for s in range(N_SPLITS):
        status = run_split(name, s)
        print(f'{name} split {s+1:02d}/{N_SPLITS}: {status}', flush=True)
print('PART A COMPLETE ->', OUT / 'splits')

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

RUN_DIR = {n: BASE for n in OLD_DATASETS}; RUN_DIR.update({n: OUT for n in NEW_DATASETS})
_MET = {}
def saved_metrics(run_dir):
    if run_dir not in _MET:
        rows = [dict(dataset=r['dataset'], split=r['split'], **m)
                for p in sorted((run_dir / 'splits').glob('*_split??.json'))
                for r in [json.loads(p.read_text())] for m in r['metrics']]
        _MET[run_dir] = pd.DataFrame(rows)
    return _MET[run_dir]

def dose_split(name, s):
    path = OUT / 'dose' / f'{name}_split{s:02d}.json'
    if path.exists() and json.loads(path.read_text()).get('signature') == SIGNATURE:
        return 'resumed'
    t0 = time.time()
    X, y, groups, meta = DATA[name]; seed = 1000 + 37 * s
    tr, va, te = SPLITTERS.get(name, grouped_split)(y, groups, seed)
    z = np.load(RUN_DIR[name] / 'splits' / f'{name}_split{s:02d}_scores.npz')
    assert np.array_equal(z['test_index'], te), 'split differs from the saved run'
    met = saved_metrics(RUN_DIR[name]); met = met[(met.dataset == name) & (met.split == s)]
    prep = make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
    Xtr, Xva, Xte = [prep.transform(X.iloc[i]).astype(np.float32) for i in (tr, va, te)]
    rf = fit_rf(Xtr, y[tr], RF_PARENT, seed)
    Tva, Tte = tree_probabilities(rf, Xva), tree_probabilities(rf, Xte)
    max_diff = float(np.max(np.abs(Tte.mean(0) - z['test_rf_parent'])))
    sizes = {}
    if s == 0:
        for k in DOSE_TREES:
            th = pickle.loads(pickle.dumps(rf)); th.estimators_ = th.estimators_[:k]; th.n_estimators = k
            sizes[k] = serialized_bytes(th); del th
        sizes[RF_PARENT] = serialized_bytes(rf)
    del rf; gc.collect()
    yte = y[te]; g_unique, g_inv = np.unique(groups[te], return_inverse=True)
    boot_idx = np.random.default_rng(seed + 5).integers(0, len(g_unique), size=(BOOTSTRAP_DRAWS, len(g_unique)))
    retr = [f'rf_retrain_{k}' for k in range(1, N_RETRAIN + 1)]
    rows = []
    for b in ALERT_BUDGETS:
        mb = met[met.budget == b].set_index('model')['threshold']
        thr_p = float(np.quantile(Tva.mean(0), 1 - b))
        dec = {'rf_parent': Tte.mean(0) >= thr_p}
        for r in retr: dec[r] = z['test_' + r] >= mb[r]
        for k in DOSE_TREES:
            thr_k = float(np.quantile(Tva[:k].mean(0), 1 - b)); key = f'thin_{k}'
            dec[key] = Tte[:k].mean(0) >= thr_k
            ex = excess_comparison(yte, g_inv, len(g_unique), boot_idx, dec, 'rf_parent', key, retr)
            mech = thinning_mechanism(yte, Tte, thr_p, thr_k, k, dec['rf_parent'], dec[key])
            mech.pop('_quart'); mech.pop('_pos_alerted')
            rows.append(dict(budget=b, trees=k, **ex, **{f'mech_{a}': v for a, v in mech.items()}))
    out = dict(signature=SIGNATURE, dataset=name, split=s, n_train=len(tr), n_test=len(te), events_test=int(yte.sum()),
               parent_max_abs_diff_vs_saved=max_diff, sizes_bytes=sizes, rows=rows, seconds=round(time.time() - t0, 1))
    tmp = path.with_suffix('.part'); tmp.write_text(json.dumps(out, default=float)); tmp.replace(path)
    return f"saved in {out['seconds']} s (parent reproduced, max |Δscore| = {max_diff:.1e})"

for name in OLD_DATASETS + NEW_DATASETS:
    for s in range(N_SPLITS):
        print(f'DOSE {name} split {s+1:02d}/{N_SPLITS}: {dose_split(name, s)}', flush=True)
print('PART B COMPLETE ->', OUT / 'dose')

dose = [json.loads(p.read_text()) for p in sorted((OUT / 'dose').glob('*_split??.json'))]
dose = [r for r in dose if r.get('signature') == SIGNATURE]
dd = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], events_test=r['events_test'], n_test=r['n_test'],
                        n_train=r['n_train'], **x) for r in dose for x in r['rows']])
dd['excess_missed_per100'] = 100 * dd.excess_newly_missed / dd.events_test
dd.to_csv(OUT / 'dose_response_per_split.csv', index=False)
worst = max(r['parent_max_abs_diff_vs_saved'] for r in dose)
print(f'Parent reproduction check: max |score difference| vs saved run = {worst:.2e}',
      '(identical forests)' if worst < 1e-5 else '(NOT identical: the rebuilt parent is used consistently as reference)')
rows = []
for (ds, b, k), g in dd.groupby(['dataset', 'budget', 'trees']):
    m, lo, hi = nadeau_bengio(g.excess_missed_per100, g.n_test.mean(), g.n_train.mean())
    mm, mlo, mhi = nadeau_bengio(g.excess_delta_mcc, g.n_test.mean(), g.n_train.mean())
    rows.append(dict(dataset=ds, budget=b, trees=k, splits=len(g), excess_missed_per100=m, ci_low=lo, ci_high=hi,
                     excess_delta_mcc=mm, mcc_ci_low=mlo, mcc_ci_high=mhi,
                     expected_missed=g.mech_expected_newly_missed.mean(), observed_missed=g.mech_observed_newly_missed.mean(),
                     auroc_flip=g.mech_auroc_predicting_changes.mean()))
dose_summary = pd.DataFrame(rows); dose_summary.to_csv(OUT / 'SUMMARY_dose_response.csv', index=False)
sz = pd.DataFrame([dict(dataset=r['dataset'], trees=int(k), bytes=v) for r in dose if r['split'] == 0 for k, v in r['sizes_bytes'].items()])
sz.to_csv(OUT / 'dose_model_sizes.csv', index=False)
v = dose_summary[dose_summary.budget == PRIMARY_BUDGET].copy()
v['excess per 100 [NB 95% CI]'] = [f'{a:+.2f} [{b:+.2f}, {c:+.2f}]' for a, b, c in zip(v.excess_missed_per100, v.ci_low, v.ci_high)]
v['expected vs observed'] = [f'{a:.1f} vs {b:.1f}' for a, b in zip(v.expected_missed, v.observed_missed)]
display(v[['dataset', 'trees', 'excess per 100 [NB 95% CI]', 'expected vs observed', 'auroc_flip']].round(3))

import matplotlib.pyplot as plt
NAVY, TEAL, ORANGE, GREY, SLATE = '#17324D', '#178F8A', '#D98C3F', '#AFC3CF', '#5E7486'
ALL = OLD_DATASETS + NEW_DATASETS
PAL = dict(zip(ALL, [ORANGE, TEAL, NAVY, '#8E5BA8', '#C0504D']))
NICE = {'diabetes_130': 'Diabetes 130', 'physionet_2012': 'PhysioNet 2012', 'support2': 'SUPPORT2',
        'sepsis_survival': 'Sepsis survival', 'physionet_2019_AtoB': 'PhysioNet 2019 A→B'}

S5 = pd.concat([pd.read_csv(BASE / 'SUMMARY_excess_over_retraining.csv'), pd.read_csv(OUT / 'SUMMARY_excess_over_retraining.csv')])
S5.to_csv(OUT / 'SUMMARY_excess_5datasets.csv', index=False)
prim = S5[(S5.budget == PRIMARY_BUDGET) & S5.candidate.isin(['rf_thinned', 'gbm_truncated'])].copy()
prim['excess per 100 [NB 95% CI]'] = [f'{a:+.2f} [{b:+.2f}, {c:+.2f}]' for a, b, c in
                                      zip(prim.excess_newly_missed_per100, prim.excess_newly_missed_per100_ci_low, prim.excess_newly_missed_per100_ci_high)]
prim['excess ΔMCC [NB 95% CI]'] = [f'{a:+.3f} [{b:+.3f}, {c:+.3f}]' for a, b, c in
                                   zip(prim.excess_delta_mcc, prim.excess_delta_mcc_ci_low, prim.excess_delta_mcc_ci_high)]
print('PRIMARY ENDPOINT ACROSS 5 DATASETS (10% budget)')
display(prim[['dataset', 'candidate', 'excess per 100 [NB 95% CI]', 'excess ΔMCC [NB 95% CI]', 'share_splits_excess_missed_gt0']])

# Full metric suite requested by Davide (medians over splits, primary budget)
M5 = pd.concat([pd.read_csv(BASE / 'metrics_per_split.csv'), pd.read_csv(OUT / 'metrics_per_split.csv')])
M5 = M5[(M5.budget == PRIMARY_BUDGET) & ~M5.model.str.contains('retrain')].copy()
tp, fp, fn, tn = [M5[c].astype(float) for c in ('TP', 'FP', 'FN', 'TN')]
M5['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
M5['precision'] = tp / (tp + fp).replace(0, np.nan)
M5['npv'] = tn / (tn + fn).replace(0, np.nan)
M5['f1'] = 2 * tp / (2 * tp + fp + fn).replace(0, np.nan)
M5['balanced_accuracy'] = (M5.sensitivity + M5.specificity) / 2
suite = M5.groupby(['dataset', 'model'])[['mcc', 'f1', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision',
                                          'npv', 'brier', 'auroc', 'auprc']].median().round(3)
suite.to_csv(OUT / 'FULL_METRIC_SUITE_5datasets.csv')
spread = suite.groupby('dataset').agg(accuracy_range=('accuracy', lambda x: x.max() - x.min()), mcc_range=('mcc', lambda x: x.max() - x.min()))
print('\nACCURACY HIDES DIFFERENCES: range across models of accuracy vs MCC (primary budget)')
display(spread.round(3)); display(suite)

# Figure A: primary endpoint, 5 datasets
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
cands = [('rf_thinned', 'RF 128→16 thinned'), ('rf_direct_small', 'RF 16 direct'), ('gbm_truncated', 'GBM 400→100 truncated'), ('gbm_direct_small', 'GBM small direct')]
pr = S5[S5.budget == PRIMARY_BUDGET]
for j, ds in enumerate(ALL):
    e = pr[pr.dataset == ds].set_index('candidate')
    ys = [len(cands) - 1 - i + (j - 2) * 0.14 for i in range(len(cands))]
    for yv, (c, _) in zip(ys, cands):
        if c in e.index:
            r = e.loc[c]
            ax.errorbar(r.excess_newly_missed_per100, yv, xerr=[[r.excess_newly_missed_per100 - r.excess_newly_missed_per100_ci_low],
                        [r.excess_newly_missed_per100_ci_high - r.excess_newly_missed_per100]], fmt='o', color=PAL[ds], capsize=2, ms=4,
                        label=NICE[ds] if c == cands[0][0] else None)
ax.axvline(0, color=SLATE, ls='--', lw=1); ax.set_yticks(range(len(cands))); ax.set_yticklabels([c[1] for c in cands][::-1])
ax.set_xlabel('Excess newly missed positives per 100 test events (>0 = worse than retraining)')
ax.set_title('Excess over retraining noise · 5 EHR tasks · 10% alert budget · Nadeau–Bengio 95% CI', fontsize=10, color=NAVY, loc='left')
ax.legend(frameon=False, fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.18)); ax.grid(axis='x', alpha=.25)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'figA_excess_5datasets.png', dpi=200, facecolor='white'); plt.show(); plt.close(fig)

# Figure B: dose–response + mechanism
dv = dose_summary[dose_summary.budget == PRIMARY_BUDGET]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
for ds in ALL:
    e = dv[dv.dataset == ds].sort_values('trees')
    if e.empty: continue
    a1.errorbar(e.trees, e.excess_missed_per100, yerr=[e.excess_missed_per100 - e.ci_low, e.ci_high - e.excess_missed_per100],
                marker='o', color=PAL[ds], capsize=2, label=NICE[ds])
    a2.scatter(e.expected_missed, e.observed_missed, color=PAL[ds], label=NICE[ds])
a1.set_xscale('log', base=2); a1.set_xticks(sorted(dv.trees.unique())); a1.set_xticklabels([str(int(t)) for t in sorted(dv.trees.unique())])
a1.invert_xaxis(); a1.axhline(0, color=SLATE, ls='--', lw=1)
a1.set_xlabel(f'Trees kept from the {RF_PARENT}-tree parent'); a1.set_ylabel('Excess newly missed per 100 events')
a1.set_title('Dose–response: excess harm vs trees kept', fontsize=10, color=NAVY, loc='left'); a1.legend(frameon=False, fontsize=8)
lim = [0, 1.05 * max(1, dv[['expected_missed', 'observed_missed']].max().max())]
a2.plot(lim, lim, ls='--', color=SLATE, lw=1); a2.set_xlim(lim); a2.set_ylim(lim)
a2.set_xlabel('Expected (tree-sampling model)'); a2.set_ylabel('Observed'); a2.set_title('Mechanism at every compression level', fontsize=10, color=NAVY, loc='left')
for ax in (a1, a2):
    ax.grid(alpha=.25)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'figB_dose_response_mechanism.png', dpi=200, facecolor='white'); plt.show(); plt.close(fig)
print('Figures ->', OUT / 'figures')

manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG, signature=SIGNATURE,
                python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, platform=platform.platform(),
                data={n: DATA[n][3] for n in DATA},
                limits=['Sepsis Survival: no patient identifier, episode-level splits',
                        'PhysioNet 2019 A→B: fixed external test set; NB interval conservative; alert rate on B differs from budget',
                        'Repeated splits re-use one cohort per dataset (correlated); NB-corrected intervals used',
                        'Alert budgets are illustrative, not clinically chosen'])
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, default=str))
share = PROJECT_HOME / 'results' / (RUN_NAME + '_SHAREABLE.zip')
with zipfile.ZipFile(share, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ['MANIFEST.json', 'PROTOCOL_LOCK.json', 'SUMMARY_excess_over_retraining.csv', 'SUMMARY_excess_5datasets.csv',
              'SUMMARY_dose_response.csv', 'dose_response_per_split.csv', 'dose_model_sizes.csv', 'FULL_METRIC_SUITE_5datasets.csv',
              'comparisons_per_split.csv', 'metrics_per_split.csv', 'mechanism_per_split.csv']:
        if (OUT / f).exists(): zf.write(OUT / f, arcname=f)
    for f in sorted((OUT / 'figures').glob('*.png')): zf.write(f, arcname='figures/' + f.name)
print('Shareable ZIP (no patient-level data):', share)