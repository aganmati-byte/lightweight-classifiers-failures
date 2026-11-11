import os, sys, io, re, json, time, zipfile, hashlib, pickle, platform, warnings, gc, urllib.request, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
warnings.filterwarnings('ignore')

# ======================= USER CONFIGURATION =======================
QUICK_TEST = True
RUN_NAME   = 'GBM_EQUAL_STORAGE_20261111_R4'
GBM_LEVELS = [200, 100, 50, 25, 10]          # boosting rounds kept from the 400-round reference
SOURCE_RUNS = {'EXCESS_REPLICATION_20261110_R1': ['physionet_2012', 'diabetes_130'],
               'EXCESS_EXTENSION_20261110_R2':   ['support2', 'sepsis_survival', 'physionet_2019_AtoB'],
               'EXCESS_TIERB_DAVIDE_20261110_R3': None}      # None = every dataset of that run
# ==================================================================
QUICK_SPLITS = 2
if QUICK_TEST: RUN_NAME += '_QUICKTEST'

try:
    from google.colab import drive
    drive.mount('/content/drive'); PROJECT_HOME = Path('/content/drive/MyDrive/CHICCO_MEGA_RESEARCH'); MODE = 'google_colab_drive'
except ImportError:
    local = os.environ.get('CHICCO_LOCAL_DRIVE_ROOT')
    if not local: raise RuntimeError('Open in Google Colab, or set CHICCO_LOCAL_DRIVE_ROOT for a local mirror.')
    PROJECT_HOME = Path(local).expanduser().resolve(); MODE = 'explicit_local_drive_mirror'

RAW, PREP = PROJECT_HOME / 'data' / 'raw', PROJECT_HOME / 'data' / 'prepared'
DAV = RAW / 'davide'
OUT = PROJECT_HOME / 'results' / RUN_NAME
for d in [OUT, OUT / 'splits', OUT / 'figures']: d.mkdir(parents=True, exist_ok=True)
for pkg in ['openpyxl', 'xlrd', 'pyreadr', 'ucimlrepo']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=False)
try:
    import lightgbm as lgb; GBM_BACKEND = 'lightgbm ' + lgb.__version__
except ImportError:
    raise ImportError('LightGBM is required (it was used in the original runs).')
try:
    from IPython.display import display
except ImportError:
    display = lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)

# Protocols of the finished runs (read only)
RUNS = {r: PROJECT_HOME / 'results' / r for r in SOURCE_RUNS}
for r, p in RUNS.items():
    if not (p / 'splits').exists(): raise FileNotFoundError(f'Finished run not found: {p}')
LOCK = json.loads((RUNS['EXCESS_REPLICATION_20261110_R1'] / 'PROTOCOL_LOCK.json').read_text())
TIERB = json.loads((RUNS['EXCESS_TIERB_DAVIDE_20261110_R3'] / 'PROTOCOL.json').read_text())
P2012_SETS = LOCK['p2012_sets']
N_RETRAIN, ALERT_BUDGETS, GBM_PARENT, BOOTSTRAP_DRAWS = LOCK['n_retrain'], LOCK['alert_budgets'], LOCK['gbm_parent'], LOCK['bootstrap']
assert (TIERB['n_retrain'], TIERB['gbm_parent'], TIERB['alert_budgets']) == (N_RETRAIN, GBM_PARENT, ALERT_BUDGETS), 'runs use different protocols'
MIN_PATIENTS, MIN_MINORITY = 250, 50
PRIMARY_BUDGET = ALERT_BUDGETS[0]
LEVELS = [k for k in GBM_LEVELS if k < GBM_PARENT]

RUN_OF, NSPLITS_OF = {}, {}
for r, ds in SOURCE_RUNS.items():
    if ds is None: ds = TIERB['datasets']
    n = LOCK['n_splits'] if r != 'EXCESS_TIERB_DAVIDE_20261110_R3' else TIERB['n_splits']
    for d in ds:
        RUN_OF[d] = RUNS[r]; NSPLITS_OF[d] = min(n, QUICK_SPLITS) if QUICK_TEST else n

CONFIG = dict(run_name=RUN_NAME, source_runs=list(SOURCE_RUNS), datasets=list(RUN_OF), n_splits=NSPLITS_OF, levels=LEVELS,
              gbm_parent=GBM_PARENT, n_retrain=N_RETRAIN, alert_budgets=ALERT_BUDGETS, bootstrap=BOOTSTRAP_DRAWS,
              gbm_backend=GBM_BACKEND.split()[0], protocol_version='2026-11-11_gbm_equal_storage_v1')
lock = OUT / 'PROTOCOL_LOCK.json'
if lock.exists() and json.loads(lock.read_text()) != CONFIG:
    raise RuntimeError(f'{OUT} holds a run with another protocol. Change RUN_NAME.')
lock.write_text(json.dumps(CONFIG, indent=2))
SIGNATURE = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
print('MODE:', MODE, '| GBM:', GBM_BACKEND, '| results ->', OUT)
print('Levels:', LEVELS, '| datasets:', {d: NSPLITS_OF[d] for d in RUN_OF})

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

from sklearn.model_selection import GroupShuffleSplit

def read_tables(p):
    s = p.suffix.lower()
    try:
        if s in {'.xlsx', '.xls'}:
            return {f'{p.name}::{k}': v for k, v in pd.read_excel(p, sheet_name=None).items()}
        if s in {'.csv', '.txt', '.tsv'}:
            return {p.name: pd.read_csv(p, sep=None, engine='python')}
        if s in {'.rda', '.rdata'}:
            import pyreadr
            return {f'{p.name}::{k}': v for k, v in pyreadr.read_r(str(p)).items()}
        if s == '.sav': return {p.name: pd.read_spss(p)}
        if s == '.dta': return {p.name: pd.read_stata(p)}
        if s == '.zip':
            out = {}
            with zipfile.ZipFile(p) as z:
                z.extractall(p.parent / (p.stem + '_unzipped'))
            for q in (p.parent / (p.stem + '_unzipped')).rglob('*'):
                if q.is_file(): out.update(read_tables(q))
            return out
    except Exception as e:
        print('  cannot read', p.name, '-', type(e).__name__, e)
    return {}


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

LOADERS = {'diabetes_130': load_diabetes_130, 'physionet_2012': load_physionet_2012, 'support2': load_support2,
           'sepsis_survival': load_sepsis_survival, 'physionet_2019_AtoB': load_physionet_2019_AtoB}
DATA = {}
for name in RUN_OF:
    if name in LOADERS:
        X, y, g, meta = LOADERS[name]()
    else:
        d, info = build_dataset(name)
        if d is None: raise RuntimeError(f'{name}: {info.get("status")}')
        tgt = TIERB['targets'][name]
        assert [str(info['target']), str(info['positive'])] == [str(tgt[0]), str(tgt[1])], f'{name}: target differs from run R3'
        X, y, g, meta = d
    DATA[name] = (X, y, g, meta)
    print(f'{name:26s} {len(y):7d} rows  {int(y.sum()):6d} events ({100 * y.mean():.1f}%)  {X.shape[1]} features')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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

def serialized_bytes(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

def counts(y, alert):
    y = y.astype(bool); a = alert.astype(bool)
    return dict(TP=int((y & a).sum()), FP=int((~y & a).sum()), FN=int((y & ~a).sum()), TN=int((~y & ~a).sum()))

def mcc_from(tp, tn, fp, fn):
    tp, tn, fp, fn = [np.asarray(v, float) for v in (tp, tn, fp, fn)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)

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

def nadeau_bengio(x, n_test, n_train):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; J = len(x)
    if J < 2: return (float(x.mean()) if J else np.nan, np.nan, np.nan)
    se = np.sqrt((1 / J + n_test / n_train) * x.var(ddof=1)); t = stats.t.ppf(0.975, J - 1)
    return float(x.mean()), float(x.mean() - t * se), float(x.mean() + t * se)

def dersimonian_laird(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float); k = len(est)
    w = 1 / se**2; fe = np.sum(w * est) / np.sum(w); Q = np.sum(w * (est - fe)**2)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w))) if k > 1 else 0.0
    wr = 1 / (se**2 + tau2); mu = np.sum(wr * est) / np.sum(wr); se_mu = np.sqrt(1 / np.sum(wr))
    t = stats.t.ppf(0.975, max(1, k - 2)); pi = (mu - t * np.sqrt(tau2 + se_mu**2), mu + t * np.sqrt(tau2 + se_mu**2))
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(k=k, pooled=mu, ci_low=mu - 1.96 * se_mu, ci_high=mu + 1.96 * se_mu, pi_low=pi[0], pi_high=pi[1], tau2=tau2, I2=i2)

SPLITTERS = {'physionet_2019_AtoB': split_A_to_B}

def gbm_split(name, s):
    path = OUT / 'splits' / f'{name}_split{s:02d}.json'
    if path.exists():
        try:
            if json.loads(path.read_text()).get('signature') == SIGNATURE: return 'resumed'
        except Exception: pass
    t0 = time.time()
    X, y, groups, meta = DATA[name]; seed = 1000 + 37 * s
    tr, va, te = SPLITTERS.get(name, grouped_split)(y, groups, seed)
    z = np.load(RUN_OF[name] / 'splits' / f'{name}_split{s:02d}_scores.npz')
    assert np.array_equal(z['test_index'], te), 'split differs from the saved run'
    prep = make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
    Xtr, Xva, Xte = [prep.transform(X.iloc[i]).astype(np.float32) for i in (tr, va, te)]
    ytr, yte = y[tr], y[te]

    g = fit_gbm(Xtr, ytr, GBM_PARENT, seed)
    S = {'gbm_parent': (gbm_scores(g, Xva), gbm_scores(g, Xte))}
    for k in LEVELS: S[f'trunc_{k}'] = (gbm_scores(g, Xva, k), gbm_scores(g, Xte, k))
    sizes = {}
    if s == 0:
        sizes = dict(parent_pickle=serialized_bytes(g), parent_text=len(g.booster_.model_to_string()),
                     **{f'trunc_{k}_text': len(g.booster_.model_to_string(num_iteration=k)) for k in LEVELS})
    del g
    retr = []
    for r in range(1, N_RETRAIN + 1):
        m = fit_gbm(Xtr, ytr, GBM_PARENT, seed + 100 * r)
        S[f'gbm_retrain_{r}'] = (gbm_scores(m, Xva), gbm_scores(m, Xte)); retr.append(f'gbm_retrain_{r}'); del m
    check = {'parent': float(np.max(np.abs(S['gbm_parent'][1] - z['test_gbm_parent'])))}
    if 'test_gbm_truncated' in z.files and 100 in LEVELS:
        check['trunc_100'] = float(np.max(np.abs(S['trunc_100'][1] - z['test_gbm_truncated'])))
    for r in retr:
        if 'test_' + r in z.files: check[r] = float(np.max(np.abs(S[r][1] - z['test_' + r])))

    g_unique, g_inv = np.unique(groups[te], return_inverse=True)
    boot_idx = np.random.default_rng(seed + 5).integers(0, len(g_unique), size=(BOOTSTRAP_DRAWS, len(g_unique)))
    rows, quart_rows = [], []
    for b in ALERT_BUDGETS:
        thr = {n: float(np.quantile(v, 1 - b)) for n, (v, t) in S.items()}
        dec = {n: S[n][1] >= thr[n] for n in S}
        for k in LEVELS:
            rows.append(dict(budget=b, rounds=k, **excess_comparison(yte, g_inv, len(g_unique), boot_idx, dec, 'gbm_parent', f'trunc_{k}', retr)))
        posal = (yte == 1) & dec['gbm_parent']
        margin = S['gbm_parent'][1][posal] - thr['gbm_parent']
        q = np.quantile(margin, [0.25, 0.5, 0.75]) if posal.sum() >= 8 else np.array([0, 0, 0])
        quart = np.digitize(margin, q)
        qr = dict(budget=b, n=[int((quart == i).sum()) for i in range(4)],
                  lost_retrain_mean=[float(np.mean([(~dec[r][posal])[quart == i].sum() for r in retr])) for i in range(4)])
        for k in LEVELS: qr[f'lost_trunc_{k}'] = [int((~dec[f'trunc_{k}'][posal])[quart == i].sum()) for i in range(4)]
        quart_rows.append(qr)
    out = dict(signature=SIGNATURE, dataset=name, split=s, seed=seed, n_train=len(tr), n_test=len(te), events_test=int(yte.sum()),
               reproduction_max_abs_diff=check, sizes_bytes=sizes, rows=rows, quartiles=quart_rows, seconds=round(time.time() - t0, 1))
    tmp = path.with_suffix('.part'); tmp.write_text(json.dumps(out, default=float)); tmp.replace(path)
    return f"saved in {out['seconds']} s (reference reproduced: max |Δscore| = {check['parent']:.1e})"

for name in RUN_OF:
    for s in range(NSPLITS_OF[name]):
        print(f'{name} split {s+1:02d}/{NSPLITS_OF[name]}: {gbm_split(name, s)}', flush=True)
print('PART A COMPLETE ->', OUT / 'splits')

recs = [json.loads(p.read_text()) for p in sorted((OUT / 'splits').glob('*_split??.json'))]
recs = [r for r in recs if r.get('signature') == SIGNATURE]
assert recs, 'No completed split yet.'

# Reproduction of the original runs
rep = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], **r['reproduction_max_abs_diff']) for r in recs])
worst = rep.drop(columns=['dataset', 'split']).max()
print('REPRODUCTION CHECK (max |score difference| vs saved runs):'); print(worst.to_string())
print('=> identical models: the new levels are directly comparable with the article' if worst.max() < 1e-6 else
      '=> models NOT bit-identical: all comparisons below use the rebuilt reference and rebuilt retrained copies consistently')
rep.to_csv(OUT / 'reproduction_check.csv', index=False)

# Storage (relative to the full 400-round model, same measure as the article: truncated text model / full pickled model)
sz = pd.DataFrame([dict(dataset=r['dataset'], **r['sizes_bytes']) for r in recs if r['split'] == 0 and r['sizes_bytes']])
for k in LEVELS: sz[f'rel_{k}'] = sz[f'trunc_{k}_text'] / sz['parent_pickle']
sz.to_csv(OUT / 'storage_relative.csv', index=False)
print('\nSTORAGE relative to the 400-round reference (article measure):')
display(sz.set_index('dataset')[[f'rel_{k}' for k in LEVELS]].mul(100).round(1))

# Per-dataset excess (Nadeau–Bengio)
dd = pd.DataFrame([dict(dataset=r['dataset'], split=r['split'], events_test=r['events_test'], n_test=r['n_test'], n_train=r['n_train'], **x)
                   for r in recs for x in r['rows']])
dd['excess_missed_per100'] = 100 * dd.excess_newly_missed / dd.events_test
dd['excess_net_per100'] = 100 * dd.excess_net_tp_lost / dd.events_test
dd.to_csv(OUT / 'excess_per_split.csv', index=False)
rows = []
for (ds, b, k), g in dd.groupby(['dataset', 'budget', 'rounds']):
    r = dict(dataset=ds, budget=b, rounds=k, splits=len(g), share_splits_gt0=float((g.excess_newly_missed > 0).mean()))
    for col in ['excess_missed_per100', 'excess_net_per100', 'excess_delta_mcc']:
        m, lo, hi = nadeau_bengio(g[col], g.n_test.mean(), g.n_train.mean()); r[col] = m; r[col + '_lo'] = lo; r[col + '_hi'] = hi
    rows.append(r)
PER = pd.DataFrame(rows); PER.to_csv(OUT / 'SUMMARY_per_dataset.csv', index=False)

# Random-effects meta-analysis per level and budget
meta_rows = []
for (b, k), e in PER.groupby(['budget', 'rounds']):
    e = e.dropna(subset=['excess_missed_per100_lo'])
    if len(e) < 1: continue
    out = dict(budget=b, rounds=k, k=len(e), datasets_gt0=int((e.excess_missed_per100 > 0).sum()))
    out['sign_test_p'] = float(stats.binomtest(out['datasets_gt0'], len(e), 0.5, alternative='greater').pvalue)
    for col, name in [('excess_missed_per100', 'missed'), ('excess_net_per100', 'net'), ('excess_delta_mcc', 'dmcc')]:
        tq = stats.t.ppf(0.975, e.splits - 1); se = np.maximum((e[col + '_hi'] - e[col + '_lo']) / (2 * tq), 1e-9)
        r = dersimonian_laird(e[col], se)
        out.update({f'{name}_{a}': r[a] for a in ['pooled', 'ci_low', 'ci_high', 'pi_low', 'pi_high', 'I2']})
    meta_rows.append(out)
META = pd.DataFrame(meta_rows)
REF_DS = 'diabetes_130' if 'diabetes_130' in set(sz.dataset) else (sz.dataset.iloc[0] if len(sz) else None)
if REF_DS is not None and len(META):
    META['storage_diabetes130_pct'] = META.rounds.map(lambda k: 100 * float(sz.set_index('dataset').loc[REF_DS, f'rel_{k}']))
    print('Storage reference dataset:', REF_DS)
META.to_csv(OUT / 'META_gbm_levels.csv', index=False)
f = lambda a, b, c, d=2: f'{a:+.{d}f} [{b:+.{d}f}, {c:+.{d}f}]'
view = META.assign(excess_missed=[f(*v) for v in zip(META.missed_pooled, META.missed_ci_low, META.missed_ci_high)],
                   PI=[f'[{a:+.2f}, {b:+.2f}]' for a, b in zip(META.missed_pi_low, META.missed_pi_high)],
                   net_lost=[f(*v) for v in zip(META.net_pooled, META.net_ci_low, META.net_ci_high)],
                   dMCC=[f(*v, 3) for v in zip(META.dmcc_pooled, META.dmcc_ci_low, META.dmcc_ci_high)],
                   I2=(100 * META.missed_I2).round(0))
print('\nMETA-ANALYSIS · GBM truncation · excess over retraining per 100 events (DerSimonian–Laird)')
display(view[['budget', 'rounds'] + (['storage_diabetes130_pct'] if 'storage_diabetes130_pct' in view else []) +
             ['k', 'excess_missed', 'PI', 'I2', 'datasets_gt0', 'sign_test_p', 'net_lost', 'dMCC']].round(3))

# Margin quartiles (same analysis as Figure 3c for the forest), pooled over datasets and splits
Q = []
for r in recs:
    for q in r['quartiles']:
        Q.append(dict(dataset=r['dataset'], budget=q['budget'], **{f'n_q{i+1}': q['n'][i] for i in range(4)},
                      **{f'retrain_q{i+1}': q['lost_retrain_mean'][i] for i in range(4)},
                      **{f'trunc{k}_q{i+1}': q[f'lost_trunc_{k}'][i] for k in LEVELS for i in range(4)}))
Q = pd.DataFrame(Q); Q.to_csv(OUT / 'quartiles_per_split.csv', index=False)
qp = Q[Q.budget == PRIMARY_BUDGET].sum(numeric_only=True)
QT = pd.DataFrame({lab: [100 * qp[f'{c}_q{i+1}'] / qp[f'n_q{i+1}'] for i in range(4)]
                   for lab, c in [('retrained copies (mean of 3)', 'retrain')] + [(f'truncated to {k}', f'trunc{k}') for k in LEVELS]},
                  index=['Q1 closest to threshold', 'Q2', 'Q3', 'Q4 furthest above']).round(1)
QT.to_csv(OUT / 'quartiles_pooled_primary.csv')
print('\nTRUE-POSITIVE ALERTS LOST (%) by quartile of the reference margin · primary budget')
print('(forest, article: retrained 35/12/3/1 %, thinned 46/27/14/6 %)'); display(QT)

import matplotlib.pyplot as plt
m = META[META.budget == PRIMARY_BUDGET].sort_values('rounds', ascending=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
a1.errorbar(m.rounds, m.missed_pooled, yerr=[m.missed_pooled - m.missed_ci_low, m.missed_ci_high - m.missed_pooled],
            marker='o', color='#0072B2', capsize=3, label='GBM truncation (pooled, 95% CI)')
a1.axhline(0, color='#555555', ls='--', lw=0.8)
a1.axhline(2.75, color='#D55E00', ls=':', lw=1); a1.text(m.rounds.max(), 2.85, 'RF 128→16 thinning (12.4% storage): +2.75', color='#D55E00', fontsize=7, ha='right')
a1.set_xscale('log', base=2); a1.set_xticks(sorted(m.rounds)); a1.set_xticklabels([str(int(k)) for k in sorted(m.rounds)]); a1.invert_xaxis()
if 'storage_diabetes130_pct' in m:
    top = a1.secondary_xaxis('top'); top.set_xticks(sorted(m.rounds)); top.set_xticklabels([f'{v:.1f}%' for v in m.sort_values('rounds').storage_diabetes130_pct])
    top.set_xlabel('Storage relative to the reference', fontsize=8)
a1.set_xlabel('Boosting rounds kept from the 400-round reference'); a1.set_ylabel('Excess newly missed positives per 100 events')
x = np.arange(4); w = 0.8 / (1 + len(LEVELS)); cols = ['#BBBBBB', '#56B4E9', '#0072B2', '#009E73', '#E69F00', '#D55E00']
for j, c in enumerate(QT.columns):
    a2.bar(x + (j - len(QT.columns) / 2 + 0.5) * w, QT[c], w, color=cols[j % len(cols)], label=c)
a2.set_xticks(x); a2.set_xticklabels(['Q1\nclosest', 'Q2', 'Q3', 'Q4\nfurthest']); a2.set_ylabel('True-positive alerts lost (%)')
a2.legend(frameon=False, fontsize=7)
for ax in (a1, a2):
    ax.grid(color='#E6E6E6', lw=0.5); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(OUT / 'figures' / f'fig_gbm_equal_storage.{ext}', dpi=200, facecolor='white')
plt.show(); plt.close(fig)

manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG, signature=SIGNATURE, python=sys.version.split()[0],
                sklearn=sklearn.__version__, lightgbm=lgb.__version__, numpy=np.__version__, platform=platform.platform(),
                reproduction_worst=worst.to_dict())
(OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, default=str))
share = PROJECT_HOME / 'results' / (RUN_NAME + '_SHAREABLE.zip')
with zipfile.ZipFile(share, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ['MANIFEST.json', 'PROTOCOL_LOCK.json', 'reproduction_check.csv', 'storage_relative.csv', 'SUMMARY_per_dataset.csv',
              'META_gbm_levels.csv', 'excess_per_split.csv', 'quartiles_per_split.csv', 'quartiles_pooled_primary.csv']:
        if (OUT / f).exists(): zf.write(OUT / f, arcname=f)
    for f in sorted((OUT / 'figures').glob('*')): zf.write(f, arcname='figures/' + f.name)
print('SHAREABLE ZIP ->', share)