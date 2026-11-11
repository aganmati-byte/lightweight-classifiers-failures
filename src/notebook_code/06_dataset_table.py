import os, sys, io, re, json, time, zipfile, hashlib, pickle, platform, warnings, gc, urllib.request, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, scipy, sklearn
from scipy import stats
warnings.filterwarnings('ignore')

# ======================= USER CONFIGURATION =======================
QUICK_TEST = False
RUN_NAME   = 'DATASET_TABLE_20261111'
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
              gbm_backend=GBM_BACKEND.split()[0], protocol_version='2026-11-11_dataset_table_v1')
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

# ===== Dataset description table (Davide, 11 Nov) — run after sections 0 and 1 =====
INFO = [  # key, name in the article, reference key, release year, location
 ('sepsis_survival',          'Sepsis survival',              'chicco2020sepsis',            2020, 'Norway (nationwide)'),
 ('diabetes_130',             'Diabetes 130',                 'strack2014impact',            2014, '130 hospitals, USA'),
 ('physionet_2019_AtoB',      'PhysioNet 2019 A$\\rightarrow$B', 'reyna2020early',           2020, 'Boston, Massachusetts (A); Atlanta, Georgia (B); USA'),
 ('support2',                 'SUPPORT2',                     'knaus1995support',            1995, 'five medical centers, USA'),
 ('physionet_2012',           'PhysioNet 2012',               'silva2012predicting',         2012, 'Boston, Massachusetts, USA'),
 ('sepsis_sirs',              'Sepsis/SIRS ICU',              'gucyetmez2016c',              2016, 'Istanbul, Turkey'),
 ('ckd_uae',                  'CKD (UAE)',                    'alshamsi2018chronic',         2018, 'Al Ain, Abu Dhabi, UAE'),
 ('depression_heart_failure', 'Depression \\& heart failure', 'jani2016comorbid',            2016, 'Olmsted County area, Minnesota, USA'),
 ('cardiac_arrest_spain',     'Cardiac arrest (Spain)',       'requena2017mortality',        2017, 'Alicante, Valencian Community, Spain'),
 ('heart_failure_chicco',     'Heart failure',                'ahmad2017survival',           2017, 'Faisalabad, Punjab, Pakistan'),
 ('lung_cancer_icu',          'Lung cancer ICU',              'song2019effect',              2019, 'Seongnam, Gyeonggi, South Korea'),
]
th = lambda v: f'{v:,}'.replace(',', '{,}')
rows, check = [], []
for key, name, ref, year, place in INFO:
    if key not in DATA: print('MISSING dataset:', key); continue
    X, y, g, meta = DATA[key]
    n, p = X.shape
    miss = int(X.isna().sum().sum())
    check.append(dict(dataset=key, samples=n, variables=p, missing=miss, missing_pct=round(100 * miss / (n * p), 2), events=int(y.sum())))
    rows.append(f'{name} & {th(n)} & {p} & {th(miss)} ({100 * miss / (n * p):.1f}\\%) & \\cite{{{ref}}} & {year} & {place} \\\\')
display(pd.DataFrame(check))
latex = '\n'.join([
 '\\rowcolors{2}{white}{LightGreen}',
 '\\begin{table}[!htb]',
 '\\centering',
 '\\caption{\\textbf{Datasets.} \\#samples: patients or hospital stays. \\#variables: clinical variables used as predictors (outcome excluded). '
 '\\#missing elements: missing values among the predictors (percentage of all predictor values). Release year: year of publication of the reference article. '
 'PhysioNet 2019 A$\\rightarrow$B: models trained on hospital system A and tested on hospital system B.}',
 '\\label{tab:DATASETS}',
 '{\\small',
 '\\begin{tabularx}{\\textwidth}{lrrrcc>{\\raggedright\\arraybackslash}X}',
 '\\toprule',
 '\\rowcolor{white}Dataset name & \\#samples & \\#variables & \\shortstack{\\#missing\\\\elements} & Reference & \\shortstack{Release\\\\year} & City, region, country \\\\',
 '\\midrule',
 *rows,
 '\\bottomrule',
 '\\end{tabularx}',
 '}',
 '\\end{table}'])
print('\n========== COPY EVERYTHING BELOW INTO tables/tab0_datasets.tex ==========\n')
print(latex)

(OUT / 'tab0_datasets.tex').write_text(latex + '\n')
pd.DataFrame(check).to_csv(OUT / 'datasets_description.csv', index=False)
print('\nSaved ->', OUT / 'tab0_datasets.tex', 'and', OUT / 'datasets_description.csv')
