"""Builds CHICCO_EXCESS_EXTENSION.ipynb, reusing code blocks of the replication notebook builder."""
import json, sys, os, tempfile

# --- recover the code cells of the replication notebook (same functions => same models, same splits)
ns = {}
_argv = sys.argv
tmp = tempfile.mkdtemp()
sys.argv = ['x', os.path.join(tmp, 'a.ipynb'), os.path.join(tmp, 'a.py')]
exec(open('/home/claude/nb/build_nb.py').read(), ns)
sys.argv = _argv
base_code = [src for kind, src in ns['cells'] if kind == 'code']
DATA_FUNCS = base_code[1].split('LOADERS = {')[0]
MODELS = base_code[2]
METRICS = base_code[3]
RUN_SPLIT = base_code[4].rsplit('\nfor name in DATASETS:', 1)[0].replace(
    'tr, va, te = grouped_split(y, groups, seed)', 'tr, va, te = SPLITTERS.get(name, grouped_split)(y, groups, seed)')
assert 'SPLITTERS.get' in RUN_SPLIT
AGG = base_code[5]

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Lightweight Classifiers and Their Failures — **Extension notebook**
## Three more EHR tasks · dose–response · five-dataset summary — **without retraining the first analysis**

**Abdelilah Ganmati — collaboration with Dr. Davide Chicco · 10 November 2026**

**En français.** Ce notebook complète la réplication `EXCESS_REPLICATION_20261110_R1` déjà terminée. **Rien n'est ré-entraîné pour Diabetes 130 et PhysioNet 2012** : les scores sauvegardés sur Drive sont réutilisés.
1. **Partie A** : même protocole (30 partitions, 3 ré-entraînements, RF, GBM, LR) sur trois nouvelles tâches : **SUPPORT2** (décès à l'hôpital, 5 centres américains), **Sepsis Survival** (Chicco & Jurman 2020, Norvège) et **PhysioNet 2019, hôpital A → hôpital B** (sepsis, validation externe).
2. **Partie B — dose–réponse** : on garde 64, 32, 16, 8 puis 4 arbres de la forêt parent, sur les **5 datasets**. Seule la forêt parent est reconstruite (même graine, donc la même forêt, vérifié automatiquement) ; les copies ré-entraînées sont relues depuis Drive.
3. **Partie C — synthèse sur 5 datasets** : critère principal, toutes les métriques demandées par Davide (MCC, F1, accuracy, sensibilité, spécificité, précision, NPV, Brier, ROC AUC, PR AUC), figures.

**Run (Colab):** upload → *Runtime → Run all* → authorise Drive. Start with `QUICK_TEST = True` (uses the quick-test base run), then `False`. Everything is checkpointed on Drive and resumes after a disconnection.

| Pre-specified addition | Definition |
|---|---|
| Primary endpoint (new tasks) | Same as the replication: excess newly missed positives per 100 events, RF 128→16, 10 % budget, Nadeau–Bengio 95 % CI |
| Dose–response hypothesis | Excess grows as fewer trees are kept; the tree-sampling model predicts the observed number of missed patients at every size |
| Caveats declared now | Sepsis Survival has no patient identifier (episodes of one patient may fall in train and test); PhysioNet 2019 A→B uses a fixed external test set, so its NB interval is conservative and its alert rate on B differs from 10 % |
""")

md("### 1 · Configuration, Drive, liens avec l'analyse précédente")
code(r"""
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
""")

md("""
### 2 · Données
- **Anciennes** (Diabetes 130, PhysioNet 2012) : mêmes chargeurs que la réplication, données en cache sur Drive.
- **SUPPORT2** (UCI 880) : prédicteurs **déclarés à l'avance**, mesurés à l'entrée ou au 3e jour de l'étude ; toutes les variables de suivi, de coût, de pronostic ou de résultat sont exclues.
- **Sepsis Survival** (UCI 827, Chicco & Jurman 2020) : âge, sexe, numéro d'épisode ; cible = décès. ⚠️ Aucun identifiant patient : les partitions sont faites par épisode.
- **PhysioNet 2019 A→B** : entraînement et validation sur l'hôpital A, **test toujours sur l'hôpital B** ; même préparation « heure 12 » que les notebooks précédents (cache `data/v3_prepared` réutilisé).
""")
code(DATA_FUNCS + r"""

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
""")

md("### 3 · Modèles et mesures (identiques à la réplication)")
code("from sklearn.model_selection import GroupShuffleSplit\n" + MODELS + "\nSPLITTERS = {'physionet_2019_AtoB': split_A_to_B}")
code(METRICS)

md("""
### 4 · Partie A — trois nouvelles tâches, même protocole
Checkpoint sur Drive après chaque partition ; relancer reprend automatiquement.
""")
code(RUN_SPLIT + r"""

for name in NEW_DATASETS:
    for s in range(N_SPLITS):
        status = run_split(name, s)
        print(f'{name} split {s+1:02d}/{N_SPLITS}: {status}', flush=True)
print('PART A COMPLETE ->', OUT / 'splits')
""")

md("### 5 · Résumé de la Partie A (mêmes calculs que la réplication)")
code(AGG)

md("""
### 6 · Partie B — dose–réponse sur les 5 datasets (seule la forêt parent est reconstruite)
Pour chaque partition : on reconstruit la forêt parent (même graine) et on **vérifie** qu'elle redonne exactement les scores sauvegardés. Les décisions des copies ré-entraînées sont relues depuis Drive (scores de test et seuils de validation sauvegardés). On garde ensuite les k premiers arbres, pour k = 64, 32, 16, 8, 4.
""")
code(r"""
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
""")

md("### 7 · Résumé de la dose–réponse")
code(r"""
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
""")

md("### 8 · Partie C — synthèse sur les 5 datasets et figures")
code(r"""
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
""")

md("### 9 · Manifeste et archive partageable (sans données individuelles)")
code(r"""
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
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"e{i:02d}"}
    if kind == "code": c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
open(sys.argv[2], "w").write("\n\n".join(src for kind, src in cells if kind == "code"))
print("cells:", len(cells))
