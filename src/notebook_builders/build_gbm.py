"""Builds CHICCO_GBM_EQUAL_STORAGE.ipynb: GBM truncation at several storage levels (incl. ~12% = RF thinning) on all 11 datasets."""
import json, sys

EXT = open('/home/claude/nb/ext_flat.py').read()
TIB = open('/home/claude/nb/tierb_flat.py').read()

def between(src, start, end):
    i = src.index(start); j = src.index(end, i)
    return src[i:j].rstrip() + '\n'

def func(src, name):
    i = src.index(f'\ndef {name}(') + 1
    j = i
    lines = src[i:].split('\n')
    out = [lines[0]]
    for ln in lines[1:]:
        if ln and not ln[0].isspace() and not ln.startswith(')'): break
        out.append(ln)
    return '\n'.join(out).rstrip() + '\n'

LOADERS_LARGE = between(EXT, 'def download(url, path):', 'LOADERS = {')
TIERB_BUILD = between(TIB, 'def read_tables(p):', 'DATA, REPORT = {}, []')
CORE = ''.join(func(EXT, n) + '\n' for n in ['grouped_split', 'make_preprocessor', 'fit_gbm', 'gbm_scores', 'serialized_bytes',
                                            'counts', 'mcc_from', 'change_vectors', 'excess_comparison', 'nadeau_bengio'])
DL = func(TIB, 'dersimonian_laird')

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Lightweight Classifiers — **GBM truncation at equal storage** (complementary experiment)

**But.** Dans l'article, la troncature du GBM (400 → 100 rondes) garde **25,3 %** du stockage, alors que l'élagage de la forêt (128 → 16 arbres) n'en garde que **12,4 %**. Ce notebook teste la troncature du GBM à **plusieurs niveaux de stockage, dont ≈ 12 % (400 → 50)**, sur **les 11 datasets**, avec exactement le même protocole (mêmes partitions, mêmes graines, mêmes copies ré-entraînées, mêmes budgets d'alerte).

**Ce qu'il calcule**
| Analyse | Détail |
|---|---|
| Dose–réponse GBM | troncature à 200, 100, 50, 25 et 10 rondes ; excès de patients oubliés au-delà du ré-entraînement, IC de Nadeau–Bengio, méta-analyse DerSimonian–Laird |
| Stockage | taille de chaque modèle tronqué par rapport au modèle complet (même mesure que dans l'article) |
| Quartiles | part des alertes vraies perdues par quartile de marge au-dessus du seuil (comme la Figure 3c pour la forêt) |
| Contrôle | le GBM de référence et ses 3 copies sont ré-entraînés avec les mêmes graines ; l'écart avec les scores sauvegardés est affiché |

**Rien n'est écrasé** : les runs R1, R2 et R3 sont seulement **lus** ; tout est écrit dans un nouveau dossier `results/GBM_EQUAL_STORAGE_…`. Reprise automatique si Colab s'interrompt.

**Exécution (Colab)** : *Exécution → Tout exécuter* avec `QUICK_TEST = True` (≈ 5 min), puis `QUICK_TEST = False` et *Redémarrer et tout exécuter* (≈ 1 h).
""")

code(r"""
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
""")

md(r"""
## 1 · Données (mêmes chargeurs et mêmes cibles que les runs d'origine, lus depuis le Drive)
""")
code(LOADERS_LARGE + r"""
from sklearn.model_selection import GroupShuffleSplit

""" + TIERB_BUILD + r"""
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
""")

md(r"""
## 2 · Modèles et mesures (code identique aux runs d'origine)
""")
code(r"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

""" + CORE + DL + r"""
SPLITTERS = {'physionet_2019_AtoB': split_A_to_B}
""")

md(r"""
## 3 · Troncature du GBM à chaque niveau, sur chaque partition (reprise automatique)
""")
code(r"""
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
""")

md(r"""
## 4 · Résultats : stockage, excès par dataset, méta-analyse, quartiles
""")
code(r"""
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
        tq = stats.t.ppf(0.975, e.splits - 1); se = (e[col + '_hi'] - e[col + '_lo']) / (2 * tq); se = se.where(se > 0, se[se > 0].min())
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
""")

md(r"""
## 5 · Figures et archive à partager
""")
code(r"""
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
""")

nb = dict(nbformat=4, nbformat_minor=5,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                        language_info=dict(name='python'), colab=dict(provenance=[])),
          cells=[dict(cell_type=k, metadata={}, source=s, **({'outputs': [], 'execution_count': None} if k == 'code' else {}))
                 for k, s in cells])
out_nb = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/nb/CHICCO_GBM_EQUAL_STORAGE.ipynb'
json.dump(nb, open(out_nb, 'w'), indent=1)
if len(sys.argv) > 2:
    open(sys.argv[2], 'w').write('\n\n'.join(s for k, s in cells if k == 'code'))
print('written', out_nb, len(cells), 'cells')
