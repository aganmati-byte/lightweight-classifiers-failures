import json, sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'results' / '2026-11-11_paper_assets'
from scipy import stats
RUNS = [ROOT / 'results' / d for d in ('2026-11-10_R1_excess_replication', '2026-11-10_R2_excess_extension', '2026-11-10_R3_tierB_small_cohorts')]
OUT = ASSETS / 'data'; OUT.mkdir(parents=True, exist_ok=True)
def cat(name):
    fr = [pd.read_csv(r / name).assign(run=r.parent.name) for r in RUNS if (r / name).exists()]
    return pd.concat(fr, ignore_index=True)
S = cat('SUMMARY_excess_over_retraining.csv').drop_duplicates(['dataset', 'budget', 'family', 'candidate'])
M = cat('metrics_per_split.csv'); K = cat('mechanism_per_split.csv'); C = cat('comparisons_per_split.csv')
meta = {}
for r in RUNS:
    for k, v in json.loads((r / 'MANIFEST.json').read_text()).get('data', {}).items(): meta.setdefault(k, v)
D = pd.read_csv(RUNS[1] / 'SUMMARY_dose_response.csv'); DS = pd.read_csv(RUNS[1] / 'dose_model_sizes.csv')
Z = pd.read_csv(RUNS[0] / 'model_sizes_first_split.csv')
for n, d in [('S', S), ('M', M), ('K', K), ('C', C), ('D', D), ('DS', DS), ('Z', Z)]: d.to_csv(OUT / f'{n}.csv', index=False)
(OUT / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
print('datasets:', sorted(S.dataset.unique()), len(S.dataset.unique()))
print('splits per dataset:', S.groupby('dataset').splits.max().to_dict())
