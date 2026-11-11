import json, sys
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# Lightweight Classifiers — **Data acquisition & inventory** (datasets from Davide Chicco's articles)

**En français.** Ce notebook télécharge automatiquement les datasets cliniques publics cités dans les articles de Davide (Figshare, UCI, CRAN), les range sur Drive selon la structure qu'il recommande (`data/raw/davide/<dataset>/`), puis produit un **inventaire** : pour chaque fichier, ses dimensions, ses colonnes, les valeurs manquantes et les **colonnes binaires candidates comme cible**. L'inventaire ne contient **aucune donnée individuelle** : envoyez le ZIP `DATA_INVENTORY.zip` pour définir les cibles et ajouter les datasets à l'analyse.

Si un téléchargement échoue, déposez le fichier à la main dans `data/raw/davide/<nom>/` et relancez la dernière cellule.
""")
code(r"""
import os, sys, io, json, re, zipfile, tarfile, subprocess, urllib.request, urllib.parse
from pathlib import Path
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive'); PROJECT_HOME = Path('/content/drive/MyDrive/CHICCO_MEGA_RESEARCH')
except ImportError:
    PROJECT_HOME = Path(os.environ['CHICCO_LOCAL_DRIVE_ROOT'])
DAV = PROJECT_HOME / 'data' / 'raw' / 'davide'; DAV.mkdir(parents=True, exist_ok=True)
DOC = PROJECT_HOME / 'doc'; DOC.mkdir(parents=True, exist_ok=True)
for pkg in ['openpyxl', 'xlrd', 'pyreadr', 'ucimlrepo']:
    try: __import__(pkg)
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', pkg], check=False)
UA = {'User-Agent': 'Mozilla/5.0 (research notebook; Chicco-Ganmati project)'}
def http(url, data=None, headers=None):
    h = dict(UA, **(headers or {}))
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=180) as r: return r.read()
print('Datasets folder:', DAV)
""")

md("""
### 1 · Sources (from Davide's articles and the links he shared on 10 November)
`figshare_ids`: articles shared by Davide. `figshare_titles`: supporting-information datasets of the original PLOS ONE articles, found by title through the public Figshare API (no password).
""")
code(r"""
SOURCES = {
  # name: ('figshare_id', id) | ('figshare_title', exact article title) | ('uci', id) | ('cran', package)
  'cardiac_arrest_spain':      ('figshare_id', 4876247),    # Requena-Morales et al. 2017, PLOS ONE
  'depression_heart_failure':  ('figshare_id', 3916224),    # Jani et al. 2016, PLOS ONE
  'lung_cancer_icu':           ('figshare_id', 7713893),    # Song et al. 2019, PLOS ONE
  'clinical_data_14201600':    ('figshare_id', 14201600),   # shared by Davide (to identify)
  'sepsis_icu_90day':          ('figshare_id', 28873667),   # ICU sepsis, 90-day mortality (to confirm: Gao et al. 2024?)
  'sepsis_sirs':               ('figshare_title', 'C-Reactive Protein and Hemogram Parameters for the Non-Sepsis Systemic Inflammatory Response Syndrome and Sepsis: What Do They Mean?'),
  'ckd_uae':                   ('figshare_title', 'Chronic kidney disease in patients at high risk of cardiovascular disease in the United Arab Emirates: A population-based study'),
  'mesothelioma':              ('uci', 351),                # Er et al. 2012; Chicco & Rovelli 2019
  'heart_failure_chicco':      ('uci', 519),                # Chicco & Jurman 2020
  'glioblastoma_ehrs':         ('cran', 'glioblastomaEHRsData'),  # Tainan / Munich / Berendsen (small)
}
MANUAL = {'neuroblastoma_shanghai': 'PeerJ 6:e5665 (Ma et al. 2018) – download the Supplemental Information files from https://peerj.com/articles/5665/'}
""")

md("### 2 · Téléchargement (relancer sans risque : les fichiers déjà présents sont conservés)")
code(r"""
def figshare_article(aid):
    return json.loads(http(f'https://api.figshare.com/v2/articles/{aid}'))

def figshare_search(title):
    body = json.dumps({'search_for': f'"{title}"', 'page_size': 20}).encode()
    hits = json.loads(http('https://api.figshare.com/v2/articles/search', data=body, headers={'Content-Type': 'application/json'}))
    norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
    return [h for h in hits if norm(title)[:60] in norm(h.get('title', ''))] or hits[:3]

def save_files(art, folder):
    folder.mkdir(parents=True, exist_ok=True); got = []
    meta = {k: art.get(k) for k in ('id', 'title', 'doi', 'url_public_html', 'resource_doi', 'resource_title', 'license')}
    (folder / f"figshare_{art['id']}.json").write_text(json.dumps(meta, indent=2, default=str))
    for f in art.get('files', []):
        dest = folder / f['name']
        if not dest.exists():
            dest.write_bytes(http(f['download_url']))
        got.append(dest.name)
    return got

def fetch(name, kind, ref):
    folder = DAV / name
    if folder.exists() and any(p.suffix.lower() in {'.csv', '.xlsx', '.xls', '.txt', '.tsv', '.rda', '.rdata', '.sav', '.dta'} for p in folder.rglob('*')):
        return 'already present'
    if kind == 'figshare_id':
        return save_files(figshare_article(ref), folder)
    if kind == 'figshare_title':
        got = []
        for h in figshare_search(ref):
            got += save_files(figshare_article(h['id']), folder)
        return got or 'no figshare match'
    if kind == 'uci':
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=ref); folder.mkdir(parents=True, exist_ok=True)
        df = d.data.original if getattr(d.data, 'original', None) is not None else pd.concat([d.data.features, d.data.targets], axis=1)
        df.to_csv(folder / f'uci_{ref}.csv', index=False); return [f'uci_{ref}.csv']
    if kind == 'cran':
        idx = http(f'https://cloud.r-project.org/web/packages/{ref}/index.html').decode('utf-8', 'ignore')
        tgz = re.search(rf'{ref}_[0-9.\-]+\.tar\.gz', idx).group(0)
        blob = http(f'https://cloud.r-project.org/src/contrib/{tgz}'); folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.lower().endswith(('.rda', '.rdata', '.csv')):
                    (folder / Path(m.name).name).write_bytes(t.extractfile(m).read())
        return [p.name for p in folder.iterdir()]

status = {}
for name, (kind, ref) in SOURCES.items():
    try:
        status[name] = fetch(name, kind, ref)
    except Exception as e:
        status[name] = f'FAILED: {type(e).__name__}: {e}'
    print(f'{name:28s} -> {status[name]}')
for name, how in MANUAL.items():
    (DAV / name).mkdir(parents=True, exist_ok=True)
    print(f'{name:28s} -> MANUAL: {how}  (put files in {DAV / name})')
""")

md("### 3 · Inventaire (aucune donnée individuelle exportée)")
code(r"""
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

inventory = []
for folder in sorted(d for d in DAV.iterdir() if d.is_dir()):
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or '_unzipped' in p.parent.name and p.suffix.lower() == '.zip': continue
        for key, df in read_tables(p).items():
            if not isinstance(df, pd.DataFrame) or df.empty: continue
            cols = []
            for c in df.columns:
                v = df[c]; nun = int(v.nunique(dropna=True))
                info = dict(column=str(c), dtype=str(v.dtype), unique=nun, missing_pct=round(100 * float(v.isna().mean()), 1))
                if nun == 2:
                    info['binary_counts'] = {str(k): int(n) for k, n in v.value_counts(dropna=True).items()}
                elif pd.api.types.is_numeric_dtype(v) and nun > 2:
                    info['range'] = [float(np.nanmin(v)), float(np.nanmax(v))]
                elif nun <= 12:
                    info['levels'] = [str(x) for x in v.dropna().unique()[:12]]
                cols.append(info)
            inventory.append(dict(dataset=folder.name, table=key, rows=int(df.shape[0]), columns=int(df.shape[1]), schema=cols))

lines = ['# Data inventory (no individual-level data)\n']
for it in inventory:
    bins = [c['column'] + ' ' + str(c['binary_counts']) for c in it['schema'] if 'binary_counts' in c]
    lines.append(f"## {it['dataset']} — {it['table']}\n- rows: {it['rows']}, columns: {it['columns']}\n"
                 f"- binary candidate targets: {', '.join(bins) if bins else 'none'}\n"
                 f"- columns: {', '.join(c['column'] + ' (' + c['dtype'] + ')' for c in it['schema'])}\n")
(DOC / 'DATA_INVENTORY.md').write_text('\n'.join(lines))
(DOC / 'data_inventory.json').write_text(json.dumps(dict(status={k: (v if isinstance(v, str) else list(v)) for k, v in status.items()},
                                                          inventory=inventory), indent=1, default=str))
with zipfile.ZipFile(DOC / 'DATA_INVENTORY.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(DOC / 'DATA_INVENTORY.md', 'DATA_INVENTORY.md'); z.write(DOC / 'data_inventory.json', 'data_inventory.json')
print('\n'.join(lines)[:6000])
print('\nSend this file:', DOC / 'DATA_INVENTORY.zip')
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "cells": []}
for i, (k, s) in enumerate(cells):
    c = {"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True), "id": f"d{i:02d}"}
    if k == "code": c.update(outputs=[], execution_count=None)
    nb["cells"].append(c)
json.dump(nb, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
open(sys.argv[2], "w").write("\n\n".join(s for k, s in cells if k == "code"))
print("cells", len(cells))
