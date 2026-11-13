# Lightweight supervised machine learning classifiers can miss data samples when applied to medical data

Code, protocols and results for the article *Lightweight supervised machine learning classifiers can miss data samples when applied to medical data*
(Abdelilah Ganmati, Davide Chicco; manuscript in preparation).

**Question.** When a clinical classifier is made lighter (fewer trees, fewer boosting rounds),
does it miss patients that its full-size reference detected, beyond the disagreement that
simply retraining the reference would cause?

**Approach.** On 11 public binary clinical datasets, each compressed classifier is compared with
its reference on the same held-out patients. The *excess* of newly missed positive patients is
the number missed by the compressed model minus the mean number missed by three independently
retrained copies of the reference. Alerts use a fixed budget (10%, and 20% as sensitivity
analysis) with thresholds chosen on validation data only. Results are pooled over 30 or 50
patient-grouped splits per dataset (Nadeau–Bengio corrected intervals) and across datasets
(DerSimonian–Laird random effects).

## Repository structure

The layout follows W. S. Noble (2009), *A quick guide to organizing computational biology
projects*, PLOS Computational Biology 5(7):e1000424, https://doi.org/10.1371/journal.pcbi.1000424.

```
doc/        protocol notes and documentation (the manuscript itself is kept in Overleaf)
data/       no raw data (licences): README explaining how every dataset is obtained
src/        source code: notebook builders and the plain-Python code of every notebook
bin/        what is executed: the Google Colab notebooks (in run order) and the scripts
            that turn the run outputs into the figures and tables of the article
results/    one dated folder per experiment, with its outputs and the executed notebook
```

## How to reproduce

All experiments were run on Google Colab, with data and results stored on Google Drive under
`MyDrive/CHICCO_MEGA_RESEARCH/` (folders `data/raw`, `data/prepared`, `results/`).
Each notebook mounts the Drive, resumes automatically after an interruption, and refuses to
overwrite a run made with a different protocol.

| Step | Notebook (`bin/notebooks/`) | Output (`results/`) |
|---|---|---|
| 1 | `01_data_acquisition.ipynb` — downloads the public datasets | Drive `data/raw` |
| 2 | `02_excess_replication.ipynb` — PhysioNet 2012, Diabetes 130 | `2026-11-10_R1_excess_replication` |
| 3 | `03_excess_extension.ipynb` — SUPPORT2, Sepsis survival, PhysioNet 2019 A→B; dose–response; mechanism | `2026-11-10_R2_excess_extension` |
| 4 | `04_tierB_small_cohorts.ipynb` — six small cohorts (Figshare) | `2026-11-10_R3_tierB_small_cohorts` |
| 5 | `05_gbm_equal_storage.ipynb` — boosting truncation at 200/100/50/25/10 rounds | `2026-11-11_R4_gbm_equal_storage` |
| 6 | `06_dataset_table.ipynb` — samples, variables and missing values per dataset | `2026-11-11_dataset_table` |

Every notebook first runs with `QUICK_TEST = True` (a few splits, a few minutes) and then with
`QUICK_TEST = False` (the complete run, used in the article).

Figures and tables of the article are then rebuilt from the run outputs, on any computer:

```bash
pip install -r requirements.txt
python bin/combine.py      # gathers R1, R2 and R3 into results/2026-11-11_paper_assets/data
python bin/facts.py        # every number quoted in the text -> results/2026-11-11_paper_assets/facts.json
python bin/pub_figs.py     # Figures 1-3, S1, S2 (vector PDF) -> .../figures
python bin/pub_tables.py   # tables -> .../tables
```

`results/2026-11-11_paper_assets/tables_final/` holds the tables exactly as they appear in the
manuscript. They add to the generated tables the alternating row colours, the
“GBM 400→50, truncated” rows and Supplementary Table S5 (both taken from
`results/2026-11-11_R4_gbm_equal_storage/META_gbm_levels.csv`) and the dataset table
(from `results/2026-11-11_dataset_table/`).

## Map from the article to the files

| Article | Source |
|---|---|
| Table 1 (datasets) | `results/2026-11-11_dataset_table/` |
| Table 2 (pooled excess), Table 3 (per dataset), Figure 2 | R1 + R2 + R3, via `bin/combine.py`, `bin/pub_tables.py`, `bin/pub_figs.py` |
| Figure 1, Supplementary Table S1, Figure S1 (accuracy vs MCC, all metrics) | `metrics_per_split.csv` of R1, R2, R3 |
| Figure 3, Supplementary Table S4 (dose–response, mechanism) | R2 (`SUMMARY_dose_response.csv`, `mechanism_per_split.csv`) and `mechanism_per_split.csv` of R1, R3 |
| Supplementary Table S2 (20% budget), Table S3 / Figure S2 (all variants) | R1 + R2 + R3 |
| “GBM 400→50” rows, Supplementary Table S5 | `results/2026-11-11_R4_gbm_equal_storage/META_gbm_levels.csv` |

## Software

Python 3.13 (Google Colab), scikit-learn 1.6.1, LightGBM 4.6.0, NumPy 2.4, pandas, SciPy,
matplotlib. Exact versions are recorded in each `MANIFEST.json`.

## Licence

MIT License (see LICENSE).

## Contact

Davide Chicco (corresponding author) and Abdelilah Ganmati.
