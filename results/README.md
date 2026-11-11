# Results

One dated folder per experiment (Noble 2009). Each folder contains the outputs downloaded from
Google Drive (`*_SHAREABLE.zip`), the protocol lock, a `MANIFEST.json` with software versions and,
when available, the executed notebook (`executed_notebook.ipynb`) as a record of what was run.

| Folder | What was done | Main observation |
|---|---|---|
| `2026-11-10_R1_excess_replication` | 30 patient-grouped splits; PhysioNet 2012, Diabetes 130; seven classifiers; 10% and 20% alert budgets | Forest thinning (128→16 trees) misses positives beyond retraining noise |
| `2026-11-10_R2_excess_extension` | Same protocol on SUPPORT2, Sepsis survival, PhysioNet 2019 A→B; forest dose–response (64…4 trees) on the five large datasets; tree-sampling mechanism | Excess grows as trees are removed; the tree-sampling model predicts the lost patients |
| `2026-11-10_R3_tierB_small_cohorts` | 50 splits on the six small Figshare cohorts; targets and predictors in `TARGETS_USED.md` | All point estimates of the forest excess are positive; intervals are wide |
| `2026-11-11_R4_gbm_equal_storage` | Boosting truncation to 200/100/50/25/10 rounds on all 11 datasets, same splits, seeds and retrained copies; reference models reproduced (max score difference 3e-8) | At equal storage (~12%), truncation misses 0.76 positives per 100 events, against 2.75 for forest thinning, with no net loss of true positives |
| `2026-11-11_dataset_table` | Samples, predictors and missing values per dataset, on the data used in the experiments | Source of Table 1 of the article |
| `2026-11-11_paper_assets` | Combined data, `facts.json` (every number quoted in the text), figures and tables, rebuilt by the scripts in `bin/` | `tables_final/` = tables as in the manuscript |

The executed notebook of the complete R1 run was not kept; the clean notebook in
`bin/notebooks/02_excess_replication.ipynb` reproduces it, and the R4 run re-trained the R1
reference models bit-for-bit (see `2026-11-11_R4_gbm_equal_storage/reproduction_check.csv`).
