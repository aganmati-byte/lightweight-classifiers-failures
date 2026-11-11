# Data

No raw data are stored in this repository: the datasets are public, but their licences do not
always allow redistribution. `bin/notebooks/01_data_acquisition.ipynb` (and the loaders inside the
other notebooks) download them automatically into Google Drive (`MyDrive/CHICCO_MEGA_RESEARCH/data/raw`).

| Dataset (name in the article) | Source | Outcome used | Reference |
|---|---|---|---|
| Sepsis survival | UCI Machine Learning Repository, dataset 827 | death during hospital stay | Chicco & Jurman 2020, https://doi.org/10.1038/s41598-020-73558-3 |
| Diabetes 130 | UCI Machine Learning Repository, dataset 296, https://doi.org/10.24432/C5230J | readmission within 30 days | Strack et al. 2014, https://doi.org/10.1155/2014/781670 |
| PhysioNet 2019 A→B | PhysioNet/CinC Challenge 2019, training sets A and B (public S3 mirror `physionet-open`) | sepsis label at ICU hour 12; trained on A, tested on B | Reyna et al. 2020, https://doi.org/10.1097/CCM.0000000000004145 |
| SUPPORT2 | UCI Machine Learning Repository, dataset 880 | in-hospital death (`hospdead`) | Knaus et al. 1995, https://doi.org/10.7326/0003-4819-122-3-199502010-00007 |
| PhysioNet 2012 | PhysioNet/CinC Challenge 2012, sets a and b | in-hospital death | Silva et al. 2012, http://hdl.handle.net/1721.1/93166 |
| Sepsis/SIRS ICU | Figshare (PLOS ONE supporting information) | `Mortality` | Gucyetmez & Atalan 2016, https://doi.org/10.1371/journal.pone.0148699 |
| CKD (UAE) | Figshare (PLOS ONE supporting information) | `EventCKD35` | Al-Shamsi et al. 2018, https://doi.org/10.1371/journal.pone.0199920 |
| Depression & heart failure | Figshare (PLOS ONE supporting information) | `Death (1=Yes, 0=No)` | Jani et al. 2016, https://doi.org/10.1371/journal.pone.0158570 |
| Cardiac arrest (Spain) | Figshare (PLOS ONE supporting information) | `Exitus` | Requena-Morales et al. 2017, https://doi.org/10.1371/journal.pone.0175818 |
| Heart failure | Figshare (PLOS ONE supporting information) | `DEATH_EVENT` | Ahmad et al. 2017, https://doi.org/10.1371/journal.pone.0181001 |
| Lung cancer ICU | Figshare (PLOS ONE supporting information) | `Hospital_death` | Song et al. 2019, https://doi.org/10.1371/journal.pone.0210951 |

The predictors kept and excluded for the six small cohorts (and the reason for each exclusion)
are listed in `results/2026-11-10_R3_tierB_small_cohorts/TARGETS_USED.md`.

**Known inconsistency.** The public lung cancer ICU file contains 160 hospital deaths among
264 patients, whereas the original article reports 188. We used the public file as released.

Two further cohorts shared by Davide Chicco (glioblastoma, n = 60; sepsis ICU 90-day, n = 123)
were excluded because they have fewer than 250 patients or fewer than 50 events in the
minority class.
