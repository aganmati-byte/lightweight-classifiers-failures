# Documentation

The manuscript is written in the authors' shared Overleaf project and is not included here.

## Protocol (summary)

* **Splits.** Patient-grouped train/validation/test partitions (65/15/20%), 30 splits for the five
  large datasets and 50 for the six small cohorts; seed of split *s* = 1000 + 37*s*.
  PhysioNet 2019: models trained and validated on hospital system A, tested on hospital system B.
* **Classifiers.** Random forest with 128 trees (reference); the same forest thinned after training
  to its first *k* trees (*k* = 64, 32, 16, 8, 4); a 16-tree forest trained directly;
  LightGBM with 400 rounds (reference); the same model truncated after training to its first
  200, 100, 50, 25 or 10 rounds; a small LightGBM model trained directly; logistic regression.
* **Alerts.** Each classifier alerts the patients whose score exceeds its own threshold, set on the
  validation set so that 10% (primary) or 20% (sensitivity analysis) of patients are alerted.
* **Retraining noise.** Three copies of each reference retrained with different seeds
  (seed + 100*r*, *r* = 1, 2, 3) on the same training data.
* **Excess.** Positives detected by the reference but missed by the compressed classifier, minus
  the mean of the same count over the three retrained copies, per 100 positive test patients;
  also net true positives lost and change in MCC.
* **Inference.** Patient-cluster bootstrap within each split (500 draws); Nadeau–Bengio corrected
  confidence intervals across splits; DerSimonian–Laird random-effects meta-analysis across
  datasets, with 95% prediction interval, I², one-sided sign test and leave-one-out analysis.
* **Mechanism.** Thinning modelled as sampling *k* of *M* trees without replacement: the thinned
  score has mean equal to the reference score and variance σ²/k · (M−k)/(M−1), where σ² is the
  variance of the per-tree scores; the probability of falling below the threshold follows from
  the normal approximation.

The exact settings of every run are in the `PROTOCOL_LOCK.json` / `PROTOCOL.json` file of its
results folder.

## Reference for the repository layout

W. S. Noble (2009). A quick guide to organizing computational biology projects.
PLOS Computational Biology 5(7):e1000424. https://doi.org/10.1371/journal.pcbi.1000424
