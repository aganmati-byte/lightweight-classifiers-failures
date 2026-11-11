# Source code

* `notebook_builders/` — Python scripts that generate the Colab notebooks of `bin/notebooks/`
  (`python build_gbm.py <output.ipynb>` writes the notebook; the notebooks are therefore
  generated, never edited by hand).
* `notebook_code/` — the code cells of each notebook concatenated into one plain Python file,
  for reading, searching and reviewing the code without Jupyter.
