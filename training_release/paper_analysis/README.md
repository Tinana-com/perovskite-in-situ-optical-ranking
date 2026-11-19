# Regional photovoltaic performance ranking from in situ optical monitoring

Code and archived analysis inputs accompanying Yutian Chen's study of within-group performance ranking in two public KIT perovskite optical-monitoring datasets. This release reproduces groupwise metrics and the four main and six supplementary figures from frozen predictions and measured trajectory excerpts. D1 uses substrate groups; D2 uses date isolation with experiment-level ranking and experiment-then-date averaging.

## Quick start

Use Python 3.12 with the supplied requirements:

```bash
python -m pip install -r requirements.txt
python src/evaluate_archived_scores.py
python src/reproduce_figures.py
```

Outputs appear in `REPRODUCED/`. The evaluator checks 40 archived model-scope comparisons against the stored three-metric means within 1e-8. No model training or data download is performed. See run_reproduce.md for the exact evaluation convention and results_manifest.md for file mappings.

`SOURCE_DATA/P6/reference_source/` retains the original feature/model definitions for traceability. These historical modules have project-specific dependencies and are not reproduction entrypoints. Their original training is outside the scope of the two commands above. Frozen model and feature configurations are supplied alongside them.

## Scientific scope

Development cohorts were repeatedly explored. Author-test substrates and historical dates had already been inspected; they are retrospective historical comparisons. All failed and zero-efficiency regions remain in the frozen cohorts. Top25% gain is selected-minus-group mean PCE in percentage points, not a change in device efficiency.

Figure1 is original Python vector artwork with a real ACD trajectory inset. Figures2–4 and SI reproduce the archived analyses. D2 optical features include three PL channels and ND reflection. The datasets are modeled separately.

## Data and license

Original datasets: https://doi.org/10.5281/zenodo.7503391 and https://doi.org/10.5281/zenodo.14609789. Raw H5 files and original publication images are not bundled. See data_availability.md. MIT applies to the analysis/plotting code; it does not replace the original data owners' rights or dependency licenses.

## Citation

Chen, Y. (2026). Regional photovoltaic performance ranking from in situ optical monitoring: analysis code, version 1.0. Cite the two original dataset records as well. No paper DOI or public repository URL has yet been assigned. CITATION.cff supplies software metadata.

Correspondence: Yutian Chen, 25113090040@m.fudan.edu.cn.
