# Reproduction protocol

Follow README commands in order: prepare real data, inspect frozen plans, train each dataset separately, evaluate target-free OOF predictions and optionally predict from a saved artifact. Public entrypoints resolve repository-relative source/configuration paths; only raw data paths and fresh output paths are user arguments.

## Scientific invariants

Seed20261019 for P6/P7/P9; P8 retains its frozen seed20261020. D1 uses author foldNo0–4 by substrate,780 development regions/26 substrates and349 historical regions/12 substrates. D2 uses the seven frozen date folds,4707 development regions/158 experiments/14 dates and1150 historical regions/36 experiments/3 dates.

D1 uses LP725; D2 uses LP725/LP780/SP775 and ND reflection. Group normalization uses optical data only. D1 pairing excludes absolute target differences <=0.5 pp; D2 excludes differences <0.5 pp. Both orientations and original weighting are retained. P6 pairwise C1, no intercept, lbfgs, tolerance1e-8, max_iter2000 are unchanged. Other estimator parameters are in configs/frozen_p6_protocol.json. The P6 native configuration grid is copied unchanged; inherited controls are explicitly added by portable_models.configuration.

D1 simple-rule and author-kNN choices use outer-training author folds only. D2 intensity rule selection uses outer-training dates only. D2 inherited absolute F2 Ridge uses the original unweighted P2 alpha1 pipeline; it is distinct from P6 relative-feature, weighted Ridge alpha10. Historical targets do not select parameters.

Top25 k=ceil(n/4); ties use sample_id, Spearman uses average ranks with undefined constant cases omitted. D1 averages substrates equally; D2 averages experiments within dates, then dates equally. Descriptive bootstrap uses10,000 group resamples with frozen seed. Negative and zero-efficiency samples are retained. The cached D2 statistics preserve the exact serialization used by the original P6 training.

## Training artifacts

Each model/fold directory contains model.joblib, target-free scores, target-free selected IDs and fit audit. Root output includes plan, fit status, development freeze, OOF/historical predictions, scored metrics, negative groups and run receipt. Failed fits retain traceback files and nonzero completion status. Resource-limited fits are NOT_RUN, not replacement predictions. Saved P6 artifacts are read by src/predict.py; original extension runners retain their original parameter/output formats.

## Tests and verification

```bash
python -m pytest tests -q -p no:cacheprovider
python tools/preflight_real_data.py --prepared outputs/prepared --output outputs/input_preflight.json
```

Unit fixtures are small real-data excerpts with explicit test-only status. They are not valid replacement research cohorts; the training CLI rejects them. Mock supervised estimators test API/shape routing without changing official results. Full research training has not been rerun for this release.

All source algorithms remain in frozen_sources. P6's portable wrapper supplies complete raw/prepared inputs instead of reading other local research directories. Optional extension replay creates an isolated historical-layout workspace and adapts data loading only. P9 additionally relocates temporary Numba caches; function/class AST equality is checked.

Original results remain available for post-prediction comparison. Small score differences across BLAS/OS/compiler versions should be reported from evaluate.py, not hidden through parameter changes or sample edits.
