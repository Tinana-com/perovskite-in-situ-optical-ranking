# Reproduction

Run the two README commands from any working directory, using paths to the scripts. The code resolves inputs relative to the repository root. Python3.12 was used for the smoke test; exact analysis dependencies are pinned in requirements.txt. Arial is used if installed; Matplotlib falls back to an available sans-serif font otherwise. Numerical reproduction does not depend on the font.

## Metrics

For n eligible regions select k = ceil(n/4). Sort prediction scores descending, then stable sample identity ascending at ties. True top-k uses the same identity tie rule. Gain is selected mean target minus the full-group mean target, in PCE percentage points; overlap is intersection size divided by k. Spearman uses average ranks and excludes undefined constant-input cases from the mean.

D1's full-precision target projection replaces serialized target values before evaluation, preserving original ties. All regions in a substrate share one fold. D2 compares date-plus-experiment units; unit metrics average experiments equally within each date, then dates equally. Frozen assignments are retained; no folds are selected or regenerated. PCE=0 regions remain.

## Inputs and outputs

The metric entrypoint reads four frozen P6 prediction CSVs, both D1 full-precision label projections and ANALYSIS/recomputed_macro_metrics.csv. It writes independent unit, group and macro tables plus metric_checks.json to REPRODUCED. The plotting entrypoint reads only supplied CSVs and writes PNG/PDF/SVG figures plus input provenance. It fits no estimator and reads no original H5. All historical comparison rows and negative groups remain available in the source tables.

## Original feature/model definitions

SOURCE_DATA/P6/configs contains the frozen protocol, model grid and ablation matrix. SOURCE_DATA/P6/reference_source preserves the archived P6 feature extraction, pair construction and model definitions. These are supplied as source reference, with original project-specific paths; the documented archived-score workflow does not import or execute them. Original training environments additionally used scikit-learn and LightGBM4.6.0. Reconstructing or rerunning their training is not required to reproduce this release's metrics and figures.
