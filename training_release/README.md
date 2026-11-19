# In situ optical regional ranking: training and reproduction release

Code for Yutian Chen's manuscript, **Regional photovoltaic performance ranking from in situ optical monitoring**.

This repository provides raw-data preparation, the frozen P6 model and baseline training procedures, group-isolated out-of-fold (OOF) prediction, saved-model inference, scoring, and the manuscript analysis/figure workflow. D1 and D2 train separately. The original P7/P8/P9 supplementary training sources and explicit replay entrypoints are included.

## Install

Use Python 3.12 on Windows or Linux and a fresh environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

CPU only; frozen estimators use one thread. No GPU is needed.

## Obtain the original data

See [data_availability.md](data_availability.md). Obtain D1's two small author-provided extracted-feature H5 files and D2's 354,689,616-byte data.h5. The 29.6 GB D1 raw image file is unnecessary. Data are not downloaded automatically.

```bash
python src/prepare_data.py --d1-train /path/to/training_extracted_features_and_labels.h5 --d1-test /path/to/test_extracted_features_and_labels.h5 --d2-h5 /path/to/data.h5 --output outputs/prepared
```

Paths are arguments, not hardcoded Windows locations. Preparation verifies source SHA256, identity, channel roles, cohort sizes and frozen folds. It extracts D1 features from real LP725 arrays and audits the exact bundled D2 cache against every raw channel. Zero-efficiency records remain. To regenerate the D2 cache itself:

```bash
python src/rebuild_d2_cache.py --d2-h5 /path/to/data.h5 --output outputs/rebuilt_cache
```

## Train and generate OOF predictions

Inspect the full fixed training plan without fitting:

```bash
python src/train.py --prepared outputs/prepared --dataset D1 --output outputs/plan_D1 --dry-run
python src/train.py --prepared outputs/prepared --dataset D2 --output outputs/plan_D2 --dry-run
```

Train the focal model, using only development folds:

```bash
python src/train.py --prepared outputs/prepared --dataset D1 --models PAIR_ABS --output outputs/D1_focal
python src/train.py --prepared outputs/prepared --dataset D2 --models PAIR_ABS --output outputs/D2_focal
```

Train **all frozen P6 models, inherited controls and ablations**:

```bash
python src/train.py --prepared outputs/prepared --dataset D1 --models all --phase full --output outputs/D1_all
python src/train.py --prepared outputs/prepared --dataset D2 --models all --phase full --output outputs/D2_all
```

D1 has20 fitted configurations, five development folds and one all-development-to-history fit per configuration. D2 has14 fitted configurations, seven date folds and one all-development-to-history fit. Analytical random expectation is added during scoring, without inventing a constant-score model. The model counts include inherited position, intensity, author-kNN and absolute-Ridge controls. Parameters and the original P6 grid are frozen; inherited control definitions come from their P2/P4/P5 source protocols.

Default phase is development. Explicit `--phase full` replays the previously inspected historical cohorts after development outputs freeze. These historical results are not a new blind test. Every trained artifact and its target-free scores are retained; errors are recorded, not silently replaced. Use a new output directory for each attempt. Runtime limit defaults to7,200 seconds per dataset command; failures and resource-limited configurations remain recorded.

## Predict and evaluate

```bash
python src/predict.py --prepared outputs/prepared --model outputs/D1_all/models/history_fold-1/PAIR_ABS/model.joblib --role history --output outputs/D1_inference
python src/evaluate.py --prepared outputs/prepared --dataset D1 --predictions outputs/D1_all/predictions_BEFORE_label_merge.csv --compare-reference --output outputs/D1_evaluation
python src/evaluate.py --prepared outputs/prepared --dataset D2 --predictions outputs/D2_all/predictions_BEFORE_label_merge.csv --compare-reference --output outputs/D2_evaluation
```

The standalone P6 predictor does not open labels.csv. Scores and selected IDs are saved before scoring labels are merged. Evaluation reports score differences against the corresponding archived model, fold and sample identities; it does not use archived scores to fit the new models.

## Supplementary archived methods

The P7/P8/P9 sources are complete frozen training implementations. Their input/output adaptation creates an isolated workspace, supplies the exact prepared inputs and preserves the original nested/cross-fitting procedures, candidate registries, parameters and failure status handling:

```bash
python src/run_extension.py --stage P7 --prepared outputs/prepared --output outputs/P7_plan --dry-run
python src/run_extension.py --stage P8 --prepared outputs/prepared --output outputs/P8_plan --dry-run
python src/run_extension.py --stage P9 --prepared outputs/prepared --output outputs/P9_plan --dry-run
```

Remove `--dry-run` to replay that explicitly selected stage. No other stage starts automatically. P7/P8 reproduce their original bounded internal selection, not a new search design. P9 replays R1 development-only M0–M3; no R2 is implemented or started. Original D2 MiniRocket iteration-limit outcomes remain flagged. Full supplementary replays may require several hours; their original resource checks are retained. P9 uses the real raw D2 channel arrays in addition to prepared statistics. Its disposable Numba cache is relocated to avoid Windows path-length errors; model-function ASTs and frozen parameters are unchanged.

## Reproduce paper metrics and figures from archived outputs

```bash
python paper_analysis/src/evaluate_archived_scores.py
python paper_analysis/src/reproduce_figures.py
```

This separate fast path checks the existing40 model-scope comparisons and regenerates figures without training. It is distinct from the training commands above.

## Validation actually completed for this release

- Original H5 identities and SHA256 checked; no data downloaded.
- D2 cache regenerated byte-identically from raw data.
- Pair-ID hashes match the original P6 run in all12 development folds; training-only scaler parameters match archived values.
-57 interface/data-isolation tests passed. Supervised estimators are mocked in dispatch tests.
- P6 dry-run plans and P7/P8/P9 entrypoint import/IO checks executed.
- A full new supervised training replay was **not executed** during packaging. The package supplies that workflow; it does not claim an additional independent training replication.

See run_reproduce.md, results_manifest.md and VALIDATION_STATUS.json. Original training source bytes and provenance are retained under frozen_sources and SOURCE_PROVENANCE.json.

## License and citation

This complete training distribution uses **GPL-3.0**, with the original GPLv3 MiniRocket license retained. The previously released paper-analysis component retains its MIT notice. Dataset-derived tables/excerpts are attributed to their original CC BY4.0 data records. See THIRD_PARTY_NOTICES.md.

Chen, Y. (2026). In situ optical regional ranking: training and reproduction code, version1.0. No public repository URL or manuscript DOI has yet been assigned. Cite the D1/D2 records and the original MiniRocket paper when using that component.

Yutian Chen — The School of Information Science and Technology, Fudan University, Shanghai, China. Correspondence:25113090040@m.fudan.edu.cn.
