# Regional photovoltaic performance ranking from in situ optical monitoring

**Reproducible analysis and model training for in situ optical screening of blade-coated perovskite photovoltaics.**

This repository accompanies the manuscript *Regional photovoltaic performance ranking from in situ optical monitoring*. It provides browsable Python source code, fixed experimental configurations, grouped-validation routines, archived predictions and scoring workflows. D1 and D2 are modeled **separately**.

## Research overview

- **Task:** rank candidate device regions within a substrate (D1) or experiment (D2), and quantify enrichment from selecting the highest-ranked fraction.
- **Data:** publicly released KIT blade-coating optical datasets (D1: LP725 photoluminescence; D2: three PL channels plus diffuse reflection).
- **Method:** compact optical descriptors, linear pairwise ranking, and matched regression, intensity and positional controls.
- **Evaluation:** substrate-isolated D1 and date-isolated D2 development folds; within-group ranking, Top-25% PCE enrichment, overlap and rank correlation.
- **Reproducibility:** training code, inference/scoring entry points, archived result checks, deterministic configurations and tests.

## Repository contents

| Path | Contents |
| --- | --- |
| [`training_release/src/`](training_release/src/) | Data preparation, feature extraction, model training, prediction and evaluation |
| [`training_release/configs/`](training_release/configs/) | Fixed model and evaluation configurations |
| [`training_release/tests/`](training_release/tests/) | Interface, cohort-integrity and leakage-control tests |
| [`training_release/reference_results/`](training_release/reference_results/) | Archived predictions and benchmark outputs |
| [`training_release/paper_analysis/`](training_release/paper_analysis/) | Archived-score evaluation and associated figure workflow |
| [`training_release/run_reproduce.md`](training_release/run_reproduce.md) | Detailed commands and output descriptions |
| [`training_release/data_availability.md`](training_release/data_availability.md) | Original public data sources and expected input files |
| [`training_release/results_manifest.md`](training_release/results_manifest.md) | Result files and verification map |
| [`training_release/THIRD_PARTY_NOTICES.md`](training_release/THIRD_PARTY_NOTICES.md) | License and third-party attributions |

The original frozen distribution is also preserved as [`PL_SPATIAL_RANKING_TRAINING_REPRODUCTION_V1.zip`](PL_SPATIAL_RANKING_TRAINING_REPRODUCTION_V1.zip) for checksums and archival reference.

## Quick start

**Python 3.12**, CPU; no GPU required.

```bash
cd training_release
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Download the original datasets as instructed in [data availability](training_release/data_availability.md), then prepare the analysis tables:

```bash
python src/prepare_data.py \
  --d1-train /path/to/training_extracted_features_and_labels.h5 \
  --d1-test /path/to/test_extracted_features_and_labels.h5 \
  --d2-h5 /path/to/data.h5 \
  --output outputs/prepared
```

Preview the fixed training plan:

```bash
python src/train.py --prepared outputs/prepared --dataset D1 --output outputs/plan_D1 --dry-run
python src/train.py --prepared outputs/prepared --dataset D2 --output outputs/plan_D2 --dry-run
```

Train the 12-descriptor pairwise-ranking baseline:

```bash
python src/train.py --prepared outputs/prepared --dataset D1 --models PAIR_ABS --output outputs/D1_pairwise
python src/train.py --prepared outputs/prepared --dataset D2 --models PAIR_ABS --output outputs/D2_pairwise
```

Re-evaluate the archived prediction outputs:

```bash
python paper_analysis/src/evaluate_archived_scores.py
python paper_analysis/src/reproduce_figures.py
```

See [full reproduction instructions](training_release/run_reproduce.md) for complete baseline training, prediction, evaluation, file formats and verification.

## Data and evaluation

- **D1:** 780 development regions from 26 substrates; 349 historical regions from 12 substrates.
- **D2:** 4,707 development regions in 158 experiments across 14 dates; 1,150 historical regions in 36 experiments across three dates.
- Selection uses the top `ceil(q × n)` predicted regions per comparison group; gain is selected mean PCE minus the group mean in percentage points.
- Original experimental data remain available from their public authors' archives rather than being duplicated here.

**Dataset records:** [D1](https://doi.org/10.5281/zenodo.7503391) · [D2](https://doi.org/10.5281/zenodo.14609789)

## Citation and license

If using this code or its outputs, please cite the accompanying manuscript and the original D1/D2 dataset publications. Citation metadata and upstream acknowledgements are provided in [`training_release/CITATION.cff`](training_release/CITATION.cff) and [third-party notices](training_release/THIRD_PARTY_NOTICES.md).

The complete training distribution is provided under GPL-3.0; component-specific upstream notices are retained. The original research data are credited to their respective authors.

---

For full experimental settings, archived validation records, and reproducibility checks, see the [reproduction documentation](training_release/run_reproduce.md) and [validation manifest](training_release/VALIDATION_STATUS.json).
