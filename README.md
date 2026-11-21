# BAR-Rank: regional photovoltaic performance ranking

Training, evaluation and figure-generation code accompanying **BAR-Rank: Batch-Aligned Residual and Pairwise Score Fusion for Regional Photovoltaic Performance Ranking**, by Yutian Chen.

The workflow ranks device regions using in situ optical measurements. It combines a linear pairwise ranker with a training-target-centered TabPFN predictor through fixed equal-weight score fusion. D1 and D2 are trained separately, with substrate-isolated and date-isolated validation, respectively.

## Start here

| Goal | Entry point | Requires original H5? | Fits models? |
|---|---|---:|---:|
| Verify published-result tables from archived predictions | `python src/reproduce_results.py` | No | No |
| Run interface, metric and leakage-isolation tests | `python -m pytest tests -q` | No | No supervised fits |
| Prepare optical inputs and frozen folds | `src/prepare_data.py` | Yes | No |
| Train pairwise ranking and classical baselines | `src/train.py` | Prepared inputs | Yes |
| Train raw/centered/position-augmented TabPFN | `src/train_tabpfn.py` | Prepared inputs + licensed checkpoint | Yes |
| Fuse independently generated scores | `src/fuse_scores.py` | No | No |
| Evaluate target-free predictions | `src/evaluate.py` | Prepared labels | No |
| Recreate manuscript5 plots | `src/reproduce_figures.py` | No | No |

## Repository layout

```text
src/                    Public preparation, training, inference and evaluation commands
  bar_rank/             Target centering, optical representations and fixed score fusion
configs/                Frozen model and data protocols
analysis/manuscript5/   Latest figure code and the exact archived plot inputs
reference_results/      Sample-level predictions, targets and reference metric tables
tests/                  Real-data excerpts, identity and leakage-isolation tests
data_cache/             D2 author-data-derived optical statistics
archived_methods/       Original implementations of later controlled comparisons
frozen_sources/         Original baseline and supplementary implementations
paper_analysis/         Earlier baseline analysis workflow and its MIT notice
environments/           Separate TabPFN dependencies and recorded GPU environment
docs/                   Reproduction, module map and GitHub upload guide
```

### Installation and fast result verification

Python 3.12 is the baseline training/reproduction environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv with your operating system's standard command.
python -m pip install -r requirements.txt
python src/reproduce_results.py --output outputs/archived_check
python -m pytest tests -q
```

The fast path recomputes sample-level ranking, fixed fusion, substrate/date means and paired bootstrap increments. It uses archived predictions, so it needs neither a GPU nor TabPFN model weights.

### Train from original public data

Obtain the two small D1 extracted-feature H5 files and D2 `data.h5` from the [D1 Zenodo record](https://zenodo.org/records/7503391) and [D2 Zenodo record](https://zenodo.org/records/14609789). The 29.6 GB D1 image archive is unnecessary. Exact input hashes are in `configs/frozen_p6_protocol.json`.

```bash
python src/prepare_data.py --d1-train DATA/training_extracted_features_and_labels.h5 --d1-test DATA/test_extracted_features_and_labels.h5 --d2-h5 DATA/data.h5 --output outputs/prepared
python src/train.py --prepared outputs/prepared --dataset D2 --models PAIR_ABS --output outputs/D2_pairwise
```

For TabPFN, create the separate environment described in [docs/REPRODUCING.md](docs/REPRODUCING.md), obtain the specified checkpoint under its original terms, and run:

```bash
python src/train_tabpfn.py --prepared outputs/prepared --dataset D2 --variant centered --device cuda --checkpoint MODELS/tabpfn-v3.5-20260909.safetensors --output outputs/D2_centered
python src/fuse_scores.py --pair-scores outputs/D2_pairwise/predictions_BEFORE_label_merge.csv --tabpfn-scores outputs/D2_centered/predictions_BEFORE_label_merge.csv --tabpfn-model D2_TABPFN_CTR_ABS12 --output outputs/D2_fusion
python src/evaluate.py --prepared outputs/prepared --dataset D2 --predictions outputs/D2_fusion/predictions_BEFORE_label_merge.csv --output outputs/D2_fusion_evaluation
```

Use `--dry-run` with either training command to validate the full fold/input plan without fitting. Development folds are the default. `--phase full` explicitly includes previously inspected historical groups; use the same phase for both fusion components. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for D1, raw-target controls, the joint branch and controlled ablations.

## Reference results

Top25% gain is the selected regions' mean measured PCE minus the full comparison-unit mean, in PCE percentage points. Selection uses `ceil(0.25*n)` regions. D1 averages substrates equally; D2 first averages experiments within dates and then dates equally.

| Development cohort | Absolute pairwise | Optical fusion | Paired fusion increment |
|---|---:|---:|---:|
| D1: 780 regions, 26 substrates | 1.241586 | 1.352640 | +0.111054 |
| D2: 4,707 regions, 158 experiments, 14 dates | 0.779041 | 0.889969 | +0.110928 |

D2 raw-target and centered-target TabPFN gains are 0.545121 and 0.813798 pp. The D2 position-plus-optical fusion gain is 0.951203 pp. Complete matched comparisons, uncertainty intervals and historical results are regenerated by `src/reproduce_results.py`. D1's optical-fusion increment interval includes zero. Development cohorts were examined iteratively; historical cohorts had previously been viewed.

## Reproducibility and scope

The release includes training implementations, frozen configurations, sample-level scores, negative results and figure inputs. Original sources are traced in `SOURCE_PROVENANCE.json`; file integrity is recorded in `SHA256SUMS.txt`. [VALIDATION_STATUS.json](VALIDATION_STATUS.json) records checks actually executed while assembling this release. Packaging validation does not claim a new full supervised training replay. GPU/CPU backend and dependency differences may change floating-point predictions.

## License, attribution and citation

The complete code distribution retains **GPL-3.0**, including the original MiniRocket license. The earlier `paper_analysis` component retains its MIT notice. Dataset-derived tables and real-data excerpts are attributed to the original KIT records under their data licenses. TabPFN and its model weights have their own terms; weights are obtained separately. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Software citation is supplied in [CITATION.cff](CITATION.cff). Cite the original datasets and methods used in your work. The manuscript DOI will be added when assigned.

Yutian Chen · School of Information Science and Technology, Fudan University · 25113090040@m.fudan.edu.cn
