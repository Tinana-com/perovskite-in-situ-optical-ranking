# Source and result map

| Component | Location |
| --- | --- |
| Raw-data verification and D1 extraction | src/prepare_data.py |
| Original D2 cache reconstruction | src/rebuild_d2_cache.py; frozen_sources/P2/src/run_p2.py |
| P6 training, feature normalization and pair construction | src/train.py; src/portable_models.py; frozen_sources/P6/src/ |
| Inherited author kNN, intensity and position controls | frozen_sources/P2,P4,POSITION,P5; portable_models.py adapters |
| Target-free inference | src/predict.py |
| Scoring and archived-score comparison | src/evaluate.py |
| P7 dual-view training and bounded nested selection | frozen_sources/P7; src/run_extension.py --stage P7 |
| P8 reliability gate and double-OOF training | frozen_sources/P8; src/run_extension.py --stage P8 |
| P9 R1 trajectory/MiniRocket training | frozen_sources/P9; src/run_extension.py --stage P9 |
| Original model, ablation and grouping settings | configs/ and frozen_sources/*/configs/ |
| Archived golden predictions and failures | reference_results/ |
| Main/SI metric and plotting reproduction | paper_analysis/ |
| Small real-data test excerpts | tests/fixtures/; test-only, no replacement cohort |
| Provenance, release checks and hashes | SOURCE_PROVENANCE.json; VALIDATION_STATUS.json; SHA256_MANIFEST.json |

The main P6 route retrains inherited controls from their original implementations; it does not copy their archived scores into new training predictions. Archived scores are read only by the explicit comparison command. Supplementary extension runners retain their original frozen comparator behavior, recorded in original code. An analytical random expectation has metrics but no fitted model or fabricated patch-score vector.

The focal archived development gains are D1+1.241585904708604 pp and D2+0.779041050990444 pp. These remain reference values, not new fitted results from package construction. P9's unconverged D2 endpoints and all historical negative outcomes remain unchanged.
