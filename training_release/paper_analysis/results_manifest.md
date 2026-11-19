# Results and input map

| Output | Input and interpretation |
| --- | --- |
| 40 macro comparisons | SOURCE_DATA/P6/*predictions.csv plus full-precision D1 targets; checked against ANALYSIS/recomputed_macro_metrics.csv |
| Figure1 | Original workflow_figure.py graphics; ANALYSIS/figure1_ACD_LP725.csv measured curves |
| Figure2 | recomputed_macro_metrics.csv; archived original group-bootstrap intervals |
| Figure3 | V2_position_summary.csv, paired_gain_comparisons.csv and D1/D2 position-paired group tables |
| Figure4 | real_LP725_examples.csv, real_LP725_example_metadata.csv and real_example_all_regions.csv |
| FiguresS1–S5 | Cohort/group tables; complete P6 macro/configuration and ablation tables; matched contrasts |
| FigureS6 | P6 group results; P7/P8 endpoint tables; P9 dataset metrics and resource summary |

Frozen focal development gains are D1 1.241585904708604 and D2 0.779041050990444 PCE percentage points. Main tables round means to three decimals; complete tables retain precision. Descriptive intervals and historical failures remain in the supplied tables. Configurations and source code have not been tuned in this release. SHA256_MANIFEST.json covers all distributed release inputs and source files; generated REPRODUCED files are excluded from that manifest.
