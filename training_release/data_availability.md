# Original public data and acquisition

D1 record: https://doi.org/10.5281/zenodo.7503391.
Author source repository: https://github.com/AI-InSu-Pero/ML-PerovskitePV-InSituLuminescene
Frozen author commit:b775fea4b043b039585cba9200ca6d7c3890c847.
Use the two files under extracted_features/:
- training_extracted_features_and_labels.h5 —10,138,216 bytes — SHA256998c3f7e680082881cfd73ea20c82c6045ae4ebae8c556e6296120cd64662032
- test_extracted_features_and_labels.h5 —5,119,912 bytes — SHA2568e55611f29ddadb3c87c536c8b11db4a9f1306175e8433903c482965156a0dd5

The author README documents these extracted PL/label tables. The29.6GB dataset.hdf5 image archive is unnecessary.

D2 record: https://doi.org/10.5281/zenodo.14609789.
Use data.h5 —354,689,616 bytes — SHA256050bf429ae6145f0201b940a25c6b89c5cb1fcdf961adc310fe3a0037f436f05; official MD5bbd23be2d7df03c071d0623ba2aeb255. Do not download forecasting_models.zip for this workflow.

Both archived Zenodo metadata records list CC BY4.0. Retain attribution to Laufer and coauthors and cite the original records. The package includes derived tables, exact D2 feature cache and small real-data unit-test excerpts; original H5 files are excluded. No original publication images are redistributed. See the manuscript and paper_analysis/data_availability.md for label/channel definitions.
