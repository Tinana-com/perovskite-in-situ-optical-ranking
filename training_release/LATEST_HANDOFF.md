# Training release handoff

The package adds data preparation, native P6 and inherited-control training, OOF scoring, saved-model inference and evaluation to the existing paper-analysis release. Original P7/P8/P9 training sources and explicit replay adapters are included. Only this new directory was changed; historical research and manuscript files were read-only.

Full real H5 data and exact optical cache identities were checked. The D2 cache was rebuilt byte-identically. Pair IDs and preprocessing match all12 original development folds. Interface tests use mocked supervised estimators; full new supervised training replay was not executed. Original negative, historical and unconverged results remain reference evidence.

P9 required relocation of disposable Numba caches for Windows path lengths. The original sources remain unchanged in frozen_sources; runtime copying adjusts only cache paths, checking all function/class ASTs. No model family, parameter, data cohort or metric changed.

This release is GPL-3.0 with retained third-party/component notices. It is prepared for the author's GitHub upload; no repository URL is invented. New ZIP and cloud receipt hashes are external to the signed ZIP. Stop after delivery.
