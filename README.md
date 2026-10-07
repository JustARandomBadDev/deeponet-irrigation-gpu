# deeponet-irrigation-gpu

DeepONet-based soil-moisture prediction on the Arnesano precision-irrigation
dataset, with a reproducible Python pipeline and later GPU inference
optimization using C++, CUDA, and TensorRT.

## Dataset setup

The data tooling is pinned to version 2 of *Soil Moisture, Irrigation Actuator
and Weather Dataset from a Multi-Sector Precision-Irrigation*:
[DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2).
It never requests the unversioned latest release.

```bash
uv sync
uv run python scripts/setup_data.py
uv run python scripts/inspect_dataset.py
```

`setup_data.py` downloads the official version-qualified Mendeley Data ZIP,
checks its server-published SHA-256, validates the documented directory layout,
and records provenance in `data/raw/arnesano-v2.metadata.json`. Re-running it
validates and reuses both the archive and extracted files.

`inspect_dataset.py` checks the supplied README and data dictionary, lists the
downloaded files, then streams every CSV in chunks to report shape, schema,
missing values, documented timestamp ranges, and documented sector identifiers.

Downloaded and generated artifacts under `data/`, `models/`, and `results/`
are intentionally excluded from Git. For future modeling work, the
author-provided tables under `02_processed_data/preprocessed/` should be the
starting point; rebuilding the multi-million-row raw actuator exports is out of
scope for this setup step.
