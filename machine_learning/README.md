# Machine Learning

This directory contains the machine-learning work for the `testing-ml` branch. It is intentionally separate from deterministic FDCA code and includes the data-acquisition utilities required by the ML experiments.

## Scope and current status

The current implementation is exploratory. It contains data-preparation notebooks and ConvLSTM prototypes, but it does not yet contain a reproducible training pipeline, a trained checkpoint, or a production prediction command.

The intended first task is temporal fire prediction from GOES ABI observations. The proposed input uses B07 and B14 image patches, while `ABI-L2-FDCF-Mask` is currently used as an operational reference label. The exact forecast horizon and label definition are still open decisions.

## Layout

```text
machine_learning/
├── README.md
├── image_acquisition/
│   ├── goes_hf.py                    # GOES/Hugging Face data access
│   ├── viirs_firms.py                # VIIRS/FIRMS access helpers
│   ├── download_viirs.py             # VIIRS download script
│   └── viirs_smoke_test.py           # Exploratory acquisition test
├── models/
│   └── cnn_lstm.py                    # ConvLSTM model, training loop, metrics
├── tracking.py                         # Optional Comet experiment-tracking adapter
├── requirements-comet.txt             # Pinned optional tracking dependency
├── notebooks/
│   ├── huggingface_conv_lstm.ipynb    # Hugging Face data and ConvLSTM prototype
│   └── goes2go_prediction_experiments.ipynb
│                                      # GOES-19 download and model experiments
└── experiments/
    └── decision_tree_goes_demo.py     # Small legacy/demo experiment
```

The image-acquisition helpers are part of this ML branch because they retrieve the data used by these experiments:

```text
machine_learning/image_acquisition/
├── goes_hf.py                         # GOES data retrieval from Hugging Face
├── viirs_firms.py                     # VIIRS/FIRMS retrieval helpers
├── download_viirs.py                  # VIIRS download script
└── viirs_smoke_test.py                # Exploratory VIIRS smoke test
```

The local `implementacion/` directory is not part of this branch's ML organization. It was copied into the workspace from the FDCA checkout and must remain untouched.

## Main components

### `models/cnn_lstm.py`

Defines `FireConvLSTM`, which expects tensors with shape:

```text
(batch, time, channels, height, width)
(batch, time, 2, 64, 64)
```

The prototype uses one ConvLSTM layer with 32 hidden channels, global average pooling, and a binary output. It also contains a basic training loop and confusion-matrix, precision, recall, and F1 calculations.

The file imports `ConvLSTM` from an external `convlstm` module that is not currently included in this branch. A clean environment therefore cannot train the model until that dependency is restored or replaced.

### Notebooks

- **`huggingface_conv_lstm.ipynb`** prepares B07/B14/FDCF data downloaded from Hugging Face, extracts patches, creates temporal sequences, and attempts to train `FireConvLSTM`.
- **`goes2go_prediction_experiments.ipynb`** contains direct GOES-19/NOAA retrieval experiments using `goes2go`, radiance-to-brightness-temperature conversion, patch extraction, FDCF labels, feature analysis, and alternative ConvLSTM/CNN prototypes.
- **`decision_tree_goes_demo.py`** is an isolated demo mixing a toy decision tree with GOES visualization. It is not a validated ML baseline.

## Data contract currently assumed

- Inputs: `ABI-L1b-Rad-B07` and `ABI-L1b-Rad-B14`.
- Reference target: `ABI-L2-FDCF-Mask`.
- Fire codes currently used in experiments: `[13, 14, 15]`.
- Patch size in the prototypes: `64 x 64`.
- Proposed sequence length in the Hugging Face notebook: 5 frames.
- Files are named using `YYYYMMDD_HHMM` timestamps.

These assumptions are not yet a finalized dataset specification. Before training, timestamps must be inventoried and aligned across products, spatial grids must be verified, inputs must be normalized using training data only, and the future target must be separated from the input sequence.

## Running the current prototypes

Run notebooks from the repository root so package imports resolve consistently:

```bash
jupyter notebook machine_learning/notebooks/huggingface_conv_lstm.ipynb
jupyter notebook machine_learning/notebooks/goes2go_prediction_experiments.ipynb
```

The current scripts still assume a local `dataset/` directory and require external credentials/dependencies. Do not commit raw datasets, `.env` files, FIRMS keys, virtual environments, caches, checkpoints, or generated outputs.

Expected dependencies include NumPy, Pandas, Matplotlib, Jupyter, `huggingface_hub`, `python-dotenv`, PyTorch, scikit-learn, `goes2go`, xarray, pyproj, and a working ConvLSTM implementation. Optional Comet tracking is pinned in `requirements-comet.txt`.

## Comet experiment tracking

Comet is integrated as an optional tracker in `tracking.py`. It is disabled by default, so the notebook and training loop work without the Comet package or credentials.

Install the pinned optional dependency:

```bash
python -m pip install -r machine_learning/requirements-comet.txt
```

Set the credentials and enable tracking in the shell or a local `.env` file:

```bash
export COMET_ENABLED=true
export COMET_API_KEY="your-comet-api-key"
export COMET_PROJECT_NAME="geris-pfc-ml"
export COMET_WORKSPACE="your-workspace"
```

The Hugging Face notebook records model/training parameters, training loss, validation loss, precision, recall, and F1. It does not upload raw datasets or credentials. If `COMET_ENABLED` is false or unset, a no-op tracker is used. The API key is read by the Comet SDK from the environment and is never logged as a parameter.

The current integration follows Comet's Python SDK `comet_ml.start()` flow and standard parameter/metric logging. See the [Comet Python SDK documentation](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/start-experiment/) and [metrics and parameters guide](https://www.comet.com/docs/v2/guides/experiment-management/log-data/metrics-and-parameters/).

## Quickstart: tiny decision-tree experiment with Comet

A minimal toy example is available at `machine_learning/experiments/decision_tree_simple_comet.py`. It trains a small scikit-learn decision tree on a synthetic binary dataset and logs parameters/metrics to Comet when enabled.

Install the project requirements into your active environment and copy the environment template:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the static defaults once in `.env`, then override them per run when you want a different experiment name or tags without editing the file again:

```bash
export COMET_ENABLED=true
export COMET_API_KEY="your-comet-api-key"
export COMET_PROJECT_NAME="geris-pfc-ml"
export COMET_WORKSPACE="your-workspace"
python machine_learning/experiments/decision_tree_simple_comet.py --experiment-name "toy-decision-tree-v2" --tags "toy,baseline,debug"
```

You can also pass the values directly inline without editing `.env`:

```bash
COMET_ENABLED=true COMET_EXPERIMENT_NAME="toy-decision-tree-v3" COMET_TAGS="toy,baseline" \
python machine_learning/experiments/decision_tree_simple_comet.py
```

If `COMET_ENABLED` is false or unset, the script still works in local mode using the no-op tracker.

## Reproducibility risks

- The local dataset is not versioned and timestamps are not guaranteed to be regular.
- Several notebooks contain exploratory cells with hard-coded paths or environment-specific behavior.
- Some cells force CUDA even when the installed PyTorch build only supports CPU.
- Labels and temporal splits need review to avoid leakage and to distinguish current-time classification from future prediction.
- FDCF is an operational reference product, not an independent ground-truth dataset.
- Class imbalance requires precision, recall, F1, PR-AUC, and per-scene metrics rather than accuracy alone.

## Next implementation steps

1. Add a shared data-root and timestamp-inventory utility.
2. Define the forecast horizon and target-generation rule.
3. Build a reproducible sequence dataset with date/event-based splits.
4. Implement a persistence and thermal-threshold baseline.
5. Restore or replace the ConvLSTM dependency.
6. Add a configurable CPU/GPU training script and save checkpoints plus metadata.
7. Add richer Comet assets once the dataset and evaluation protocol are stable.
8. Evaluate ConvLSTM against the baseline and the FDCF reference.
