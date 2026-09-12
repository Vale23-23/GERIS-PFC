# GERIS-PFC — `testing-ml`

Machine-learning work for fire detection and future fire prediction over Uruguay using GOES ABI observations.

## Branch scope

This branch is dedicated to machine learning. It contains experimental model code, notebooks, and the first organization of the ML workflow.

The deterministic FDCA implementation belongs to another branch and is not part of this branch. The local `implementacion/` directory was copied into the workspace during a checkout and is intentionally ignored for this reorganization. It must remain untouched.

The ML-specific image-acquisition utilities are grouped under `machine_learning/image_acquisition/` so it is clear which files retrieve data for this branch and which files define or evaluate models.

## Repository layout

```text
.
├── README.md
├── machine_learning/
│   ├── README.md
│   ├── image_acquisition/
│   │   ├── goes_hf.py                # GOES/Hugging Face data access
│   │   ├── viirs_firms.py            # VIIRS/FIRMS access helpers
│   │   ├── download_viirs.py         # VIIRS download script
│   │   └── viirs_smoke_test.py       # Exploratory acquisition test
│   ├── models/
│   │   └── cnn_lstm.py               # ConvLSTM model and prototype utilities
│   ├── tracking.py                   # Optional Comet experiment-tracking adapter
│   ├── requirements-comet.txt       # Pinned optional tracking dependency
│   ├── notebooks/
│   │   ├── huggingface_conv_lstm.ipynb
│   │   └── goes2go_prediction_experiments.ipynb
│   └── experiments/
│       └── decision_tree_goes_demo.py
└── implementacion/                   # External FDCA checkout artifact; untouched
```

Raw data, credentials, virtual environments, caches, checkpoints, and generated outputs are local artifacts and must not be committed.

## Current ML status

The ML work is exploratory. The repository currently has:

- a `FireConvLSTM` prototype using B07 and B14 image channels;
- a Hugging Face notebook that extracts patches and temporal sequences;
- a GOES-19/`goes2go` notebook with alternative ConvLSTM and CNN experiments;
- a preliminary toy decision-tree demo;
- optional Comet tracking for the Hugging Face training notebook.

There is not yet a reproducible training command, a finalized temporal dataset, a documented forecast horizon, a trained checkpoint, or a production inference pipeline. The current notebooks must therefore be treated as experiments rather than completed model results.

## Where to start

Read [`machine_learning/README.md`](machine_learning/README.md) for the ML-specific data contract, notebook descriptions, dependencies, known reproducibility risks, and the recommended next steps.

The immediate priority is to define the temporal dataset and target correctly before increasing model complexity:

1. inventory common B07/B14/FDCF timestamps;
2. define the future horizon and label rule;
3. extract aligned `64 x 64` patches;
4. split by date or event;
5. establish a persistence/thermal baseline;
6. train and evaluate ConvLSTM with saved configuration and metrics.

## Dependencies

No formal package manifest exists yet. The exploratory workflow currently expects Python, NumPy, Pandas, Matplotlib, Jupyter, `huggingface_hub`, `python-dotenv`, PyTorch, scikit-learn, `goes2go`, xarray, pyproj, and a ConvLSTM implementation. Comet tracking is optional and can be installed from `machine_learning/requirements-comet.txt`.

The model currently imports `ConvLSTM` from an external `convlstm` module that is not included in this branch. Resolving that dependency is required before a clean training run can be reproduced.

## Data and security

The data loaders use paths such as `dataset/uruguay/<product>/<YYYYMMDD_HHMM>.npy`. These paths should later be replaced by a configurable data root and a validated timestamp inventory. FDCF codes `[13, 14, 15]` are used as provisional fire labels in the experiments; FDCF must be treated as an operational reference rather than unquestionable ground truth.

Credentials must be supplied through environment variables or a local `.env` file. Comet uses `COMET_API_KEY`, `COMET_PROJECT_NAME`, `COMET_WORKSPACE`, and `COMET_ENABLED`; never place these values in source files or notebooks.
