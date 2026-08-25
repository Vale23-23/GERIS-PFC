# GERIS-PFC

Python implementation of NOAA's **Fire Detection and Characterization Algorithm (FDCA)** for GOES-19 ABI imagery over Uruguay. The algorithm is organized into:

- **Part I:** pixel-level filters and fire-candidate generation.
- **Part II:** confirmation, classification, and temporal filtering.
- **Full:** Part I followed by Part II for one scene.

This README documents the reproducible workflow for running **Part I from a terminal**, including the case where the dataset is not present after cloning the repository.

## Current status

Part I can be run with `python -m fdca.run_part1`. The runner validates the required input files and, when `--download` is supplied, downloads only the requested scene from the public Hugging Face dataset:

<https://huggingface.co/datasets/valentina2323/GERIS-Goes19-uruguay-fires>

The dataset is not versioned in Git because it contains large `.npy` and `.nc` files. Therefore, a clean clone does not contain the ABI scenes; the data must be downloaded during setup or on the first run.

## Requirements and installation

Python 3.10 or newer is recommended. From the repository root:

```bash
git clone <REPOSITORY_URL>
cd GERIS-PFC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

`HF_TOKEN` is optional while the dataset is public. If Hugging Face requests authentication, add a read-only token to `.env`; never commit it to Git. The `.env` file is already ignored.

## Available timestamps

The available evaluation scenes are within **2025-11-15 00:00 UTC — 2025-12-15 23:59 UTC**. 

Timestamps use the `YYYYMMDD_HHMM` format and are interpreted as UTC. The local development data also contains the test scene `20251117_1820` when that scene has already been downloaded.

## Running Part I from the terminal

Change into the implementation directory so that the `fdca` package and its default paths resolve correctly:

```bash
cd implementacion
```

Run a scene and download its data automatically if it is not already present:

```bash
python -m fdca.run_part1 \
  --timestamp 20251120_1800 \
  --region uruguay \
  --dataset-root data \
  --config fdca/config.yaml \
  --download
```

The `--download` flag is safe to reuse: the downloader requests only the selected timestamp and Hugging Face reuses files already available locally. For a scene that has already been downloaded, the flag can be omitted:

```bash
python -m fdca.run_part1 \
  --timestamp 20251120_1800 \
  --region uruguay \
  --dataset-root data \
  --config fdca/config.yaml
```

If the required files are missing and `--download` is omitted, the runner stops with a list of missing files and prints the command needed to download them. This avoids silently starting an incomplete or non-reproducible run.

### What is downloaded

For the requested timestamp, the runner obtains the scene-specific ABI files from Hugging Face, including the available B02, B07, B13, B14, and B15 radiance arrays, Planck calibration JSON files, data-quality flags, and the shared `geometry.json`. B07 and B14 radiances, their Planck coefficients, and the geometry file are mandatory. B02, B13, B15, CAMEL emissivity, and data-quality flags are optional inputs handled by the adapter when unavailable.

### Part I outputs

When the command is run from `implementacion/`, results are saved under the repository at `implementacion/results/part1/<timestamp>/`. The runner prints the absolute path at the end. You can override it with `--output-dir`, for example `--output-dir /path/to/my/results`.

The command saves these files:

```text
fire_mask_part1.npy    Part I mask for every pixel
fail_char_part1.npy    Part I failure-characterization codes
candidates_part1.npy   row/column coordinates of fire candidates
summary.json           timestamp, shape, and candidate counts
```

The number of candidates is also printed in the terminal. The Part I walkthrough notebook, `implementacion/FDCA_Part1_walkthrough.ipynb`, remains available for a step-by-step analysis and visual inspection of intermediate stages.

## Running the Full pipeline

The existing Full runner executes Part I and Part II and generates figures. From `implementacion/`, run:

```bash
python -m fdca.run_fdca \
  --timestamp 20251120_1800 \
  --region uruguay \
  --dataset-root data \
  --config fdca/config.yaml \
  --download \
  --save-outputs \
  --output-dir figures
```

Figures are written to `figures/<timestamp>/`. With `--save-outputs`, the final Part II arrays and summary are written to `data/<timestamp>/`. The Full runner now uses the same dataset preflight and download behavior as Part I.

Full temporal processing is still a work in progress: the `PreviousFireMaskStore` implementation exists, but it is not yet connected to the command-line adapter. Consequently, a single-scene Full run currently uses no previous-fire mask.

## Repository structure

```text
GERIS-PFC/
├── implementacion/
│   ├── fdca/
│   │   ├── part1.py          Part I algorithm
│   │   ├── part2.py          Part II algorithm
│   │   ├── dataset.py        Dataset validation and Hugging Face download
│   │   ├── run_part1.py      Part I command-line runner
│   │   ├── run_fdca.py       Full command-line runner
│   │   └── fdca_adapter.py   Scene files → FDCAInput
│   ├── FDCA_Part1_walkthrough.ipynb
│   └── tests/
├── requirements.txt
└── README.md
```

## Tests

From the repository root, with the virtual environment activated:

```bash
python -m pytest implementacion/tests/ -v
```

The tests use synthetic inputs and do not require downloading the satellite dataset.
