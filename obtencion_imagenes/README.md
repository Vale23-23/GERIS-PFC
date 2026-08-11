# GOES-19 Image Acquisition

This module downloads satellite images from the GOES-19 satellite and saves them to your computer for use in training detection models.

---

## What does this do?

The GOES-19 satellite takes photos of South America every ten minutes. This code downloads those images, crops them to the region of interest (for example, Uruguay), and saves them organized into folders.

Each image is saved as an `.npy` file (Python's numeric format). There is one file per ten minutes, per product.


The DQF (Data Quality Flag) field of the FDCF product indicates:
            DQF = 0: fire detected with good quality → marked as 1
            DQF = 1: land pixel with no fire (good quality)
            DQF = 2: invalid due to opaque cloud
            DQF = 3: invalid due to surface type or sun glint
            DQF = 4: invalid due to bad input data
            DQF = 5: invalid due to algorithm failure
---

## Project files

```
obtencion_imagenes/
├── config.yaml      ← Configuration: what to download and from which region
├── pipeline.py      ← The main script you'll use
├── downloader.py    ← Internal download logic (do not touch)
├── manifest.py      ← Log of what was downloaded (do not touch)
```

---

## Before you start

Make sure you have the dependencies installed. Use an environment. From the project's root folder:

```bash
pip install goes2go pyproj pyyaml numpy
```
or

```bash
pip install -r requirements.txt
```

All commands are run from inside the `obtencion_imagenes/` folder:

```bash
cd obtencion_imagenes
```

---

## Basic usage

### 1. See which products are available

```bash
python pipeline.py list-products
```

This shows the products configured, for example:

```
📡 Available products in config.yaml:

  ABI-L1b-Rad-B07     B07  Shortwave IR 3.9µm - fire thermal signature
  ABI-L1b-Rad-B14     B14  Longwave IR 11.2µm - thermal context
  ABI-L2-FDCF           -  Fire detection mask - ground truth label
```
### 2. See which regions are available

```bash
python pipeline.py list-regions
```

### 3. Download images

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-02 23:00" \
  --products ABI-L1b-Rad-B07 ABI-L2-FDCF
  --interval 10
```

This downloads Band 7 and the fire mask for Uruguay every ten minutes, between September 1 and 2, 2025.

Start of operational data: April 7, 2025

While downloading, you'll see something like this:

```
🚀 Downloading 48 files with 4 workers...

  💾 20250901_0000  ABI-L1b-Rad-B07    downloaded
  💾 20250901_0000  ABI-L2-FDCF        downloaded
  ✅ 20250901_0100  ABI-L1b-Rad-B07    exists
  ...

✔ Downloaded: 40  |  Already existed: 8  |  Errors: 0
```

- 💾 = downloaded now
- ✅ = already existed, not downloaded again
- ⚠️ = no data available for that hour
- ❌ = connection error or other problem


### 4. Add a new band without re-downloading everything

If you already have Band 7 downloaded and want to add Band 14, simply run the same command with the new product. The script automatically detects what already exists and only downloads what's missing:

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-02 23:00" \
  --products ABI-L1b-Rad-B14
  --interval 10
```

### 5. Check the dataset status

```bash
python pipeline.py status \
  --region uruguay \
  --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 ABI-L2-FDCF
```

This shows how many files were downloaded per product and whether there are incomplete timestamps (hours where some product is missing):

```
📦 Status by product:
  ABI-L1b-Rad-B07     ✅ 48  ❌ 0
  ABI-L1b-Rad-B14     ✅ 48  ❌ 0
  ABI-L2-FDCF         ✅ 48  ❌ 0

✅ Complete timestamps (all products): 48
```

### 6. Retry downloading images that returned errors

With this you can run:
```bash
python pipeline.py retry --region uruguay --products ABI-L1b-Rad-B07 ABI-L2-FDCF
```
```bash
or simply:
python pipeline.py retry --region uruguay
```

If the terminal returns something like:
❌ 20250922_1600  ABI-L1b-Rad-B07                 error
❌ 20250922_1600 -> error_aws_gap

it means the error is that the file cannot be found on AWS.

### 7. View fire statistics

```bash
python pipeline.py fire-stats --region uruguay
```

Shows how many timestamps have detected fire, the percentage, and a ranking of the top 10 with the most fire pixels:

```
🔥 FIRE STATISTICS — uruguay
=============================================
  Total timestamps analyzed   : 48
  With fire detected          : 12  (25.0%)
  Without fire                : 36  (75.0%)

🔝 Top 10 timestamps with most fire pixels:
  Timestamp              Fire pixels
  -----------------------------------
  20250915_1400                    143
  20250912_1600                     87
  ...
```


python pipeline.py spatial-report --region uruguay --timestamp 20250926_1900

### 8. Visualize an image and its mask

```bash
python pipeline.py visualize --region uruguay --timestamp 20250901_1200
```

Opens a window with two panels: the Band 7 image (thermal infrared) on the left and the fire mask on the right. Useful for visually inspecting the dataset before training.

If you don't know which timestamps have fire, first run `fire-stats` to see the ranking.

---

## Where are the files saved?

Files are saved in a `dataset/` folder inside `obtencion_imagenes/`, organized as follows:

```
dataset/
└── uruguay/
    ├── ABI-L1b-Rad-B07/
    │   ├── 20250901_0000.npy
    │   ├── 20250901_0100.npy
    │   └── ...
    ├── ABI-L1b-Rad-B14/
    │   └── ...
    ├── ABI-L2-FDCF/
    │   └── ...
    └── manifest.json   ← log of everything downloaded
```

Each `.npy` file is an image cropped to the chosen region, saved as a numeric matrix.

The `manifest.json` file is an automatic log of everything that was downloaded, with status and dimensions. There's no need to open it manually.

Example structure:

{
  "20250901_1100": {
    "status": "complete",
    "fire": {
      "fire_pixels": 42,
      "has_fire": true,
      "class_label": "fire"
    },
    "bands": {
      "ABI-L1b-Rad-B07": {
        "status": "exists",
        "path": "dataset/uruguay/ABI-L1b-Rad-B07/20250901_1100.npy",
        "shape": [500, 700]
      },
      "ABI-L2-FDCF": {
        "status": "downloaded",
        "path": "dataset/uruguay/ABI-L2-FDCF/20250901_1100.npy",
        "fire_pixels": 42
      }
    }
  }
}

---

## Adding a new product or region

### New product (band)

Open `config.yaml` and add a new block under `products`:

```yaml
  - id: ABI-L1b-Rad-B02
    product: ABI-L1b-Rad
    band: 2
    variable: Rad
    dtype: float32
    description: "Visible 0.64µm - visible light"
```

You can then download it with `--products ABI-L1b-Rad-B02` without touching any other file.

### New region

Add an entry under `regions` in `config.yaml`:

```yaml
  patagonia:
    lat_min: -55.0
    lat_max: -40.0
    lon_min: -75.0
    lon_max: -60.0
```

And use it with `--region patagonia`.

---

## Advanced options

| Option | Description | Default |
|---|---|---|
| `--interval N` | Download every N hours instead of every 1 | 1 |
| `--workers N` | How many parallel downloads | 4 (config.yaml) |

Example: download every 3 hours with 6 workers:

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-30 23:00" \
  --products ABI-L1b-Rad-B07 ABI-L2-FDCF \
  --interval 3 \
  --workers 6
```

> ⚠️ Don't use more than 8 workers. NOAA's servers may block connections if too many simultaneous downloads are made.

---

## Syncing the dataset with Hugging Face

The dataset is stored in a shared way on Hugging Face so the whole team can access it without needing to download everything from scratch.

### When should you use this script?

Use `sync_hf.py` every time you download new data with `pipeline.py` and want the rest of the team to have it available. The typical flow is:

1. Download new data with `pipeline.py`
2. Verify everything is fine with `pipeline.py status`
3. Upload the changes to Hugging Face with `sync_hf.py`

Only the person who downloaded the data needs to run this script. Everyone else just downloads from HF.

### Upload data to Hugging Face

From the project root:

```bash
.venv/bin/python obtencion_imagenes/sync_hf.py
```

You'll see something like this:

```
📦 Repository ready: https://huggingface.co/datasets/tu-usuario/goes19-uruguay-fires
⬆️  Uploading dataset/ → tu-usuario/GERIS-Goes19-uruguay-fires ...

✅ Dataset synced successfully.
   View at: https://huggingface.co/datasets/tu-usuario/GERIS-Goes19-uruguay-fires
```

The script only uploads new or modified files; it doesn't re-upload what was already there.

### Download the dataset (for the rest of the team)

If you're a teammate who wants to have the dataset locally, from the project root:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='valentina2323/GERIS-Goes19-uruguay-fires',
    repo_type='dataset',
    local_dir='obtencion_imagenes/dataset',
    token='hf_tu_token'
)
"
```

> ⚠️ The Hugging Face token is personal and private. Don't share it or upload it to GitHub. It's already protected in the `.env` file, which is in `.gitignore`.


### Error Handling and Data Gaps

#### Corrupt files in cache (truncated file / xarray backend error)

If you see errors like:

```
❌ error_local
👉 Real detail: File exists on S3 but download failed: Unable to synchronously open file (truncated file: eof = ...)
```

or:

```
👉 Real detail: did not find a match in any of xarray's currently installed IO backends
```

It means `goes2go` has corrupt `.nc` files in its local cache (`~/data/noaa-goes19/`). These were downloaded incompletely due to an internet interruption or timeout.

**Solution:** Delete the cache and retry:

```bash
rm -rf ~/data/noaa-goes19/
python pipeline.py retry --region uruguay
```

This is safe: the cache is just a temporary intermediary. Your actual dataset (the `.npy` files in `dataset/`) is not affected. The files will be re-downloaded from AWS automatically.

If you don't want to delete the entire cache, you can delete only the corrupt files (suspiciously small ones):

```bash
find ~/data/noaa-goes19/ -name "*.nc" -size -1M -delete
```

#### Data Gaps from NOAA

Notes on data availability (Gaps):
Some download commands may return FileNotFound or IndexError errors. This doesn't always indicate a script failure — it reflects the lack of operational data on NOAA's servers (AWS). The GOES-19 satellite began its operational phase on April 7, 2025; any earlier date will result in an error.

If a 2025 timestamp fails persistently, you can verify the file's existence directly in the AWS S3 bucket using the following command (requires AWS CLI):

```bash
aws s3 ls s3://noaa-goes19/ABI-L1b-RadF/2025/265/17/ --no-sign-request
```

If the command returns an empty list, the data is an official Data Gap from the satellite and is not available for download.
