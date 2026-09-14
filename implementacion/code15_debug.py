'''
Debugging of fire code 15 false positives
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fdca.planck import planck_temp

'''
Get timestamps
'''
time_ini = "2025-11-20 00:00"
time_end = "2025-11-28 06:00"
timestep = 2 #Hours
dates = pd.date_range(
    start=time_ini,
    end=time_end,
    freq=f"{timestep}h"
)
timestamps = dates.strftime("%Y%m%d_%H%M").tolist()
N_timestamps = len(timestamps)

'''
Dirs
'''
output_dir = "results/audit/20260905_181446/" 
noaa_dir = "dataset/uruguay/ABI-L2-FDCF-Mask/"
band7_dir = "dataset/uruguay/ABI-L1b-Rad-B07/"
band14_dir = "dataset/uruguay/ABI-L1b-Rad-B14/"


'''
Aux functions
'''
def local_median(arr, size=5):
    """Median of the surrounding pixels, excluding the center pixel."""
    pad = size // 2

    padded = np.pad(
        arr,
        pad,
        mode="constant",
        constant_values=np.nan
    )

    windows = []

    for i in range(size):
        for j in range(size):
            window = padded[
                i:i + arr.shape[0],
                j:j + arr.shape[1]
            ]

            # Exclude the center pixel
            if i == pad and j == pad:
                window = np.full_like(window, np.nan)

            windows.append(window)

    return np.nanmedian(np.stack(windows), axis=0)



'''
Analysis
'''
results = []
all_pixels = []

for t in  timestamps:


    noaa_output = np.load(noaa_dir + str(t) + ".npy") #Official NOAA mask
    impl_output = np.load(output_dir + str(t) + "/arrays.npz")["fire_mask_p2"]   #Code output

    b07 = np.load(band7_dir  + str(t) + ".npy")
    b14 = np.load(band14_dir + str(t) + ".npy")
    b07 = planck_temp(7,b07)
    b14 = planck_temp(14,b14)

    fp15 = (impl_output == 15) & (noaa_output != 15)
    tp15 = (impl_output == 15) & (noaa_output == 15)
    fn15 = (impl_output != 15) & (noaa_output == 15)

    '''n_code15 = np.sum(impl_output == 15)
    n_valid = np.sum(impl_output != 0)
    print(t, n_code15, n_valid, n_code15 / n_valid)'''


    mask = impl_output == 15
    rows, cols = np.where(mask)

    df_t = pd.DataFrame({
        "timestamp": t,
        "row": rows,
        "col": cols,

        "impl_output": impl_output[mask],
        "noaa_output": noaa_output[mask],

        "b07": b07[mask],
        "b14": b14[mask],

        "category": np.where(
            noaa_output[mask] == 15,
            "TP15",
            "FP15"
        )
    })

    df_t["b07_minus_b14"] = (
        df_t["b07"] - df_t["b14"]
    )

    all_pixels.append(df_t)

    # --------------------------------------------------
    # Calculate local background
    # --------------------------------------------------

    b07_background = local_median(b07, size=5)
    b14_background = local_median(b14, size=5)

    # How much hotter is the pixel than its surroundings?
    b07_contrast = b07 - b07_background
    b14_contrast = b14 - b14_background

    # --------------------------------------------------
    # Find code-15 pixels
    # --------------------------------------------------

    code15 = impl_output == 15

    rows, cols = np.where(code15)

    for r, c in zip(rows, cols):

        results.append({
            "timestamp": t,
            "row": r,
            "col": c,

            "category": (
                "TP15"
                if noaa_output[r, c] == 15
                else "FP15"
            ),

            "b07": b07[r, c],
            "b14": b14[r, c],

            "b07_background": b07_background[r, c],
            "b14_background": b14_background[r, c],

            "b07_contrast": b07_contrast[r, c],
            "b14_contrast": b14_contrast[r, c],
        })


df_spatial = pd.DataFrame(results)
print(
    df_spatial
    .groupby("category")[["b07_contrast", "b14_contrast"]]
    .agg(["count", "mean", "median", "min", "max"])
)
'''for threshold in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:

    fp = df_spatial[
        (df_spatial["category"] == "FP15") &
        (df_spatial["b07_contrast"] >= threshold)
    ]

    tp = df_spatial[
        (df_spatial["category"] == "TP15") &
        (df_spatial["b07_contrast"] >= threshold)
    ]

    print(
        f"{threshold:4.2f} K : "
        f"FP15 = {len(fp):5d}, "
        f"TP15 = {len(tp):2d}"
    )

for hour in sorted(df_spatial["timestamp"].str[9:11].astype(int).unique()):

    d = df_spatial[
        df_spatial["timestamp"].str[9:11].astype(int) == hour
    ]

    fp = d[d["category"] == "FP15"]
    tp = d[d["category"] == "TP15"]

    fp_1k = (fp["b07_contrast"] >= 1.0).sum()
    tp_1k = (tp["b07_contrast"] >= 1.0).sum()

    print(
        f"{hour:02d} UTC: "
        f"FP15 >= 1K: {fp_1k}/{len(fp)}, "
        f"TP15 >= 1K: {tp_1k}/{len(tp)}"
    )'''
'''for hour in [12, 14, 16, 18, 20]:

    d = df_spatial[
        df_spatial["timestamp"].str[9:11].astype(int) == hour
    ]

    print(f"\n===== {hour:02d} UTC =====")

    for category in ["FP15", "TP15"]:

        x = d.loc[
            d["category"] == category,
            "b07_contrast"
        ]

        if len(x) == 0:
            print(f"{category}: no pixels")
            continue

        print(
            f"{category}: "
            f"n={len(x)}, "
            f"p10={x.quantile(0.10):.3f}, "
            f"p25={x.quantile(0.25):.3f}, "
            f"median={x.median():.3f}, "
            f"p75={x.quantile(0.75):.3f}, "
            f"p90={x.quantile(0.90):.3f}"
        )

'''


# Combine all timestamps
df = pd.concat(
    all_pixels,
    ignore_index=True
)




def count(df):
    print("General count:")

    counts = df["category"].value_counts()

    fp15_count = counts.get("FP15", 0)
    tp15_count = counts.get("TP15", 0)

    print("FP15:", fp15_count)
    print("TP15:", tp15_count)

    print(
        "FP rate:",
        fp15_count / (fp15_count + tp15_count)
    )

def b7_b14_info(df):
    print("Bands 7 and 14 info:")

    for category in ["TP15", "FP15"]:

        print()
        print("-" * 50)
        print(category)
        print("-" * 50)

        subset = df[df["category"] == category]

        for feature in [
            "b07",
            "b14",
            "b07_minus_b14"
        ]:

            print()
            print(feature)

            print(
                subset[feature].quantile(
                    [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
                )
            )

        print("-"*50)
        print("difference bins")
        print("-"*50)

        df["b07_b14_bin"] = pd.cut(
            df["b07_minus_b14"],
            bins=20
        )
        spectral_table = pd.crosstab(
            df["b07_b14_bin"],
            df["category"],
            normalize="index"
        )

        print(spectral_table)

        print("-"*50)
        print("spectral counts")
        print("-"*50)
        spectral_counts = (
            df
            .groupby("b07_b14_bin", observed=True)
            .agg(
                total=("category", "size"),
                fp15=("category", lambda x: (x == "FP15").sum()),
                tp15=("category", lambda x: (x == "TP15").sum()),
            )
        )

        spectral_counts["fp_rate"] = (
            spectral_counts["fp15"] /
            spectral_counts["total"]
        )

        print(spectral_counts)


        df["hour"] = pd.to_datetime(
            df["timestamp"],
            format="%Y%m%d_%H%M"
        ).dt.hour

        summary = (
            df.groupby(["hour", "category"])[["b07", "b14", "b07_minus_b14"]]
            .agg(["count", "mean", "median", "min", "max"])
        )

        print(summary.to_string())


def timestamp_stats(df):
    timestamp_stats = (
    df
    .groupby(["timestamp", "category"])
    .agg(
        pixels=("category", "size"),
        b07_mean=("b07", "mean"),
        b07_median=("b07", "median"),
        b14_mean=("b14", "mean"),
        b14_median=("b14", "median"),
        b07_b14_mean=("b07_minus_b14", "mean"),
        b07_b14_median=("b07_minus_b14", "median"),
    )
    )

    print(timestamp_stats)


    fp_tp_counts = pd.crosstab(
    df["timestamp"],
    df["category"]
    )

    print(fp_tp_counts)

    df.groupby(["timestamp", "category"]).size().unstack(fill_value=0)

    counts = (
    df.groupby(["timestamp", "category"])
      .size()
      .unstack(fill_value=0)
    )

    counts["fp_rate"] = counts["FP15"] / counts.sum(axis=1)

    print(counts.to_string())



    df["hour"] = pd.to_datetime(
        df["timestamp"],
        format="%Y%m%d_%H%M"
    ).dt.hour

    hourly = (
        df.groupby(["hour", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["FP15", "TP15"], fill_value=0)
    )

    hourly["fp_rate"] = hourly["FP15"] / (
        hourly["FP15"] + hourly["TP15"]
    )

    print(hourly.to_string())

def plot_evolution():
    fig, ax = plt.subplots(figsize=(10, 8))

    # First timestamp
    t = timestamps[0]

    noaa_output = np.load(noaa_dir + t + ".npy")
    impl_output = np.load(
        output_dir + t + "/arrays.npz"
    )["fire_mask_p2"]
    b07 = np.load(band7_dir + t + ".npy")

    fp15 = (impl_output == 15) & (noaa_output != 15)

    # Band 7
    im = ax.imshow(
        b07,
        cmap="inferno",
        origin="upper"
    )

    # Green FP pixels
    rows, cols = np.where(fp15)

    fp = ax.scatter(
        cols,
        rows,
        s=10,
        c="green",
        marker="s"
    )

    title = ax.set_title(f"Band 7 — FP15 — {t}")

    plt.tight_layout()
    plt.show(block=False)


    # --------------------------------------------------
    # Animation
    # --------------------------------------------------

    for t in timestamps:

        noaa_output = np.load(noaa_dir + t + ".npy")

        impl_output = np.load(
            output_dir + t + "/arrays.npz"
        )["fire_mask_p2"]

        b07 = np.load(band7_dir + t + ".npy")

        fp15 = (impl_output == 15) & (noaa_output != 15)

        # Update Band 7
        im.set_data(b07)

        # Update FP locations
        rows, cols = np.where(fp15)
        fp.set_offsets(np.column_stack((cols, rows)))

        # Update title
        title.set_text(
            f"Band 7 — FP15 — {t} ({len(rows)} pixels)"
        )

        # Redraw
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Wait ~1 second
        plt.pause(1.0)

    plt.close(fig)

#count(df)
#b7_b14_info(df)
#timestamp_stats(df)
plot_evolution()