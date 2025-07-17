import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === Config ===
csv1_path = Path("/home/leo/Scrivania/vs_code/Visual_Grouding-/metrics/coco_original.csv")
csv2_path = Path("/home/leo/Scrivania/vs_code/Visual_Grouding-/stanza_beta05_batch32.csv")
smooth_window = 5

# === Load ===
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# === Clean column names ===
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# === Define groups of metrics ===
group1 = ["train_loss", "val_mean_iou", "val_accuracy"]
group2 = ["rac_loss", "mrc_loss", "att_reg_loss"]
group3 = ["w_adw", "w_odw", "bbox_loss"]
all_columns = set(["epoch"] + group1 + group2 + group3)

# === Ensure numeric conversion ===
for col in all_columns:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")
    if col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

# === Drop rows with NaN in any required field ===
df1 = df1.dropna(subset=["epoch"])
df2 = df2.dropna(subset=["epoch"])

# === Align on common epochs ===
df1["epoch"] = df1["epoch"].astype(int)
df2["epoch"] = df2["epoch"].astype(int)
common_epochs = sorted(set(df1["epoch"]) & set(df2["epoch"]))
df1c = df1[df1["epoch"].isin(common_epochs)].copy().sort_values("epoch")
df2c = df2[df2["epoch"].isin(common_epochs)].copy().sort_values("epoch")

# === Rolling smoothing function ===
def rolling_smooth(series, window):
    return series.rolling(window=window, min_periods=1, center=False).mean()

# === Colors ===
colors = {
    "stanza": "blue",
    "coco": "orange"
}

# === Plotting function ===
def plot_group(df1, df2, group, title_prefix):
    for col in group:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Missing column '{col}' in one of the CSVs.")
        df1[f"{col}_smooth"] = rolling_smooth(df1[col], smooth_window)
        df2[f"{col}_smooth"] = rolling_smooth(df2[col], smooth_window)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for ax, col in zip(axs, group):
        ax.plot(df1["epoch"], df1[col], label="Coco base (raw)", alpha=0.3, linestyle="--", color=colors["stanza"])
        ax.plot(df2["epoch"], df2[col], label="Stanza batc32(raw)", alpha=0.3, linestyle="--", color=colors["coco"])
        ax.plot(df1["epoch"], df1[f"{col}_smooth"], label=f"Stanza (MA{smooth_window})", linewidth=2.2, color=colors["stanza"])
        ax.plot(df2["epoch"], df2[f"{col}_smooth"], label=f"Stanza batc32(MA{smooth_window})", linewidth=2.2, color=colors["coco"])
        ax.set_title(f"{title_prefix}: {col}")
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# === Plot each group ===
plot_group(df1c, df2c, group1, "Core Metrics")
plot_group(df1c, df2c, group2, "Loss Components")
plot_group(df1c, df2c, group3, "Weight/BBox Losses")
