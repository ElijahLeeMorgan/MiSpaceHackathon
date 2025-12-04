"""
Sea Ice Forecasting Prediction Script.

This script generates short-term sea ice concentration forecasts using a trained
U-Net model. It loads the most recent 7-day sequence from the processed dataset
and predicts the next 4 days. The model output is physically constrained, flipped
upright so north is at the top, and saved as both PNG images and a GIF.

Main improvements in this version:
- Images are rotated 180 degrees before export so geographic orientation is correct.
- Only the next 4 days are predicted.
- Output folder auto-increments so runs never overwrite each other.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from train_unet import UNet


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    "unet_epoch5.pth"
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed"
)

X_PATH = os.path.join(DATA_PATH, "X.npy")


# -------------------------------------------------------------
# AUTO-INCREMENT OUTPUT DIRECTORY
# -------------------------------------------------------------
base = os.path.join(os.path.dirname(__file__), "predictions_ver")
i = 1
OUT_DIR = f"{base}_{i}"
while os.path.exists(OUT_DIR):
    i += 1
    OUT_DIR = f"{base}_{i}"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Saving outputs to: {OUT_DIR}")


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
print("Loading dataset...")
X = np.load(X_PATH)
print("X shape:", X.shape)

# Last 7 days used as seed input
last_seq = X[-1].astype(np.float32)
print("Seed sequence shape:", last_seq.shape)


# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
print(f"Loading model checkpoint: {MODEL_PATH}")
model = UNet(in_channels=7, out_channels=1).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


# -------------------------------------------------------------
# LAND MASK (light blue land)
# -------------------------------------------------------------
print("Computing land mask...")

jan_maps = [X[i, -1] for i in range(len(X))]
jan_arr = np.stack(jan_maps)

# Land pixels never change and always stay 0 in real data
land_mask = (jan_arr.mean(axis=0) == 0)

LAND_VALUE = 0.35     # Light blue water visualization baseline


# -------------------------------------------------------------
# MULTI-STEP PREDICTOR (Now only 4 days)
# -------------------------------------------------------------
def predict_next_days(model, init_seq, steps=4):
    """
    Predicts the next N days of ice concentration using iterative forecasting.

    Maintains physical constraints:
    - Clamp 0–6 range
    - Preserve land mask
    - Max 20% daily melt
    - Smoothing using 70/30 blend
    """

    seq = init_seq.copy()
    preds = []

    for step in range(steps):

        inp = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp)
            pred = out.cpu().numpy()[0, 0]

        # 1. Clamp valid range
        pred = np.clip(pred, 0.0, 6.0)

        # 2. Land stays constant light blue
        pred[land_mask] = LAND_VALUE

        # 3. Ice cannot melt >20% per day
        pred = np.maximum(pred, seq[-1] * 0.80)

        # 4. Stability smoothing
        pred = 0.70 * pred + 0.30 * seq[-1]

        preds.append(pred)

        # Slide window forward
        seq = np.concatenate([seq[1:], pred[np.newaxis, ...]], axis=0)

        print(f"Predicted day {step + 1}/{steps}")

    return np.array(preds)


# -------------------------------------------------------------
# RUN PREDICTION (4 days)
# -------------------------------------------------------------
print("Generating next 4-day forecast...")
preds = predict_next_days(model, last_seq, steps=4)

# Save raw predictions
np.save(os.path.join(OUT_DIR, "4day_predictions.npy"), preds)


# -------------------------------------------------------------
# SAVE PNGs WITH FIXED SCALE + 180° ROTATION
# -------------------------------------------------------------
print("Saving PNG images...")

image_paths = []
day_names = ["Day1", "Day2", "Day3", "Day4"]

V_MIN = 0.0
V_MAX = 6.0

for pred, name in zip(preds, day_names):

    # 180-degree flip so north is up
    pred_flipped = np.rot90(pred, k=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(pred_flipped, cmap="Blues", vmin=V_MIN, vmax=V_MAX)
    plt.title(f"Ice Forecast – {name}")
    plt.colorbar(label="Ice Concentration (0–6 scale)")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    image_paths.append(save_path)


# -------------------------------------------------------------
# CREATE GIF
# -------------------------------------------------------------
print("Creating GIF...")

gif_path = os.path.join(OUT_DIR, "4day_forecast.gif")
frames = [imageio.imread(p) for p in image_paths]
imageio.mimsave(gif_path, frames, duration=0.5)

print("\nAll outputs saved in:", OUT_DIR)
print("GIF saved as:", gif_path)
print("Done.")
