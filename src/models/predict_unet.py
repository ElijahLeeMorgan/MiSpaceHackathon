"""Sea Ice Forecasting Prediction Script.

This module generates 28-day sea ice concentration forecasts using a trained U-Net model.
It loads historical ice data, applies the model iteratively to predict future states,
and produces visualizations including individual PNG images and an animated GIF.

The prediction process includes several constraints to maintain physical realism:
- Ice concentration clamping (0-6 scale)
- Land mask preservation
- Gradual melt constraint (maximum 20% per day)
- Temporal smoothing to prevent abrupt changes

Typical usage example:
    python predict_ice.py
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
# Device selection: GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the trained U-Net model checkpoint
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoints",
    "unet_epoch5.pth"
)

# Directory containing preprocessed training data
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed"
)

# Input features (X) and labels (y) file paths
X_PATH = os.path.join(DATA_PATH, "X.npy")
Y_PATH = os.path.join(DATA_PATH, "y.npy")


# -------------------------------------------------------------
# AUTO-INCREMENT OUTPUT FOLDER
# -------------------------------------------------------------
# Base directory name for storing prediction outputs
base = os.path.join(os.path.dirname(__file__), "predictions_feb_ver")

# Find the next available directory number to avoid overwriting
i = 1
OUT_DIR = f"{base}_{i}"
while os.path.exists(OUT_DIR):
    i += 1
    OUT_DIR = f"{base}_{i}"

# Create the output directory
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Saving outputs to: {OUT_DIR}")


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
print("Loading dataset...")
# Load the input sequences (shape: [num_samples, sequence_length, height, width])
X = np.load(X_PATH)
print("X shape:", X.shape)

# Extract the most recent 7-day sequence as the initial seed for prediction
last_seq = X[-1].astype(np.float32)
print("Using final 7-day window as seed:", last_seq.shape)


# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
print(f"Loading model: {MODEL_PATH}")
# Initialize U-Net with 7 input channels (7-day sequence) and 1 output channel (next day)
model = UNet(in_channels=7, out_channels=1).to(DEVICE)
# Load the trained model weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
# Set model to evaluation mode (disables dropout, batch norm training, etc.)
model.eval()


# -------------------------------------------------------------
# LAND–WATER MASK (light blue)
# -------------------------------------------------------------
print("Computing land–water mask...")

# Extract the last day of each sample (January data)
jan_maps = [X[i, -1] for i in range(len(X))]
jan_arr = np.stack(jan_maps)

# Land pixels are consistently zero across all samples
land_mask = (jan_arr.mean(axis=0) == 0)

# Light blue value for land visualization (0.35 on 0-6 scale)
LAND_VALUE = 0.35   


# -------------------------------------------------------------
# MULTISTEP PREDICTOR (28 days)
# -------------------------------------------------------------
def predict_future_month(model, init_seq, steps=28):
    """Generates multi-step sea ice predictions for a specified number of days.
    
    This function performs iterative forecasting by repeatedly applying the model
    to predict the next day's ice concentration, then using that prediction as
    input for the subsequent forecast. Several constraints are applied to maintain
    physical realism in the predictions.
    
    Args:
        model: The trained PyTorch U-Net model for ice prediction.
        init_seq: Initial 7-day sequence of ice concentrations (numpy array).
                  Shape: [7, height, width]
        steps: Number of days to forecast into the future. Defaults to 28.
    
    Returns:
        A numpy array of predicted ice concentrations with shape [steps, height, width],
        where each slice represents one day's forecast.
    
    Example:
        >>> predictions = predict_future_month(model, last_week_data, steps=28)
        >>> print(predictions.shape)  # (28, 448, 304)
    """
    # Create a working copy to avoid modifying the original sequence
    seq = init_seq.copy()
    preds = []

    for step in range(steps):
        # Prepare input tensor: add batch dimension and move to device
        inp = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        # Generate prediction without computing gradients (faster, less memory)
        with torch.no_grad():
            out = model(inp)
            # Extract prediction and move back to CPU as numpy array
            pred = out.cpu().numpy()[0, 0]

        # -----------------------------------------------------
        # 1. Clamp valid ice values (0 → 6)
        # -----------------------------------------------------
        # Ensure all values are within the valid ice concentration range
        pred = np.clip(pred, 0.0, 6.0)

        # -----------------------------------------------------
        # 2. Land stays light blue
        # -----------------------------------------------------
        # Override land pixels with constant value for visualization
        pred[land_mask] = LAND_VALUE

        # -----------------------------------------------------
        # 3. Persistence constraint: ice cannot melt >20% per day
        # -----------------------------------------------------
        # Physical constraint: ice melts gradually, not instantaneously
        # Predicted value must be at least 80% of previous day's value
        pred = np.maximum(pred, seq[-1] * 0.80)

        # -----------------------------------------------------
        # 4. Stabilizer blend to prevent collapse
        # -----------------------------------------------------
        # Temporal smoothing: blend 70% prediction with 30% previous state
        # Reduces abrupt changes and improves stability
        pred = 0.70 * pred + 0.30 * seq[-1]

        # Store the constrained prediction
        preds.append(pred)

        # Slide the temporal window forward by one day
        # Remove oldest day, append new prediction
        seq = np.concatenate([seq[1:], pred[np.newaxis, ...]], axis=0)

        # Progress indicator
        print(f"Predicted day {step + 1}/{steps}")

    # Stack all predictions into a single array
    return np.array(preds)


# -------------------------------------------------------------
# RUN PREDICTIONS
# -------------------------------------------------------------
print("Generating February predictions...")
# Generate 28 days of forecasts (full February)
feb_preds = predict_future_month(model, last_seq, steps=28)

# Save raw predictions for further analysis
np.save(os.path.join(OUT_DIR, "feb_predictions.npy"), feb_preds)


# -------------------------------------------------------------
# SAVE IMAGES WITH FIXED COLOR SCALE
# -------------------------------------------------------------
print("Saving PNGs...")

# List to store file paths for GIF creation
image_paths = []
# Generate day labels (Feb01, Feb02, ..., Feb28)
day_names = [f"Feb{d:02d}" for d in range(1, 29)]

# Fixed color scale for consistent visualization across all days
V_MIN = 0.0
V_MAX = 6.0

# Generate and save individual PNG images for each day
for pred, name in zip(feb_preds, day_names):
    # Create a new figure for each day
    plt.figure(figsize=(6, 6))
    # Display ice concentration with Blues colormap
    plt.imshow(pred, cmap="Blues", vmin=V_MIN, vmax=V_MAX)
    plt.title(f"Ice Forecast – {name}")
    plt.colorbar(label="Ice Concentration (0–6 scale)")
    plt.tight_layout()

    # Save the figure as PNG with high resolution
    save_path = os.path.join(OUT_DIR, f"{name}.png")
    image_paths.append(save_path)
    plt.savefig(save_path, dpi=150)
    # Close the figure to free memory
    plt.close()


# -------------------------------------------------------------
# CREATE GIF
# -------------------------------------------------------------
print("Creating GIF...")
# Output path for the animated GIF
gif_path = os.path.join(OUT_DIR, "feb_forecast.gif")
# Load all PNG frames into memory
frames = [imageio.imread(p) for p in image_paths]
# Create animated GIF with 0.4 second duration per frame
imageio.mimsave(gif_path, frames, duration=0.4)

# Final summary
print("\nAll outputs saved in:", OUT_DIR)
print("GIF created:", gif_path)
print("Done.")
