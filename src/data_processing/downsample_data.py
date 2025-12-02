import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

# ======================================================================
# CONFIGURATION
# ======================================================================
# Toggle downsampling:
# False  -> use full-resolution 1024×1024 maps
# True   -> output 256×256 downsampled maps
DOWNSAMPLE = False
TARGET_SIZE = 256  # used only if DOWNSAMPLE = True

# ======================================================================
# PATHS
# ======================================================================
# Path to cached raw dataset created by load_and_visualize.py
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "cached_raw_ice_maps.npy")

# Output folder for 256×256 or full-res visualizations
VIS_OUT_DIR = os.path.join(os.path.dirname(__file__), "downsampled_visualizations")
os.makedirs(VIS_OUT_DIR, exist_ok=True)

# ML dataset save paths
DATA_OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
os.makedirs(DATA_OUT_DIR, exist_ok=True)

SAVE_X = os.path.join(DATA_OUT_DIR, "X.npy")         # training input
SAVE_Y = os.path.join(DATA_OUT_DIR, "y.npy")         # labels (next-day maps)

# ======================================================================
# LOAD CACHED RAW DATA
# ======================================================================
if not os.path.isfile(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"ERROR: Missing raw cache file.\nExpected at:\n  {RAW_DATA_PATH}\n\n"
        "Run load_and_visualize.py first."
    )

raw = np.load(RAW_DATA_PATH, allow_pickle=True).item()
ice_maps = raw["ice_maps"]         # list/array of daily 1024×1024 maps
valid_days = raw["valid_days"]     # matching day labels

print(f"Loaded raw dataset: {ice_maps.shape} (days, H, W)")


# ======================================================================
# OPTIONAL: DOWNSAMPLING
# ======================================================================
def downsample(arr, new_size=256):
    """Resize a 2D image to new_size x new_size using anti-aliasing."""
    return resize(arr, (new_size, new_size), mode="reflect", anti_aliasing=True)


print("\n=== Processing dataset ===")
if DOWNSAMPLE:
    print(f"Downsampling ENABLED → {TARGET_SIZE}×{TARGET_SIZE}")
    maps_processed = np.array([downsample(day, TARGET_SIZE) for day in ice_maps])
else:
    print("Downsampling DISABLED → using FULL resolution 1024×1024")
    maps_processed = np.array(ice_maps)

print("Processed dataset shape:", maps_processed.shape)


# ======================================================================
# SAVE VISUALIZATIONS (full or downsampled)
# ======================================================================
print("\nSaving visualization PNGs...")

for arr, name in zip(maps_processed, valid_days):
    plt.figure(figsize=(7, 6))
    plt.imshow(arr, cmap="Blues")
    plt.title(f"{name} ({arr.shape[0]}x{arr.shape[1]})")
    plt.colorbar()
    plt.tight_layout()

    out_path = os.path.join(VIS_OUT_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

print(f"Saved PNG files to:\n  {VIS_OUT_DIR}")


# ======================================================================
# BUILD ML DATASET (7-day input → next-day prediction)
# ======================================================================
def stack_days(arr, seq_len=7):
    """Create sliding-window sequences for time series prediction.
    
    This function generates supervised learning datasets from time series data by
    creating fixed-length sliding windows. Each window of `seq_len` consecutive
    values becomes a feature sequence (X), and the next value becomes the target (y).
    
    Args:
        arr (np.ndarray): Input time series array of shape (n_samples,) or (n_samples, n_features).
        seq_len (int, optional): Length of the sliding window sequence. Defaults to 7.
    
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Feature sequences of shape (n_samples - seq_len, seq_len) or 
              (n_samples - seq_len, seq_len, n_features).
            - y (np.ndarray): Target values of shape (n_samples - seq_len,) or 
              (n_samples - seq_len, n_features).
    
    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> X, y = stack_days(data, seq_len=3)
        >>> X.shape
        (7, 3)
        >>> y.shape
        (7,)
    
    Note:
        The output size is (len(arr) - seq_len) samples. The last seq_len values
        cannot form a complete window with a corresponding target, so they are excluded.
    """
    """Create sliding-window sequences of length 7 → 1."""
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len])
    return np.array(X), np.array(y)


X, y = stack_days(maps_processed)

print("\n=== ML Dataset Shapes ===")
print("X:", X.shape)   # (N, 7, H, W)
print("y:", y.shape)   # (N, H, W)


# ======================================================================
# SAVE ML DATASET
# ======================================================================
np.save(SAVE_X, X)
np.save(SAVE_Y, y)

print("\nSaved dataset files:")
print(f" • {SAVE_X}")
print(f" • {SAVE_Y}")

print("\nDataset READY for UNet training.")
