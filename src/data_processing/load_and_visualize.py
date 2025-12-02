"""Sea Ice Data Loading and Preprocessing Script.

This module loads daily sea ice NetCDF files from a hierarchical directory structure,
extracts temperature/ice concentration data, generates visualizations for each day,
and prepares sequential datasets for machine learning model training.

The data is organized in a sortedByDay directory structure where each day folder
contains Ice/netcdf subdirectories with NetCDF files. The script handles multiple
NetCDF engine backends for compatibility and creates 7-day input sequences for
predicting the next day's ice conditions.

Typical usage example:
    python load_ice_data.py
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# AUTOMATIC ROOT DIRECTORY SETUP (required)
# Data is outside MISpaceHackathon, so walk up three folders
# ---------------------------------------------------------
# Get the absolute path of this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up three levels to reach the sortedByDay directory
ROOT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "sortedByDay", "sortedByDay")
)

# ---------------------------------------------------------
# OUTPUT FOLDER INSIDE src/
# ---------------------------------------------------------
# Create output directory for daily ice visualizations
OUTPUT_VIS_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "daily_visualizations")
)
# Ensure the output directory exists
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


# ---------------------------------------------------------
# SAFE NETCDF LOADER
# ---------------------------------------------------------
def open_netcdf_safe(path):
    """Safely opens a NetCDF file using multiple backend engines.
    
    Attempts to open a NetCDF file first with the netcdf4 engine, then falls back
    to h5netcdf if the first attempt fails. This provides compatibility across
    different NetCDF file formats and system configurations.
    
    Args:
        path: String path to the NetCDF file to open.
    
    Returns:
        An xarray Dataset object if successful, None if both engines fail.
        Returns None and prints error messages if the file cannot be opened.
    
    Example:
        >>> ds = open_netcdf_safe("data/ice_20230101.nc")
        >>> if ds is not None:
        >>>     print(ds.data_vars)
    """
    # Try primary engine (netcdf4)
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except:
        # Fallback to alternative engine (h5netcdf)
        try:
            return xr.open_dataset(path, engine="h5netcdf")
        except Exception as e:
            # Both engines failed - report error
            print(f"Could not open file with either backend: {path}")
            print("Error:", e)
            return None


# ---------------------------------------------------------
# LOAD DAILY ICE MAPS
# ---------------------------------------------------------
def load_daily_ice_maps(root_dir):
    """Loads daily ice concentration maps from a hierarchical directory structure.
    
    Traverses a directory structure where each day folder contains Ice/netcdf
    subdirectories with NetCDF files. Extracts the 'temp' variable (ice/temperature
    data), handles multi-dimensional arrays, and converts NaN values to zero.
    
    Directory structure expected:
        root_dir/
            day1/
                Ice/
                    netcdf/
                        file.nc
            day2/
                ...
    
    Args:
        root_dir: String path to the root directory containing day folders.
    
    Returns:
        A tuple containing:
            - ice_maps: Numpy array of shape [num_days, height, width] with ice data
            - valid_days: List of day folder names that were successfully loaded
    
    Example:
        >>> maps, days = load_daily_ice_maps("/data/sortedByDay")
        >>> print(f"Loaded {len(days)} days with shape {maps.shape}")
    """
    # Get sorted list of day folders (ensures chronological order)
    day_folders = sorted(os.listdir(root_dir))
    ice_maps = []
    valid_days = []

    for day in day_folders:
        # Construct path to Ice/netcdf subdirectory
        ice_nc_dir = os.path.join(root_dir, day, "Ice", "netcdf")
        # Skip if this day doesn't have the expected directory structure
        if not os.path.isdir(ice_nc_dir):
            continue

        # Find all NetCDF files in this directory
        nc_files = [f for f in os.listdir(ice_nc_dir) if f.endswith(".nc")]
        # Skip if no NetCDF files found
        if not nc_files:
            continue

        # Load the first NetCDF file found
        file_path = os.path.join(ice_nc_dir, nc_files[0])
        ds = open_netcdf_safe(file_path)
        # Skip if file couldn't be opened
        if ds is None:
            continue

        # Extract the 'temp' variable (contains ice/temperature data)
        # Expected shape: (1, 1024, 1024) or (1024, 1024)
        if "temp" not in ds.data_vars:
            print("No temp variable in:", file_path)
            continue

        arr = ds["temp"].values  
        # Handle 3D arrays by extracting the first slice
        if arr.ndim == 3:
            arr = arr[0]  # Result: (1024, 1024)

        # Replace NaN values with zero for numerical stability
        arr = np.nan_to_num(arr)

        # Store the processed array and corresponding day label
        ice_maps.append(arr)
        valid_days.append(day)
        print(f"Loaded {day}: shape {arr.shape}")

    # Stack all daily maps into a single array
    return np.array(ice_maps), valid_days


# ---------------------------------------------------------
# RUN LOADING
# ---------------------------------------------------------
print("\nLoading daily NetCDF files...\n")
# Load all available ice maps from the data directory
ice_maps, valid_days = load_daily_ice_maps(ROOT_DIR)

# Display loading summary
print(f"\nLoaded {len(ice_maps)} days.")
print("Valid Days:", valid_days, "\n")


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
def visualize_ice_map(arr, day_name):
    """Generates and saves a visualization of a single day's ice concentration map.
    
    Creates a matplotlib figure showing the ice map with a Blues colormap,
    adds appropriate labels and title, and saves the result as a high-resolution PNG.
    
    Args:
        arr: 2D numpy array of shape [height, width] containing ice concentration values.
        day_name: String identifier for the day (used in title and filename).
    
    Returns:
        None. Saves the visualization to OUTPUT_VIS_DIR/{day_name}.png and prints
        the save path.
    
    Example:
        >>> ice_data = np.random.rand(1024, 1024)
        >>> visualize_ice_map(ice_data, "2023-01-15")
    """
    # Create a new figure with specified dimensions
    plt.figure(figsize=(8, 7))
    # Display the ice map using Blues colormap (darker = more ice)
    plt.imshow(arr, cmap="Blues")
    # Add colorbar with descriptive label
    plt.colorbar(label="Ice / Temperature Value")
    # Set title with day identifier
    plt.title(f"Ice Map - {day_name}")
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Construct save path and save figure at high resolution
    save_path = os.path.join(OUTPUT_VIS_DIR, f"{day_name}.png")
    plt.savefig(save_path, dpi=150)
    # Close figure to free memory
    plt.close()

    # Confirm save location
    print("Saved:", save_path)


# Generate visualizations for all loaded days
print("Generating visualizations...\n")
for arr, day in zip(ice_maps, valid_days):
    visualize_ice_map(arr, day)


# ---------------------------------------------------------
# CREATE ML INPUT (7→1 sequence)
# ---------------------------------------------------------
def stack_days(arr, seq_len=7):
    """Creates sequential training data from time-series ice maps.
    
    Generates (input, target) pairs where the input is a sequence of consecutive
    days and the target is the following day. This creates a supervised learning
    dataset for predicting the next day's ice conditions from the previous week.
    
    Uses a sliding window approach: for day i, takes days [i:i+seq_len] as input
    and day [i+seq_len] as the target.
    
    Args:
        arr: 3D numpy array of shape [num_days, height, width] containing all daily maps.
        seq_len: Integer length of input sequence (number of historical days). 
                 Defaults to 7 days.
    
    Returns:
        A tuple containing:
            - X: Numpy array of shape [num_samples, seq_len, height, width] with 
                 input sequences
            - y: Numpy array of shape [num_samples, height, width] with target days
    
    Example:
        >>> ice_data = np.random.rand(100, 1024, 1024)  # 100 days
        >>> X, y = stack_days(ice_data, seq_len=7)
        >>> print(X.shape)  # (93, 7, 1024, 1024)
        >>> print(y.shape)  # (93, 1024, 1024)
    """
    X, y = [], []
    # Iterate through all possible sequence positions
    for i in range(len(arr) - seq_len):
        # Input: sequence of seq_len consecutive days
        X.append(arr[i:i+seq_len])
        # Target: the day immediately following the sequence
        y.append(arr[i+seq_len])
    # Convert lists to numpy arrays for efficient computation
    return np.array(X), np.array(y)


# Create the sequential dataset (7 days input → 1 day output)
X, y = stack_days(ice_maps)

# Display dataset dimensions
print("\nML Dataset Ready:")
print("X shape:", X.shape)  # Expected: (num_samples, 7, 1024, 1024)
print("y shape:", y.shape)  # Expected: (num_samples, 1024, 1024)

# ---------------------------------------------------------
# SAVE RAW DATA FOR DOWNSAMPLING
# ---------------------------------------------------------
# Construct path for cached raw data file
cache_path = os.path.join(os.path.dirname(__file__), "cached_raw_ice_maps.npy")
# Save both the ice maps and day labels for later use
# allow_pickle=True is required for saving dictionaries
np.save(cache_path, {"ice_maps": ice_maps, "valid_days": valid_days}, allow_pickle=True)

# Confirm cache location
print("\nCached raw maps to:")
print(cache_path)
