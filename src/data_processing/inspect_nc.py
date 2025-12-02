"""NetCDF File Visualization Tool.

This module provides an interactive command-line tool for visualizing 2D data
from NetCDF (.nc) files. It automatically detects the appropriate 2D variable
in the dataset and generates a heatmap visualization saved as a PNG image.

The tool supports multiple NetCDF engines (netcdf4 and h5netcdf) for maximum
compatibility with different file formats. Output images are saved to a
dedicated directory alongside the script.

Example:
    Run the script and provide a path when prompted:
        $ python inspect_nc.py
        Enter the path to ANY .nc file:
        Paste .nc file path here: /path/to/data.nc

Attributes:
    FILE (str): User-provided path to the NetCDF file to be processed.
    SCRIPT_DIR (str): Absolute path to the directory containing this script.
    OUT_DIR (str): Output directory path for generated visualizations.
    
Todo:
    * Add support for 3D visualizations with time-stepping
    * Allow user to select specific variables when multiple options exist
    * Add command-line argument parsing for batch processing
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

print("\n=== NetCDF Visualizer ===")
print("Enter the path to ANY .nc file:")
print("Example:")
print(r"C:\Users\msans\Downloads\MiSpace 2025\sortedByDay\sortedByDay\Jan30\Ice\netcdf\2019-01-30.nc\n")

# -----------------------------------------------------
# 1) Get user input
# -----------------------------------------------------
FILE = input("Paste .nc file path here: ").strip('"')

if not os.path.isfile(FILE):
    print("\nERROR: File does not exist.")
    exit()

print("\nOpening:", FILE)


# -----------------------------------------------------
# 2) Safe NetCDF loader
# -----------------------------------------------------
def open_nc(path):
    """Opens a NetCDF file using xarray with fallback engine support.
    
    This function attempts to open a NetCDF dataset using the 'netcdf4' engine.
    If that fails, it tries again with the 'h5netcdf' engine as a fallback. This
    approach provides flexibility for handling different NetCDF file formats and
    library availability.
    
    Args:
        path (str): The file path to the NetCDF file to be opened. Can be a
            relative or absolute path.
    
    Returns:
        xarray.Dataset: An xarray Dataset object containing the data from the
            NetCDF file. This object provides convenient access to variables,
            dimensions, and metadata.
    
    Raises:
        SystemExit: Exits the program with status code 1 if both engines fail
            to open the file. An error message is printed before exiting.
    
    Example:
        >>> dataset = open_nc("data/temperature.nc")
        >>> print(dataset.data_vars)
    
    Note:
        Requires xarray, netcdf4, and h5netcdf libraries to be installed.
        The function uses a broad exception handler on the first attempt,
        which may mask other errors. Consider catching specific exceptions
        for better error handling.
    """
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except:
        try:
            return xr.open_dataset(path, engine="h5netcdf")
        except Exception as e:
            print("Could not open file with either backend.")
            print("Error:", e)
            exit()


ds = open_nc(FILE)

print("\n=== DATASET STRUCTURE ===")
print(ds)
print("\nVariables:")
for v in ds.data_vars:
    print(f" - {v}: shape {ds[v].shape}")


# -----------------------------------------------------
# 3) Auto-detect the correct 2D variable to visualize
# -----------------------------------------------------
def pick_best_variable(ds):
    """Automatically selects the most suitable 2D variable for visualization.
    
    This function searches through all data variables in the xarray Dataset
    and returns the first variable that is either natively 2D or can be
    reduced to 2D (e.g., a 3D array with a singleton first dimension).
    
    Args:
        ds (xarray.Dataset): The xarray Dataset containing one or more data
            variables to search through.
    
    Returns:
        tuple: A tuple containing two elements:
            - var (str or None): The name of the selected variable, or None
              if no suitable variable is found.
            - arr (numpy.ndarray or None): The 2D numpy array containing the
              variable's data, or None if no suitable variable is found.
    
    Example:
        >>> var_name, data_array = pick_best_variable(dataset)
        >>> if var_name:
        ...     print(f"Selected: {var_name} with shape {data_array.shape}")
    
    Note:
        The function prioritizes variables in the order they appear in the
        dataset. For 3D arrays, only those with shape (1, rows, cols) are
        considered, and the singleton dimension is squeezed out.
    """
    for var in ds.data_vars:
        arr = ds[var].values

        if arr.ndim == 2:
            return var, arr

        if arr.ndim == 3 and arr.shape[0] == 1:
            return var, arr[0]

    return None, None


var, arr = pick_best_variable(ds)

if var is None:
    print("\nERROR: No 2D variable found to visualize.")
    exit()

print(f"\nAuto-selected variable for visualization: {var}")
print("Array shape:", arr.shape)


# -----------------------------------------------------
# 4) Output folder: src/nc_visualizer_outputs/
# -----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "nc_visualizer_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------
# 5) Generate visualization
# -----------------------------------------------------
plt.figure(figsize=(8, 7))
plt.imshow(arr, cmap="Blues")
plt.colorbar(label="Value")
plt.title(f"{var} visualization")
plt.tight_layout()

# Save output PNG
base = os.path.basename(FILE).replace(".nc", "")
OUT_PATH = os.path.join(OUT_DIR, f"{base}.png")

plt.savefig(OUT_PATH, dpi=150)
plt.close()

print("\nSaved PNG to:", OUT_PATH)
print("Done.\n")
