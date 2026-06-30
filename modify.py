import glob
import os
import xarray as xr
import numpy as np

nc_files = sorted(glob.glob("./data/*.nc"))

for f in nc_files:
    print(f"Processing {os.path.basename(f)}")

    ds = xr.load_dataset(f)  # Load into memory

    # Compute mean over the time dimension
    mean_so2 = ds["SO2"].mean(dim="time")

    # Replace hour 13
    ds["SO2"][dict(time=13)] = mean_so2

    # Overwrite the original file
    tmp = f + ".tmp"
    ds.to_netcdf(tmp)
    ds.close()

    os.replace(tmp, f)

print("Done.")