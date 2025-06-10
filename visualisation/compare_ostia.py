import os
import xarray as xr
import cmocean
import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from utils.metrics import calculate_rmse
import torch

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker

# Load the datasets
ds_2020 = xr.open_dataset('data/ostia_2020.nc')
ds_1993_2020 = xr.open_dataset('data/ocean.nc')

# Select a specific date
date = '2020-05-01'

# lat and lon
lat_ostia = ds_2020.latitude.values
lon_ostia = ds_2020.longitude.values

lat = ds_1993_2020.latitude.values
lon = ds_1993_2020.longitude.values

# Extract sea surface variables for the specific date
sst_2020 = ds_2020.sel(time=date)['analysed_sst'] - 273.15
sst_1993_2020 = ds_1993_2020.sel(time=date)['thetao']
sst_1993_2020 = sst_1993_2020[0]

# Plot the data with cmocean colormap and include lat/lon values using cartopy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})

true_2020 = axes[0].contourf(lon_ostia, lat_ostia, sst_2020, cmap=cmocean.cm.thermal, transform=ccrs.PlateCarree())
axes[0].coastlines()
axes[0].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
axes[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
axes[0].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
axes[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
fig.colorbar(true_2020, ax=axes[0])
axes[0].set_title('Sea Surface Temperature - OSTIA Data')

true_1993_2020 = axes[1].contourf(lon, lat, sst_1993_2020, cmap=cmocean.cm.thermal, transform=ccrs.PlateCarree())
axes[1].coastlines()
axes[1].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
axes[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
axes[1].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
axes[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
fig.colorbar(true_2020, ax=axes[1])
axes[1].set_title('Sea Surface Temperature - Glorys Data')

# Create directory if it does not exist
output_dir = f"plots/ostia"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.tight_layout()
filename_plot = os.path.join(output_dir, f"compare_{date}.pdf")
plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
plt.close()


# Interpolate sst_1993_2020 to the same shape as sst_2020
sst_1993_2020_interp = sst_1993_2020.interp(latitude=sst_2020.latitude, longitude=sst_2020.longitude)


# Convert data to torch tensors
sst_2020_tensor = torch.tensor(sst_2020.values, dtype=torch.float32)
sst_1993_2020_interp_tensor = torch.tensor(sst_1993_2020_interp.values, dtype=torch.float32)

# Calculate RMSE using torch
rmse = torch.sqrt(torch.nanmean((sst_2020_tensor - sst_1993_2020_interp_tensor) ** 2)).item()
print(f"RMSE: {rmse}")

