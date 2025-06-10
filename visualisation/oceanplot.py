import os
import netCDF4 as nc
import cmocean
import numpy as np

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point

# Load the .nc file
file_path = 'data/ocean.nc'
dataset = nc.Dataset(file_path)

# Extract variables
thetao = dataset.variables['thetao'][1]
so = dataset.variables['so'][1]
uo = dataset.variables['uo'][1]
vo = dataset.variables['vo'][1]
zos = dataset.variables['zos'][1]

lat = dataset.variables['latitude'][:]
lon = dataset.variables['longitude'][:]

# Create directory if it does not exist
output_dir = f"plots/ocean/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a function to plot the variables
def plot_variable(data, title, cmap):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    bay_of_bengal_extent = [77, 99, 4, 23]
    # data, lon_cyclic = add_cyclic_point(data, coord=lon)
    cs = ax.contourf(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent(bay_of_bengal_extent, crs=ccrs.PlateCarree())
    ax.set_title(title)
    cbar = fig.colorbar(cs, ax=ax, orientation='vertical', pad=0.02, aspect=16, shrink=0.8)
    cbar.set_label(title)
    plt.savefig(f"{output_dir}{title.replace(' ', '_').lower()}_2.png")
    plt.close(fig)

# Plot each variable with its respective colormap
plot_variable(thetao[0, :, :], 'Sea Water Potential Temperature', cmocean.cm.thermal)
plot_variable(so[ 0, :, :], 'Sea Water Salinity', cmocean.cm.haline)
plot_variable(uo[0, :, :], 'Eastward Sea Water Velocity', cmocean.cm.speed)
plot_variable(vo[0, :, :], 'Northward Sea Water Velocity', cmocean.cm.speed)
plot_variable(zos[:, :], 'Sea Surface Height', cmocean.cm.topo)

# Close the dataset
dataset.close()