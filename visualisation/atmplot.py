import os
import netCDF4 as nc
import numpy as np

import matplotlib.pyplot as plt

# Load the .nc file
file_path = 'data/atm.nc'
dataset = nc.Dataset(file_path)

# Extract variables
ssr = dataset.variables['ssr'][0,0]
tp = dataset.variables['tp'][0,0]
u10 = dataset.variables['u10'][0,0]
v10 = dataset.variables['v10'][0,0]
msl = dataset.variables['msl'][0,0]
tcc = dataset.variables['tcc'][0,0]

# Extract coordinates
lat = dataset.variables['latitude']
lon = dataset.variables['longitude']

# Create a meshgrid for plotting
lon, lat = np.meshgrid(lon, lat)

# Create directory if it does not exist
output_dir = f"plots/atm/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to plot each variable
def plot_variable(data, title, cmap):
    plt.figure(figsize=(10, 6))
    
    # Get the dimensions of your data
    nlat, nlon = data.shape
    
    # Create proper extent for imshow (which centers pixels)
    lon_extended = np.linspace(lon.min(), lon.max(), nlon+1)
    lat_extended = np.linspace(lat.min(), lat.max(), nlat+1)
    
    # Use pcolormesh instead of contourf
    plt.pcolormesh(lon_extended, lat_extended, data, cmap=cmap)
    
    # Add a colorbar
    # plt.colorbar(label=title)
    
    # Add grid at the exact cell boundaries with fewer grid cells
    # plt.grid(True, color='black', linewidth=2)
    # plt.gca().set_xticks(lon_extended)
    # plt.gca().set_yticks(lat_extended)
    
    # Labels and title
    plt.title(title)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    
    # Save figure
    plt.savefig(f"{output_dir}{title.replace(' ', '_').lower()}.png")
    plt.close()

# Plot each variable with respective color maps
plot_variable(ssr, 'Surface Solar Radiation', 'inferno')
plot_variable(tp, 'Total Precipitation', 'Blues')
plot_variable(u10, '10m U Wind Component', 'coolwarm')
plot_variable(v10, '10m V Wind Component', 'coolwarm')
plot_variable(msl, 'Mean Sea Level Pressure', 'jet')
plot_variable(tcc, 'Total Cloud Cover', 'gray')