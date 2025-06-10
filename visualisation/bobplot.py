# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# def plot_bay_of_bengal():
#     # Define the Bay of Bengal region boundaries (approximate)
#     # Bay of Bengal is roughly from 5°N to 22°N latitude and 80°E to 100°E longitude
#     lon_min, lon_max = 77, 99
#     lat_min, lat_max = 4, 23
    
#     # Create a new figure and axes with Mercator projection
#     plt.figure(figsize=(12, 10))
#     ax = plt.axes(projection=ccrs.Mercator())
    
#     # Set the extent of the map to focus on Bay of Bengal
#     ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
#     # Add natural earth features
#     ax.add_feature(cfeature.LAND, facecolor='lightgray')
#     ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
#     ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.5)
    
#     # Add gridlines for latitude and longitude
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                       linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False
    
#     # Add title and annotations
#     plt.title('Bay of Bengal Region', fontsize=16)
#     plt.annotate('India', xy=(83, 18), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#                  fontsize=12, color='black')
#     plt.annotate('Sri Lanka', xy=(81, 8), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#                  fontsize=10, color='black')
#     plt.annotate('Bangladesh', xy=(90, 20), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#                  fontsize=10, color='black')
#     plt.annotate('Myanmar', xy=(95, 18), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#                  fontsize=10, color='black')
#     plt.annotate('Thailand', xy=(98, 12), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
#                  fontsize=10, color='black')
    
#     # Add a text explaining what is Bay of Bengal
#     plt.figtext(0.5, 0.02, "Bay of Bengal: Northern extension of the Indian Ocean between India and Myanmar",
#                 ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
#     # Save the figure if needed
#     plt.savefig('bay_of_bengal_map.pdf', dpi=300, bbox_inches='tight')
    
#     # plt.show()

# if __name__ == "__main__":
#     plot_bay_of_bengal()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create a figure with two subplots - global map and Bay of Bengal
fig = plt.figure(figsize=(15, 10))

# ----- Global Ocean Map -----
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Add natural earth features
ax1.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)

# Add gridlines
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Define Bay of Bengal region and highlight it
bay_bounds = [77, 99, 4, 23]  # [min_lon, max_lon, min_lat, max_lat]
rect = plt.Rectangle((bay_bounds[0], bay_bounds[2]), 
                     bay_bounds[1] - bay_bounds[0], 
                     bay_bounds[3] - bay_bounds[2],
                     facecolor='none', edgecolor='red', linewidth=2, 
                     transform=ccrs.PlateCarree())
ax1.add_patch(rect)

ax1.set_title('Global Ocean Map with Bay of Bengal Highlighted', fontsize=12)

# ----- Bay of Bengal Detailed Map -----
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent(bay_bounds, crs=ccrs.PlateCarree())

# Add features
ax2.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax2.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax2.add_feature(cfeature.COASTLINE, linewidth=1)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
ax2.add_feature(cfeature.RIVERS, linewidth=0.5, facecolor='blue')

# Add gridlines
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Add country names
countries = ["India", "Bangladesh", "Myanmar", "Sri Lanka"]
country_coords = {
    "India": (78, 20),
    "Bangladesh": (90, 21.5),
    "Myanmar": (97, 20),
    "Sri Lanka": (81, 7)
}

for country, coords in country_coords.items():
    ax2.text(coords[0], coords[1], country, transform=ccrs.PlateCarree(),
             fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

# Add Bay of Bengal label
ax2.text(90, 15, "Bay of Bengal", transform=ccrs.PlateCarree(),
         fontsize=10, ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5))

ax2.set_title('Bay of Bengal Region', fontsize=12)

# Add depth contours (simulated)
# In a real application, you would use actual bathymetry data
# x = np.linspace(bay_bounds[0], bay_bounds[1], 100)
# y = np.linspace(bay_bounds[2], bay_bounds[3], 100)
# X, Y = np.meshgrid(x, y)

# # Simulate depth - deeper as we go east and north
# Z = -((X - 80) * 50 + (Y - 5) * 30)

# # Create contour for depths
# levels = np.arange(-5000, 0, 500)
# contour = ax2.contour(X, Y, Z, levels=levels, colors='navy', alpha=0.6, linewidths=0.5)
# ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.0f m')

plt.tight_layout()
plt.savefig('bay_of_bengal_map.pdf', dpi=300, bbox_inches='tight')
plt.show()