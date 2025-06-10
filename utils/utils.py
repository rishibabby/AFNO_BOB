# Importing Libraries
import os
import torch
import numpy as np
import torch.nn.functional as F

# Plot
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Data preprocessing
def data_preprocess(data, mean=None, var=None, variable=None, type=None):
    """
    Preprocess the input data based on the variable type and other parameters.

    Parameters:
    data (numpy array or torch tensor): Input data to preprocess.
    mean (numpy array or torch tensor, optional): Mean values for normalization.
    std (numpy array or torch tensor, optional): Standard deviation values for normalization.
    variable (str, optional): Type of variable (e.g., 'ssr', 'tp', 'msl', 'thetao', 'so').
    type (str, optional): Type of data (e.g., 'atm').

    Returns:
    torch tensor: Preprocessed data.
    """
    if type == 'atm':
        if variable in ['ssr', 'tp', 'msl']:
            mean = mean
            std = np.sqrt(var)
            data = (data - mean) 
            data = data / std
        sample = torch.tensor(data).unsqueeze(0)
        sample = F.interpolate(sample, size=(224, 224), mode='bilinear', align_corners=False)
        return sample.reshape(1, 1, 224, 224)

    if variable in ["thetao", "so"]:
        data = data - mean

    data = torch.tensor(data, dtype=torch.float32) if not torch.is_tensor(data) else data.clone().detach().float()
    sample_mean = torch.nanmean(data)
    data = torch.where(torch.isnan(data), sample_mean, data)
    data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
    data[:, :, -20:, :] = 0.0

    return data

# Data postprocessing
def data_postprocess(data, mask, mean=None, variable=None):
    """
    Postprocess the output data based on the variable type and other parameters.

    Parameters:
    data (torch tensor): Input data to postprocess.
    data_mean (torch tensor, optional): Mean values for denormalization.
    mask (torch tensor, optional): Mask to apply on the data.
    original_size (tuple, optional): Original size to resize the data.
    variable (str, optional): Type of variable (e.g., 'thetao', 'so', 'zos').

    Returns:
    torch tensor: Postprocessed data.
    """
    data = F.interpolate(data, size=(229, 265), mode='bilinear', align_corners=False)
    
    if variable in ["thetao", "so"]:
        data = data + mean

    mask = torch.where(mask == 0.0, float('nan'), mask)
    data = data * mask
    data[:, :, -20:, :] = float('nan')

    return data

# Mask creation
def create_mask(data):
    """
    Create a mask for the input data based on NaN values.

    Parameters:
    data (numpy array or torch tensor): Input data to create the mask from.

    Returns:
    tuple: Length of the dataset and the created mask.
    """
    dataset_length, _, _, _ = data.shape
    first_timestep = data[0]
    mask = ~np.isnan(first_timestep)
    mask = mask.astype(np.float32)
    if mask.ndim == 3:
        mask = mask[np.newaxis, :, :, :]
    return dataset_length, mask

# Align datasets by date
def align_datasets_by_date(data_file_temp, data_file_atm, input_day):
    """
    Aligns two datasets by date and returns the matching date and index.
    Parameters:
    data_file_temp (xarray.Dataset): The temperature dataset containing a 'time' coordinate.
    data_file_atm (xarray.Dataset): The atmospheric dataset containing a 'time' coordinate.
    input_day (int): The index of the day in the temperature dataset to align with the atmospheric dataset.
    Returns:
    tuple: A tuple containing:
        - temp_date (numpy.datetime64): The date from the temperature dataset at the specified index.
        - matching_atm_date (numpy.datetime64): The matching date from the atmospheric dataset.
        - matching_atm_index (xarray.DataArray): The index of the matching date in the atmospheric dataset.
    Raises:
    ValueError: If no matching date is found in the atmospheric dataset for the specified temperature date.
    """
    temp_time = data_file_temp.time
    atm_time = data_file_atm.time
    temp_date = temp_time.isel(time=input_day).values
    matching_atm_index = atm_time.where(atm_time == temp_date, drop=True)
    
    if len(matching_atm_index) == 0:
        raise ValueError(f"No matching date found in atmospheric dataset for temperature date: {temp_date}")
    
    return temp_date, matching_atm_index.time.values[0], matching_atm_index

# Model output generation
def generate_data(model, current_value):
    """
    Generate output data using the model.

    Parameters:
    model (torch.nn.Module): The trained model.
    current_value (torch tensor): The input data for the model.

    Returns:
    torch tensor: The output data generated by the model.
    """
    with torch.no_grad():
        output_value = model(current_value)
    return output_value

# load ocean data
def load_ocean_data(data_file, input_day, test=False):
    """
    Load ocean data for a specific day and return all variables.

    Parameters:
    data_file (xarray.Dataset): The dataset containing ocean data.
    input_day (int): The index of the day to load data for.
    test (bool, optional): If True, convert all variables to torch tensors.

    Returns:
    tuple: A tuple containing:
        - temp (numpy array or torch tensor): Temperature data.
        - salt (numpy array or torch tensor): Salinity data.
        - u (numpy array or torch tensor): U-component of ocean current.
        - v (numpy array or torch tensor): V-component of ocean current.
        - height (numpy array or torch tensor): Sea surface height data.
    """
    temp = data_file['thetao'][input_day:input_day+1].values
    salt = data_file['so'][input_day:input_day+1].values
    u = data_file['uo'][input_day:input_day+1].values
    v = data_file['vo'][input_day:input_day+1].values
    height = data_file['zos'][input_day:input_day+1].values
    height = np.expand_dims(height, axis=1)
    
    if test:
        temp = torch.tensor(temp)
        salt = torch.tensor(salt)
        u = torch.tensor(u)
        v = torch.tensor(v)
        height = torch.tensor(height)
        temp[:, :, -20:, :] = float('nan')
        salt[:, :, -20:, :] = float('nan')
        u[:, :, -20:, :] = float('nan')
        v[:, :, -20:, :] = float('nan')
        height[:, :, -20:, :] = float('nan')

        
    return temp, salt, u, v, height

# Load atmospheric data
def load_atmospheric_data(data_file_atm, matching_atm_date):
    """
    Load atmospheric data for a specific date and return all variables.

    Parameters:
    data_file_atm (xarray.Dataset): The dataset containing atmospheric data.
    matching_atm_date (numpy.datetime64): The date to load data for.

    Returns:
    tuple: A tuple containing:
        - ssr (numpy array): Surface solar radiation data.
        - mslp (numpy array): Mean sea level pressure data.
        - tp (numpy array): Total precipitation data.
        - u10 (numpy array): U-component of 10m wind.
        - v10 (numpy array): V-component of 10m wind.
    """
    ssr = data_file_atm['ssr'].sel(time=matching_atm_date).values
    mslp = data_file_atm['msl'].sel(time=matching_atm_date).values
    tp = data_file_atm['tp'].sel(time=matching_atm_date).values
    u10 = data_file_atm['u10'].sel(time=matching_atm_date).values
    v10 = data_file_atm['v10'].sel(time=matching_atm_date).values
    tcc = data_file_atm['tcc'].sel(time=matching_atm_date).values
    
    return ssr, mslp, tp, u10, v10, tcc

# Plot all variables
def plot_variable(true_data, pred_data, error_data, lon, lat, var_name, cmap, time, variable=None):
    """
    Plots the true data, predicted data, and relative error on a map using Cartopy.
    Parameters:
    true_data (numpy.ndarray): The true data array with dimensions [time, level, lat, lon].
    pred_data (numpy.ndarray): The predicted data array with dimensions [time, level, lat, lon].
    error_data (numpy.ndarray): The relative error data array with dimensions [time, level, lat, lon].
    lon (numpy.ndarray): The longitude values.
    lat (numpy.ndarray): The latitude values.
    var_name (str): The name of the variable being plotted.
    cmap (matplotlib.colors.Colormap): The colormap to use for the true and predicted data.
    time (str): The time string to use in the filename of the saved plot.
    Returns:
    None
    """
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 2.5))
    # print(np.nanmin(true_data[0,0]))
    # print(np.nanmax(true_data[0,0]))
    # exit()

    if variable == 'so':
        vmin = 30
        data = true_data[0, 0]
        # Mask out NaN values and find max
        valid_data = data[~torch.isnan(data)]
        vmax = valid_data.max().item()
        levels =  np.linspace(vmin, vmax, 9)
        true = axes[0].contourf(lon, lat, true_data[0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=levels, extend='min')
    else:
        true = axes[0].contourf(lon, lat, true_data[0, 0], cmap=cmap, transform=ccrs.PlateCarree())    
    
    # True
    axes[0].coastlines()
    axes[0].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    axes[0].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
    axes[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
    fig.colorbar(true, ax=axes[0], format=FormatStrFormatter('%.1f'))
    axes[0].set_title('GLORYS')
    
    if variable == 'so':
        pred = axes[1].contourf(lon, lat, pred_data[0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=levels, extend='min')
    else:
        pred = axes[1].contourf(lon, lat, pred_data[0, 0], cmap=cmap, transform=ccrs.PlateCarree())
    axes[1].coastlines()
    axes[1].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    fig.colorbar(true, ax=axes[1], format=FormatStrFormatter('%.1f'))
    axes[1].set_title('AFNO')
    
    # Relative error
    cmap_rel = LinearSegmentedColormap.from_list('white_red', [(1, 1, 1), (1, 0, 0)])
    error_85th_percentile = np.nanpercentile(error_data[0, 0], 85)
    error_vmin = 0  
    error_vmax = error_85th_percentile
    error_levels = np.round(np.linspace(error_vmin, error_vmax, 20), 10)
    rel_error = axes[2].contourf(lon, lat, error_data[0, 0], cmap=cmap_rel, transform=ccrs.PlateCarree(), levels=error_levels)
    axes[2].coastlines()
    axes[2].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[2].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    cbar = fig.colorbar(rel_error, ax=axes[2])
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    axes[2].set_title('Relative Error')
    
    # Create directory if it does not exist
    output_dir = f"plots/new_{time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save figure
    filename_plot = os.path.join(output_dir, f"{var_name.lower()}.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()

def plot_velocity(true_u, true_v, pred_u, pred_v, lon, lat, time):
    """
    Plot the true and predicted velocity magnitudes and their error.

    Parameters:
    true_u (numpy array): True U-component of velocity.
    true_v (numpy array): True V-component of velocity.
    pred_u (numpy array): Predicted U-component of velocity.
    pred_v (numpy array): Predicted V-component of velocity.
    lon (numpy array): Longitudes.
    lat (numpy array): Latitudes.
    time (str): Time for the plot title and filename.

    Returns:
    None
    """
    # Calculate velocity magnitude and plot streamlines
    u_true, v_true = true_u[0, 0], true_v[0, 0]
    u_pred, v_pred = pred_u[0, 0], pred_v[0, 0]

    # Calculate magnitude
    true_velocity_magnitude = np.sqrt(u_true**2 + v_true**2)
    pred_velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2)
    error_velocity_magnitude = np.abs(true_velocity_magnitude - pred_velocity_magnitude)/np.abs(true_velocity_magnitude)

    # Plot velocity magnitude and streamlines
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 2.5))

    # True velocity magnitude and streamlines
    true_mag = axes[0].contourf(lon, lat, true_velocity_magnitude, cmap='cmo.speed', transform=ccrs.PlateCarree())
    skip = slice(None, None, 15)  # Show every 5th quiver
    axes[0].quiver(lon[skip], lat[skip], u_true[skip, skip], v_true[skip, skip], transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
    axes[0].coastlines()
    axes[0].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    axes[0].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
    axes[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
    # fig.colorbar(true_mag, ax=axes[0])
    fig.colorbar(true_mag, ax=axes[0], format=FormatStrFormatter('%.1f'))
    axes[0].set_title('GLORYS')

    # Predicted velocity magnitude and streamlines
    pred_mag = axes[1].contourf(lon, lat, pred_velocity_magnitude, cmap='cmo.speed', transform=ccrs.PlateCarree())
    skip = slice(None, None, 15)   # Show every 5th quiver
    axes[1].quiver(lon[skip], lat[skip], u_pred[skip, skip], v_pred[skip, skip], transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
    axes[1].coastlines()
    axes[1].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # fig.colorbar(true_mag, ax=axes[1])
    fig.colorbar(true_mag, ax=axes[1], format=FormatStrFormatter('%.1f'))
    axes[1].set_title('AFNO')

    # Error in velocity magnitude
    cmap_rel = LinearSegmentedColormap.from_list('white_red', [(1, 1, 1), (1, 0, 0)])
    error_85th_percentile = np.nanpercentile(error_velocity_magnitude[0, 0], 85)
    error_vmin = 0  
    error_vmax = error_85th_percentile
    error_levels = np.round(np.linspace(error_vmin, error_vmax, 20), 10)
    error_mag = axes[2].contourf(lon, lat, error_velocity_magnitude, cmap=cmap_rel, transform=ccrs.PlateCarree(), levels=error_levels)
    axes[2].coastlines()
    axes[2].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    axes[2].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    cbar = fig.colorbar(error_mag, ax=axes[2])
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    axes[2].set_title('Relative Error')

    # Save figure
    # Create directory if it does not exist
    output_dir = f"plots/new_{time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename_plot = os.path.join(output_dir, f"velocity.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()

def plot_variable_auto(true_data, pred_data, lon, lat, var_name, cmap, time, num_days, time_list):
   
    # Calculate global limits
    true_vmin = np.nanmin([np.nanmin(d) for d in true_data])
    true_vmax = np.nanmax([np.nanmax(d) for d in true_data])
    
    fig, axes = plt.subplots(2, num_days+1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(num_days+4, 2.5), sharex=True, sharey=True)
    
    # Reduce spacing between subplots
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Define the Bay of Bengal region boundaries
    bay_of_bengal_extent = [77, 99, 4, 23]
    true = axes[0,0].contourf(lon, lat, true_data[0][0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
    axes[0,0].coastlines()
    axes[0,0].set_extent(bay_of_bengal_extent, crs=ccrs.PlateCarree()) 
    axes[0,0].set_title(f'Input({time_list[0]})')

    axes[1, 0].set_visible(False)

    for day in range(num_days):
        # True
        true = axes[0, day+1].contourf(lon, lat, true_data[day+1][0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
        axes[0, day+1].coastlines()
        axes[0, day+1].set_title(time_list[day+1])
        
        # Predicted
        pred = axes[1, day+1].contourf(lon, lat, pred_data[day][0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
        axes[1, day+1].coastlines()
        
    axes[0,0].set_ylabel('CMEMS')
    # axes[1,1].set_ylabel('AFNO')
    fig.text(0.18, 0.28, 'AFNO', va='center', rotation='vertical')


    for a in axes[1, :]:
        a.set_xticks(np.arange(77.0,99.0,8), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        a.xaxis.set_major_formatter(lon_formatter)

    for a in axes[:, 0]:
        a.set_yticks(np.arange(4.0,23.0052281103605,5), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        a.yaxis.set_major_formatter(lat_formatter)

    

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.905, 0.11, 0.01, 0.78])  # Right side, adjust position to increase length
    cbar = fig.colorbar(true, cax=cbar_ax1, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Create directory if it does not exist
    output_dir = f"plots/autoregressive/new_{time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save figure
    filename_plot = os.path.join(output_dir, f"{var_name.lower()}_auto_9days.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()




def plot_variable_auto_alternate(true_data, pred_data, lon, lat, var_name, cmap, time, num_days, time_list):
   
    if var_name == 'Temperature':
        true_vmin = 26.4
        true_vmax = 32.8
    elif var_name == 'Salinity':
        true_vmin = 30.0
        true_vmax = 35.1
    else:
        true_vmin = 0.2
        true_vmax = 1.0

    # Initialize list to store all relative error arrays
    all_rel_errors = []

    
    
    selected_indices = [0] + list(range(1, num_days + 1, 2))  # 1,3,5,7,9
    num_plots = len(selected_indices) - 1  # exclude the input frame

    # Loop over selected indices (skip the first input frame)
    for idx in selected_indices[1:]:
        true_val = true_data[idx][0, 0]
        pred_val = pred_data[idx - 1][0, 0]
        
        # Compute relative error
        rel_error = np.abs(pred_val - true_val) / (np.abs(true_val) + 1e-6)  # avoid divide by zero
        all_rel_errors.append(rel_error)

    # Stack all error arrays to compute global statistics
    all_rel_errors_stacked = np.stack(all_rel_errors)

    # Compute global min and percentile-based max (ignoring NaNs)
    global_error_min = np.nanmin(all_rel_errors_stacked)
    global_error_85th = np.nanpercentile(all_rel_errors_stacked, 85)

    error_vmin = global_error_min
    error_vmax = global_error_85th

    print("error_min", error_vmin)
    print("error_max", error_vmax)
    # exit()

    fig, axes = plt.subplots(3, num_plots + 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(num_plots + 3, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    bay_of_bengal_extent = [77, 99, 4, 23]
    input_idx = selected_indices[0]
    true = axes[0, 0].contourf(lon, lat, true_data[input_idx][0, 0], cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9),  extend='both')
    axes[0, 0].coastlines()
    axes[0, 0].set_extent(bay_of_bengal_extent, crs=ccrs.PlateCarree())
    axes[0, 0].set_title(f'Input({time_list[input_idx]})')

    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)

    rel_error_vmax = 1.0  # max scale for relative error, clipped to 100%
    cmap_rel = LinearSegmentedColormap.from_list('white_red', [(1, 1, 1), (1, 0, 0)])

    for i, idx in enumerate(selected_indices[1:]):
        # True
        true_val = true_data[idx][0, 0]
        pred_val = pred_data[idx - 1][0, 0]
        rel_error = np.abs(pred_val - true_val) / (np.abs(true_val))
        # rel_error = np.clip(rel_error, 0, rel_error_vmax)

        # True data plot
        # axes[0, i + 1].contourf(lon, lat, true_val, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
        axes[0, i + 1].contourf(lon, lat, true_val, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9),  extend='both')
        axes[0, i + 1].coastlines()
        axes[0, i + 1].set_title(time_list[idx])

        # Prediction plot
        axes[1, i + 1].contourf(lon, lat, pred_val, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9),  extend='both')
        axes[1, i + 1].coastlines()

        # Relative error plot
        # error_85th_percentile = np.nanpercentile(rel_error, 85)
        # error_vmin = 0  
        # error_vmax = error_85th_percentile
        error_levels = np.round(np.linspace(error_vmin, error_vmax, 5), 5)
        err = axes[2, i + 1].contourf(lon, lat, rel_error, cmap=cmap_rel, transform=ccrs.PlateCarree(), levels=error_levels,  extend='both')
        axes[2, i + 1].coastlines()

    axes[0, 0].set_ylabel('GLORYS')
    fig.text(0.23, 0.51, 'AFNO', va='center', rotation='vertical')
    fig.text(0.23, 0.24, 'Rel. Error', va='center', rotation='vertical')
    
    if var_name == 'Temperature':
        fig.text(0.09, 0.24, '(a) SST', va='center', rotation='horizontal')
    elif var_name == 'Salinity':
        fig.text(0.09, 0.24, '(b) SSS', va='center', rotation='horizontal')
    else:
        fig.text(0.09, 0.24, '(c) SSH', va='center', rotation='horizontal')

    for a in axes[2, :]:
        a.set_xticks(np.arange(77.0, 99.0, 8), crs=ccrs.PlateCarree())
        a.xaxis.set_major_formatter(cticker.LongitudeFormatter())

    for a in axes[:, 0]:
        a.set_yticks(np.arange(4.0, 23.0052281103605, 5), crs=ccrs.PlateCarree())
        a.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    

    # Colorbar for main var
    cbar_ax1 = fig.add_axes([0.91, 0.38, 0.01, 0.48])
    cbar = fig.colorbar(true, cax=cbar_ax1, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Colorbar for relative error
    cbar_ax2 = fig.add_axes([0.91, 0.11, 0.01, 0.23])
    cbar2 = fig.colorbar(err, cax=cbar_ax2, orientation='vertical')
    cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    # Save
    output_dir = f"plots/autoregressive/new_{time}"
    os.makedirs(output_dir, exist_ok=True)
    filename_plot = os.path.join(output_dir, f"{var_name.lower()}_auto_alt_days_with_error.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()




def plot_velocity_auto(true_velocity_data, pred_velocity_data, lon, lat, time, num_days, time_list):
   
    # Calculate global limits
    true_vmin = np.nanmin([np.nanmin(np.sqrt(u**2 + v**2)) for u, v in true_velocity_data])
    true_vmax = np.nanmax([np.nanmax(np.sqrt(u**2 + v**2)) for u, v in true_velocity_data])

    fig, axes = plt.subplots(2, num_days+1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(num_days+4, 2.5), sharex=True, sharey=True)

    # Reduce spacing between subplots
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Define the Bay of Bengal region boundaries
    bay_of_bengal_extent = [77, 99, 4, 23]

    # Plot initial true velocity magnitude and streamlines
    u_true, v_true = true_velocity_data[0]
    true_velocity_magnitude = np.sqrt(u_true[0, 0]**2 + v_true[0, 0]**2)
    true_mag = axes[0, 0].contourf(lon, lat, true_velocity_magnitude, cmap='cmo.speed', transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
    skip = slice(None, None, 15)
    axes[0, 0].quiver(lon[skip], lat[skip], u_true[0, 0][skip, skip], v_true[0, 0][skip, skip], transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
    axes[0, 0].coastlines()
    axes[0, 0].set_extent(bay_of_bengal_extent, crs=ccrs.PlateCarree())
    axes[0, 0].set_title(f'Input({time_list[0]})')

    axes[1, 0].set_visible(False)

    for day in range(num_days):
        # True velocity magnitude and streamlines
        u_true, v_true = true_velocity_data[day+1]
        true_velocity_magnitude = np.sqrt(u_true[0, 0]**2 + v_true[0, 0]**2)
        true_mag = axes[0, day+1].contourf(lon, lat, true_velocity_magnitude, cmap='cmo.speed', transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
        axes[0, day+1].quiver(lon[skip], lat[skip], u_true[0, 0][skip, skip], v_true[0, 0][skip, skip], transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
        axes[0, day+1].coastlines()
        axes[0, day+1].set_title(time_list[day+1])
        
        # Predicted velocity magnitude and streamlines
        u_pred, v_pred = pred_velocity_data[day]
        pred_velocity_magnitude = np.sqrt(u_pred[0, 0]**2 + v_pred[0, 0]**2)
        pred_mag = axes[1, day+1].contourf(lon, lat, pred_velocity_magnitude, cmap='cmo.speed', transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 10))
        axes[1, day+1].quiver(lon[skip], lat[skip], u_pred[0, 0][skip, skip], v_pred[0, 0][skip, skip], transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
        axes[1, day+1].coastlines()

    axes[0, 0].set_ylabel('CMEMS')
    fig.text(0.18, 0.28, 'AFNO', va='center', rotation='vertical')

    for a in axes[1, :]:
        a.set_xticks(np.arange(77.0, 99.0, 8), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        a.xaxis.set_major_formatter(lon_formatter)

    for a in axes[:, 0]:
        a.set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        a.yaxis.set_major_formatter(lat_formatter)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.905, 0.11, 0.01, 0.78])  # Right side, adjust position to increase length
    cbar = fig.colorbar(true_mag, cax=cbar_ax1, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Create directory if it does not exist
    output_dir = f"plots/autoregressive/new_{time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save figure
    filename_plot = os.path.join(output_dir, f"velocity_auto_9days.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()


def plot_velocity_auto_alternative(true_velocity_data, pred_velocity_data, lon, lat, time, num_days, time_list):
    skip = slice(None, None, 15)
    epsilon = 1e-6  # small number to avoid division by zero

    indices = range(0, num_days, 2)
    ncols = len(indices) + 1  # +1 for initial input
    fig, axes = plt.subplots(3, ncols, subplot_kw={'projection': ccrs.PlateCarree()},
                             figsize=(ncols + 2, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Global min/max for consistent color scaling
    all_true_mags = [np.sqrt(u**2 + v**2) for u, v in true_velocity_data]
    # true_vmin = np.nanmin([np.nanmin(m) for m in all_true_mags])
    # true_vmax = np.nanmax([np.nanmax(m) for m in all_true_mags])
    true_vmin = 0.0
    true_vmax = 1.6

    extent = [77, 99, 4, 23]

    # Input plot (day 0)
    u0, v0 = true_velocity_data[0]
    mag0 = np.sqrt(u0[0, 0]**2 + v0[0, 0]**2)
    true = axes[0, 0].contourf(lon, lat, mag0, cmap='cmo.speed', transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9), extend='both')
    axes[0, 0].quiver(lon[skip], lat[skip], u0[0, 0][skip, skip], v0[0, 0][skip, skip],
                      transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
    axes[0, 0].coastlines()
    axes[0, 0].set_extent(extent)
    axes[0, 0].set_title(f'Input ({time_list[0]})')
    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)

    rel_error_vmax = 1.0  # max scale for relative error, clipped to 100%
    cmap_rel = LinearSegmentedColormap.from_list('white_red', [(1, 1, 1), (1, 0, 0)])

    all_rel_errors = []

    # Loop over selected indices (skip the first input frame)
    for idx in indices:
        u_true, v_true = true_velocity_data[idx + 1]
        u_pred, v_pred = pred_velocity_data[idx]
        mag_true = np.sqrt(u_true[0, 0]**2 + v_true[0, 0]**2)
        mag_pred = np.sqrt(u_pred[0, 0]**2 + v_pred[0, 0]**2)
        rel_error = np.abs(mag_pred - mag_true) / (mag_true)
        all_rel_errors.append(rel_error)



    # Stack all error arrays to compute global statistics
    all_rel_errors_stacked = np.stack(all_rel_errors)


    # Compute global min and percentile-based max (ignoring NaNs)
    global_error_min = np.nanmin(all_rel_errors_stacked)
    global_error_85th = np.nanpercentile(all_rel_errors_stacked, 85)


    error_vmin = global_error_min
    error_vmax = global_error_85th

    for idx, day in enumerate(indices):
        u_true, v_true = true_velocity_data[day + 1]
        u_pred, v_pred = pred_velocity_data[day]
        mag_true = np.sqrt(u_true[0, 0]**2 + v_true[0, 0]**2)
        mag_pred = np.sqrt(u_pred[0, 0]**2 + v_pred[0, 0]**2)
        rel_error = np.abs(mag_pred - mag_true) / (mag_true + epsilon)

        # True
        axes[0, idx + 1].contourf(lon, lat, mag_true, cmap='cmo.speed',
                                  transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9), extend='both')
        axes[0, idx + 1].quiver(lon[skip], lat[skip], u_true[0, 0][skip, skip], v_true[0, 0][skip, skip],
                                transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
        axes[0, idx + 1].coastlines()
        axes[0, idx + 1].set_title(time_list[day + 1])

        # Prediction
        axes[1, idx + 1].contourf(lon, lat, mag_pred, cmap='cmo.speed',
                                  transform=ccrs.PlateCarree(), levels=np.linspace(true_vmin, true_vmax, 9), extend='both')
        axes[1, idx + 1].quiver(lon[skip], lat[skip], u_pred[0, 0][skip, skip], v_pred[0, 0][skip, skip],
                                transform=ccrs.PlateCarree(), scale=10, width=0.007, color='black')
        axes[1, idx + 1].coastlines()

        # Relative Error
        error_levels = np.round(np.linspace(error_vmin, error_vmax, 5), 5)
        err_plot = axes[2, idx + 1].contourf(lon, lat, rel_error, cmap=cmap_rel,
                                             transform=ccrs.PlateCarree(), levels=error_levels, extend='both')
        axes[2, idx + 1].coastlines()

    # Labels
    axes[0, 0].set_ylabel('GLORYS')
    fig.text(0.23, 0.51, 'AFNO', va='center', rotation='vertical')
    fig.text(0.23, 0.24, 'Rel. Error', va='center', rotation='vertical')
    fig.text(0.09, 0.24, '(d) SSC', va='center', rotation='horizontal')

    # Ticks format
    for a in axes[2, :]:
        a.set_xticks(np.arange(77.0, 99.0, 8), crs=ccrs.PlateCarree())
        a.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    for a in axes[:, 0]:
        a.set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
        a.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # Colorbar for main var
    cbar_ax1 = fig.add_axes([0.91, 0.38, 0.01, 0.48])
    cbar = fig.colorbar(true, cax=cbar_ax1, orientation='vertical')
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Colorbar for relative error
    cbar_ax = fig.add_axes([0.91, 0.11, 0.01, 0.23])
    cbar2 = fig.colorbar(err_plot, cax=cbar_ax, orientation='vertical')
    cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    output_dir = f"plots/autoregressive/new_{time}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "velocity_auto_alt_days_with_error.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
