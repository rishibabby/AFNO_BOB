##########  Script to generate ocean front plot  ####################

import numpy as np
import xarray as xr
import cmocean

import matplotlib.pyplot as plt

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# For model
from model.afno.afnonet import afnonet

# Utils
from utils.utils import *
from pathlib import Path

from scipy import ndimage

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def plot_fronts_and_temp(fronts, temp, lat, lon, time):
    """
    Plot the ocean fronts and temperature using cmocean.
    
    Parameters:
    -----------
    fronts : np.ndarray
        Array containing the detected fronts
    temp : np.ndarray
        Array containing the temperature data
    lat : np.ndarray
        Array containing the latitude values
    lon : np.ndarray
        Array containing the longitude values
    time : str
        Time string for the plot title
    """
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7, 2.5))

    vmin = 21
    vmax = 31.5
    levels =  np.linspace(vmin, vmax, 9)

    # Plot temperature
    temp_plot = ax[0].contourf(lon, lat, temp[0, 0, :, :], cmap=cmocean.cm.thermal, transform=ccrs.PlateCarree(), levels=levels, extend='max')
    ax[0].coastlines()
    ax[0].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    ax[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax[0].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
    ax[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
    # ax[0].set_title(f'OSTIA SST on {time}')
    ax[0].set_title(f'OSTIA SST')
    # ax[0].set_xlabel('Longitude')
    # ax[0].set_ylabel('Latitude')
    fig.colorbar(temp_plot, ax=ax[0], orientation='vertical', format=FormatStrFormatter('%.1f'))

    # Plot fronts
    fronts_plot = ax[1].contourf(lon, lat, fronts[0, 0, :, :], cmap=cmocean.cm.dense)
    ax[1].coastlines()
    ax[1].set_xticks(np.arange(77.0, 99.0, 5), crs=ccrs.PlateCarree())
    ax[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax[1].set_yticks(np.arange(4.0, 23.0, 5), crs=ccrs.PlateCarree())
    ax[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
    # ax[1].set_title(f'Ocean Fronts on {time}')
    ax[1].set_title(f'Ocean Fronts')
    # ax[1].set_xlabel('Longitude')
    # ax[1].set_ylabel('Latitude')
    fig.colorbar(fronts_plot, ax=ax[1], orientation='vertical')

    # Save figure
    # Create directory if it does not exist
    output_dir = f"plots/fronts_ostia"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    filename_plot = os.path.join(output_dir, f"new_frontd_{time}.pdf")
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_fronts(temp, threshold=0.025, sigma=1.0):
    """
    Calculate oceanic fronts from model output data.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing model output variables
    variable : str
        Variable to use for front detection (e.g., 'temperature', 'salinity')
    threshold : float
        Threshold value for front detection
    sigma : float
        Smoothing parameter for gradient calculation
    """

    # Apply a Gaussian filter to the data
    smoothed = ndimage.gaussian_filter(temp, sigma=sigma)

    # Calculate gradients
    dy, dx = np.gradient(smoothed[0,0])
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    # Create a front mask based on threshold
    fronts = np.where(gradient_magnitude > threshold, gradient_magnitude, 0)
    
    return fronts

def plot(config):
    # """
    # Plots the model output and compares it with the true data for a given configuration.
    # Args:
    #     config (Config): Configuration object containing various parameters and file paths.
    # The function performs the following steps:
    # 1. Loads the pre-trained model from the specified file.
    # 2. Loads ocean and atmospheric data for the specified input day.
    # 3. Loads mean and variance data for preprocessing.
    # 4. Preprocesses the loaded data.
    # 5. Concatenates ocean and atmospheric data to form the input data for the model.
    # 6. Generates model output data.
    # 7. Postprocesses the model output data.
    # 8. Loads the true output data for the next day.
    # 9. Plots the true data, predicted data, and error for temperature, salinity, and height.
    # Note:
    #     The function assumes that the necessary data files and mean/variance files are available at the specified paths.
    # """

    # # Load model
    # filename1 = (
    #                     f"BOB_model_input_oc_{config.data.variable}_atm_{config.data.atm_variable}_output_{config.data.out_variable}_patch_{config.data.patch_size}_"
    #                     f"emd_dim_{config.data.emd_dim}_afno_layers_{config.afno2d.n_blocks}_blocks_{config.afno2d.num_blocks}_hd_{config.afno2d.hidden_size}"
    #                     f"mlp_in_{config.mlp.in_features}_hd_{config.mlp.hidden_features}_lr_{config.opt.lr}"
    #                 )
    # filename = ("saved_models/" + filename1 + ".pth")
    # model = afnonet(config)
    # print("Loaded model from: ", filename)
    # model.load_state_dict(torch.load(filename))
    # model.eval()

    # # Input day
    input_day = config.plot.input_day

    # # Load data files
    data_dir = config.data.data_dir
    # # ocean
    file_path = Path(data_dir)/f'{config.data.file_prefix}.nc'
    data_file = xr.open_dataset(file_path)
    # # atm
    # file_path_atm = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'
    # data_file_atm = xr.open_dataset(file_path_atm)
    # # lat and log
    lat = data_file.latitude.values
    lon = data_file.longitude.values
    # # time
    time = data_file['time'][input_day+1:input_day+2].values
    time = str(time)
    time = time[2:12]
    # print("Predicting Ocean Date: ", time)

    # # Load ocean data
    # temp, salt, u, v, height = load_ocean_data(data_file, input_day, test=True)

    # # Mask
    mask = data_file['thetao'].values
    dataset_length, mask = create_mask(mask)
    mask = torch.tensor(mask)

    # # # # Load atm data
    # temp_date, matching_atm_date, matching_atm_index = align_datasets_by_date(data_file, data_file_atm, input_day)
    # ssr, msl, tp, u10, v10, tcc = load_atmospheric_data(data_file_atm, matching_atm_date)

    # # # # Load mean and std
    # # # # ocean
    # mean_temp = np.load("data/mean/mean_thetao_1993_2018_all_months.npy")
    # mean_salt = np.load("data/mean/mean_so_1993_2018_all_months.npy")

    # # # # atm 
    # mean_ssr = np.load("data/mean/ssr_mean_1993_2018_all_months.npy")
    # mean_tp = np.load("data/mean/tp_mean_1993_2018_all_months.npy")
    # mean_msl = np.load("data/mean/msl_mean_1993_2018_all_months.npy")
    # variance_ssr = np.load("data/mean/ssr_var_1993_2018_all_months.npy")
    # variance_tp = np.load("data/mean/tp_var_1993_2018_all_months.npy")
    # variance_msl = np.load("data/mean/msl_var_1993_2018_all_months.npy")

    # # # Preprocess data
    # # # ocean
    # temp = data_preprocess(data=temp, mean=mean_temp, variable="thetao")
    # salt = data_preprocess(data=salt, mean=mean_salt, variable="so")
    # u = data_preprocess(data=u)
    # v = data_preprocess(data=v)
    # height = data_preprocess(data=height)

    # # # atm
    # u10 = data_preprocess(data=u10, type='atm')
    # v10 = data_preprocess(data=v10, type='atm')
    # tcc = data_preprocess(data=tcc, type='atm')
    # ssr = data_preprocess(data=ssr, mean=mean_ssr, var=variance_ssr,variable="ssr", type='atm')
    # msl = data_preprocess(data=msl, mean=mean_msl, var=variance_msl, variable="msl", type='atm')
    # tp = data_preprocess(data=tp, mean=mean_tp, var=variance_tp, variable="tp", type='atm')

    # # # # Concatenate ocean and atm data
    # input_data = torch.cat((ssr, tp, u10, v10, msl, tcc, temp, salt, u, v, height), dim=1)

    # # # # Model output
    # output_data = generate_data(model, input_data)

    # # # Postprocess data
    # temp = data_postprocess(data=output_data[:,0:1], mask=mask, mean=mean_temp, variable="thetao")
    # salt = data_postprocess(data=output_data[:,1:2], mask=mask, mean=mean_salt, variable="so")
    # u = data_postprocess(data=output_data[:,2:3], mask=mask)
    # v = data_postprocess(data=output_data[:,3:4], mask=mask)
    # height = data_postprocess(data=output_data[:,4:5], mask=mask)

    # True output data
    # true_temp, true_salt, true_u, true_v, true_height = load_ocean_data(data_file, input_day+1, test=True) 

    sst_ostia = xr.open_dataset('data/ostia_2020.nc')['analysed_sst']  # (366, 380, 440)
    lat = sst_ostia.latitude.values
    lon = sst_ostia.longitude.values
    # print(lat)
    temp = sst_ostia[0] - 273.15
    temp = torch.tensor(temp.values)
    temp = temp.unsqueeze(0).unsqueeze(0)
    temp = F.interpolate(temp, size=(229, 265), mode='bilinear', align_corners=False)
    temp[:, :, -20:, :] = float('nan')
    # print(temp.shape)
    # Plotting fronts
    fronts = calculate_fronts(temp)
    # Resize tensor to match the desired shape
    # mask = torch.nn.functional.interpolate(mask, size=(380, 440), mode='bilinear', align_corners=False)
    mask = torch.where(mask == 0.0, float('nan'), mask)
    fronts = fronts * mask.numpy()
    fronts[:, :, -20:, :] = float('nan')
    

    # Load data files
    data_dir = config.data.data_dir
    # ocean
    file_path = Path(data_dir)/f'{config.data.file_prefix}.nc'
    data_file = xr.open_dataset(file_path)
    # atm
    file_path_atm = Path(data_dir) / f'{config.data.file_prefix_atm}.nc'
    data_file_atm = xr.open_dataset(file_path_atm)
    # lat and log
    lat = data_file.latitude.values
    lon = data_file.longitude.values
    # print(fronts.shape)
    # exit()

    
    plot_fronts_and_temp(fronts, temp, lat, lon, time)
    

if __name__ == "__main__":

    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./temp_afno.yaml", config_name='default', config_folder='cfg/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='cfg/')
        ]
    )
    config = pipe.read_conf()

    # Plot for one day head for all variables
    plot(config)