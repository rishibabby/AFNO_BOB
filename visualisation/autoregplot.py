##########  Script to Generate One-Day Head Plot for Sea Surface Variables ####################

# Importing Libraries
import torch
import numpy as np
import torch.nn.functional as F
import xarray as xr

# For configuration
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# For model
from model.afno.afnonet import afnonet

# Utils
from utils.utils import *
from pathlib import Path


def plot(config):
    """
    Plots the model output and compares it with the true data for a given configuration.
    Args:
        config (Config): Configuration object containing various parameters and file paths.
    The function performs the following steps:
    1. Loads the pre-trained model from the specified file.
    2. Loads ocean and atmospheric data for the specified input day.
    3. Loads mean and variance data for preprocessing.
    4. Preprocesses the loaded data.
    5. Concatenates ocean and atmospheric data to form the input data for the model.
    6. Generates model output data.
    7. Postprocesses the model output data.
    8. Loads the true output data for the next day.
    9. Plots the true data, predicted data, and error for temperature, salinity, and height.
    Note:
        The function assumes that the necessary data files and mean/variance files are available at the specified paths.
    """

    # Load model
    filename1 = (
                        f"BOB_model_input_oc_{config.data.variable}_atm_{config.data.atm_variable}_output_{config.data.out_variable}_patch_{config.data.patch_size}_"
                        f"emd_dim_{config.data.emd_dim}_afno_layers_{config.afno2d.n_blocks}_blocks_{config.afno2d.num_blocks}_hd_{config.afno2d.hidden_size}"
                        f"mlp_in_{config.mlp.in_features}_hd_{config.mlp.hidden_features}_lr_{config.opt.lr}"
                    )
    filename = ("saved_models/" + filename1 + ".pth")
    model = afnonet(config)
    print("Loaded model from: ", filename)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Input day
    input_day = config.plot.input_day

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
    # time
    time = data_file['time'][input_day+1:input_day+2].values
    time = str(time)
    time = time[2:12]
    print("Predicting Ocean Date: ", time)

    # Load ocean data
    true_temp, true_salt, true_u, true_v, true_height = load_ocean_data(data_file, input_day, test=True)

    # # Mask
    mask = data_file['thetao'].values
    dataset_length, mask = create_mask(mask)
    mask = torch.tensor(mask)

    # # # Load atm data
    temp_date, matching_atm_date, matching_atm_index = align_datasets_by_date(data_file, data_file_atm, input_day+1)
    ssr, msl, tp, u10, v10, tcc = load_atmospheric_data(data_file_atm, matching_atm_date)
    
    # # # Load mean and std
    # # # ocean
    mean_temp = np.load("data/mean/mean_thetao_1993_2018_all_months.npy")
    mean_salt = np.load("data/mean/mean_so_1993_2018_all_months.npy")

    # # # atm 
    mean_ssr = np.load("data/mean/ssr_mean_1993_2018_all_months.npy")
    mean_tp = np.load("data/mean/tp_mean_1993_2018_all_months.npy")
    mean_msl = np.load("data/mean/msl_mean_1993_2018_all_months.npy")
    variance_ssr = np.load("data/mean/ssr_var_1993_2018_all_months.npy")
    variance_tp = np.load("data/mean/tp_var_1993_2018_all_months.npy")
    variance_msl = np.load("data/mean/msl_var_1993_2018_all_months.npy")

    # # Preprocess data
    # # ocean
    temp = data_preprocess(data=true_temp, mean=mean_temp, variable="thetao")
    salt = data_preprocess(data=true_salt, mean=mean_salt, variable="so")
    u = data_preprocess(data=true_u)
    v = data_preprocess(data=true_v)
    height = data_preprocess(data=true_height)

    # # atm
    u10 = data_preprocess(data=u10, type='atm')
    v10 = data_preprocess(data=v10, type='atm')
    tcc = data_preprocess(data=tcc, type='atm')
    ssr = data_preprocess(data=ssr, mean=mean_ssr, var=variance_ssr,variable="ssr", type='atm')
    msl = data_preprocess(data=msl, mean=mean_msl, var=variance_msl, variable="msl", type='atm')
    tp = data_preprocess(data=tp, mean=mean_tp, var=variance_tp, variable="tp", type='atm')

    # # # Concatenate ocean and atm data
    input_data = torch.cat((ssr, tp, u10, v10, msl, tcc, temp, salt, u, v, height), dim=1)

    num_days = config.plot.num_days
    all_true_data = []
    all_pred_data = []
    all_error_data = []
    true_velocity_data = []
    pred_velocity_data = []
    time_list = []

    all_true_data.append([true_temp, true_salt, true_height])
    true_velocity_data.append([true_u, true_v])

    # Store time for plotting
    time_str = str(data_file['time'][input_day].values)
    time_str = time_str[2:10]
    time_list.append(time_str)

    for day in range(num_days):
        # Model output
        output_data = generate_data(model, input_data)

        # Postprocess data
        temp = data_postprocess(data=output_data[:,0:1], mask=mask, mean=mean_temp, variable="thetao")
        salt = data_postprocess(data=output_data[:,1:2], mask=mask, mean=mean_salt, variable="so")
        u = data_postprocess(data=output_data[:,2:3], mask=mask)
        v = data_postprocess(data=output_data[:,3:4], mask=mask)
        height = data_postprocess(data=output_data[:,4:5], mask=mask)

        # True output data
        true_temp, true_salt, true_u, true_v, true_height = load_ocean_data(data_file, input_day+2+day, test=True)
        
        # Store data for plotting
        all_true_data.append([true_temp, true_salt, true_height])
        all_pred_data.append([temp, salt, height])
        true_velocity_data.append([true_u, true_v])
        pred_velocity_data.append([u, v])

        # atm data for next day
        temp_date, matching_atm_date, matching_atm_index = align_datasets_by_date(data_file, data_file_atm, input_day+1+day)
        ssr, msl, tp, u10, v10, tcc = load_atmospheric_data(data_file_atm, matching_atm_date)

        u10 = data_preprocess(data=u10, type='atm')
        v10 = data_preprocess(data=v10, type='atm')
        tcc = data_preprocess(data=tcc, type='atm')
        ssr = data_preprocess(data=ssr, mean=mean_ssr, var=variance_ssr,variable="ssr", type='atm')
        msl = data_preprocess(data=msl, mean=mean_msl, var=variance_msl, variable="msl", type='atm')
        tp = data_preprocess(data=tp, mean=mean_tp, var=variance_tp, variable="tp", type='atm')

        # Update input data for next day prediction
        input_data = torch.cat((ssr, tp, u10, v10, msl, tcc, output_data), dim=1)

        # Store time for plotting
        time_str = str(data_file['time'][input_day+1+day].values)
        time_str = time_str[2:10]
        time_list.append(time_str)
    

    # Plotting
    variables = ['Temperature', 'Salinity', 'Height']
    cmaps = ['cmo.thermal', 'cmo.haline', 'cmo.topo']

    for var_idx, var in enumerate(variables):
        true_data = [day_data[var_idx] for day_data in all_true_data]
        pred_data = [day_data[var_idx] for day_data in all_pred_data]
        plot_variable_auto(true_data, pred_data, lon, lat, var, cmaps[var_idx], time, num_days, time_list)
        
    plot_velocity_auto(true_velocity_data, pred_velocity_data, lon, lat, time, num_days, time_list)

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