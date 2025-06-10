##########  Script to Generate One-Day Head Plot for Sea Surface Variables ####################

# Importing Libraries
import torch
import numpy as np
import torch.nn.functional as F
import xarray as xr

# For configuration
import argparse
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# For model
from model.afno.afnonet import afnonet

# Utils
from utils.utils import *
from utils.metrics import *
from pathlib import Path

def rmse(config):
    

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

    # Input day (Starting of 2020)
    input_day = 9860 + 60 + 92 + 122 + 61
    rmse = config.rmse

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
    print("Starting Date for test set: ", time)

    #  Mask
    mask = data_file['thetao'].values
    dataset_length, mask = create_mask(mask)
    mask = torch.tensor(mask)

    # Load mean and std
    # ocean
    mean_temp = np.load("data/mean/mean_thetao_1993_2018_all_months.npy")
    mean_salt = np.load("data/mean/mean_so_1993_2018_all_months.npy")

    # atm 
    mean_ssr = np.load("data/mean/ssr_mean_1993_2018_all_months.npy")
    mean_tp = np.load("data/mean/tp_mean_1993_2018_all_months.npy")
    mean_msl = np.load("data/mean/msl_mean_1993_2018_all_months.npy")
    variance_ssr = np.load("data/mean/ssr_var_1993_2018_all_months.npy")
    variance_tp = np.load("data/mean/tp_var_1993_2018_all_months.npy")
    variance_msl = np.load("data/mean/msl_var_1993_2018_all_months.npy")

    # Initialize lists to store metrics for each variable
    rmse_temp_list, MAE_temp_list, r2_temp_list, corr_temp_list = [], [], [], []
    rmse_salt_list, MAE_salt_list, r2_salt_list, corr_salt_list = [], [], [], []
    rmse_u_list, MAE_u_list, r2_u_list, corr_u_list = [], [], [], []
    rmse_v_list, MAE_v_list, r2_v_list, corr_v_list = [], [], [], []
    rmse_height_list, MAE_height_list, r2_height_list, corr_height_list = [], [], [], []

    for day in range(31 -rmse):

        # Load ocean data
        temp, salt, u, v, height = load_ocean_data(data_file, input_day)

        # Load atm data
        temp_date, matching_atm_date, matching_atm_index = align_datasets_by_date(data_file, data_file_atm, input_day+1)
        ssr, msl, tp, u10, v10, tcc = load_atmospheric_data(data_file_atm, matching_atm_date)
        

        # Preprocess data
        # ocean
        temp = data_preprocess(data=temp, mean=mean_temp, variable="thetao")
        salt = data_preprocess(data=salt, mean=mean_salt, variable="so")
        u = data_preprocess(data=u)
        v = data_preprocess(data=v)
        height = data_preprocess(data=height)
        # atm
        u10 = data_preprocess(data=u10, type='atm')
        v10 = data_preprocess(data=v10, type='atm')
        tcc = data_preprocess(data=tcc, type='atm')
        ssr = data_preprocess(data=ssr, mean=mean_ssr, var=variance_ssr,variable="ssr", type='atm')
        msl = data_preprocess(data=msl, mean=mean_msl, var=variance_msl, variable="msl", type='atm')
        tp = data_preprocess(data=tp, mean=mean_tp, var=variance_tp, variable="tp", type='atm')

        # Concatenate ocean and atm data
        input_data = torch.cat((ssr, tp, u10, v10, msl, tcc, temp, salt, u, v, height), dim=1)

        for j in range(rmse):

            # Model output
            if config.per:
                output_data = input_data[:, -5:]
            else:
                output_data = generate_data(model, input_data)
    
            # # Day Test set next day
            temp_date, matching_atm_date, matching_atm_index = align_datasets_by_date(data_file, data_file_atm, input_day+2+j)
            ssr, msl, tp, u10, v10, tcc = load_atmospheric_data(data_file_atm, matching_atm_date)

            # atm
            u10 = data_preprocess(data=u10, type='atm')
            v10 = data_preprocess(data=v10, type='atm')
            tcc = data_preprocess(data=tcc, type='atm')
            ssr = data_preprocess(data=ssr, mean=mean_ssr, var=variance_ssr,variable="ssr", type='atm')
            msl = data_preprocess(data=msl, mean=mean_msl, var=variance_msl, variable="msl", type='atm')
            tp = data_preprocess(data=tp, mean=mean_tp, var=variance_tp, variable="tp", type='atm')

            input_data = torch.cat((ssr, tp, u10, v10, msl, tcc, output_data), dim=1)

        # Load true output data for the current day
        true_temp, true_salt, true_u, true_v, true_height = load_ocean_data(data_file, input_day + 1 + j, test=True)
        print("Predicting Date: ", temp_date)

        # Calculate metrics for each variable separately
        rmse_temp, MAE_temp, r2_temp, corr_temp = calculate_metrics_for_variable(0, true_temp, output_data, mask, mean_temp, "thetao")
        rmse_salt, MAE_salt, r2_salt, corr_salt = calculate_metrics_for_variable(1, true_salt, output_data, mask, mean_salt, "so")
        rmse_u, MAE_u, r2_u, corr_u = calculate_metrics_for_variable(2, true_u, output_data, mask, None, None)
        rmse_v, MAE_v, r2_v, corr_v = calculate_metrics_for_variable(3, true_v, output_data, mask, None, None)
        rmse_height, MAE_height, r2_height, corr_height = calculate_metrics_for_variable(4, true_height, output_data, mask, None, None)

        # Append metrics to lists
        rmse_temp_list.append(rmse_temp)
        MAE_temp_list.append(MAE_temp)
        r2_temp_list.append(r2_temp)
        corr_temp_list.append(corr_temp)

        rmse_salt_list.append(rmse_salt)
        MAE_salt_list.append(MAE_salt)
        r2_salt_list.append(r2_salt)
        corr_salt_list.append(corr_salt)

        rmse_u_list.append(rmse_u)
        MAE_u_list.append(MAE_u)
        r2_u_list.append(r2_u)
        corr_u_list.append(corr_u)

        rmse_v_list.append(rmse_v)
        MAE_v_list.append(MAE_v)
        r2_v_list.append(r2_v)
        corr_v_list.append(corr_v)

        rmse_height_list.append(rmse_height)
        MAE_height_list.append(MAE_height)
        r2_height_list.append(r2_height)
        corr_height_list.append(corr_height)

        input_day += 1
    
    # Convert lists to numpy arrays for easier manipulation
    rmse_temp_array = np.array(rmse_temp_list)
    MAE_temp_array = np.array(MAE_temp_list)
    r2_temp_array = np.array(r2_temp_list)
    corr_temp_array = np.array(corr_temp_list)

    rmse_salt_array = np.array(rmse_salt_list)
    MAE_salt_array = np.array(MAE_salt_list)
    r2_salt_array = np.array(r2_salt_list)
    corr_salt_array = np.array(corr_salt_list)

    rmse_u_array = np.array(rmse_u_list)
    MAE_u_array = np.array(MAE_u_list)
    r2_u_array = np.array(r2_u_list)
    corr_u_array = np.array(corr_u_list)

    rmse_v_array = np.array(rmse_v_list)
    MAE_v_array = np.array(MAE_v_list)
    r2_v_array = np.array(r2_v_list)
    corr_v_array = np.array(corr_v_list)

    rmse_height_array = np.array(rmse_height_list)
    MAE_height_array = np.array(MAE_height_list)
    r2_height_array = np.array(r2_height_list)
    corr_height_array = np.array(corr_height_list)

    # Calculate mean and standard deviation for each metric
    def calculate_mean_and_std(metric_array):
        return np.mean(metric_array, axis=0), np.std(metric_array, axis=0)

    rmse_temp_mean, rmse_temp_std = calculate_mean_and_std(rmse_temp_array)
    MAE_temp_mean, MAE_temp_std = calculate_mean_and_std(MAE_temp_array)
    r2_temp_mean, r2_temp_std = calculate_mean_and_std(r2_temp_array)
    corr_temp_mean, corr_temp_std = calculate_mean_and_std(corr_temp_array)

    rmse_salt_mean, rmse_salt_std = calculate_mean_and_std(rmse_salt_array)
    MAE_salt_mean, MAE_salt_std = calculate_mean_and_std(MAE_salt_array)
    r2_salt_mean, r2_salt_std = calculate_mean_and_std(r2_salt_array)
    corr_salt_mean, corr_salt_std = calculate_mean_and_std(corr_salt_array)

    rmse_u_mean, rmse_u_std = calculate_mean_and_std(rmse_u_array)
    MAE_u_mean, MAE_u_std = calculate_mean_and_std(MAE_u_array)
    r2_u_mean, r2_u_std = calculate_mean_and_std(r2_u_array)
    corr_u_mean, corr_u_std = calculate_mean_and_std(corr_u_array)

    rmse_v_mean, rmse_v_std = calculate_mean_and_std(rmse_v_array)
    MAE_v_mean, MAE_v_std = calculate_mean_and_std(MAE_v_array)
    r2_v_mean, r2_v_std = calculate_mean_and_std(r2_v_array)
    corr_v_mean, corr_v_std = calculate_mean_and_std(corr_v_array)

    rmse_height_mean, rmse_height_std = calculate_mean_and_std(rmse_height_array)
    MAE_height_mean, MAE_height_std = calculate_mean_and_std(MAE_height_array)
    r2_height_mean, r2_height_std = calculate_mean_and_std(r2_height_array)
    corr_height_mean, corr_height_std = calculate_mean_and_std(corr_height_array)

    # Print results
    print("Temperature Metrics - RMSE Mean: ", rmse_temp_mean, " RMSE Std: ", rmse_temp_std)
    print("Temperature Metrics - MAE Mean: ", MAE_temp_mean, " MAE Std: ", MAE_temp_std)
    print("Temperature Metrics - R2 Mean: ", r2_temp_mean, " R2 Std: ", r2_temp_std)
    print("Temperature Metrics - Correlation Mean: ", corr_temp_mean, " Correlation Std: ", corr_temp_std)

    print("Salinity Metrics - RMSE Mean: ", rmse_salt_mean, " RMSE Std: ", rmse_salt_std)
    print("Salinity Metrics - MAE Mean: ", MAE_salt_mean, " MAE Std: ", MAE_salt_std)
    print("Salinity Metrics - R2 Mean: ", r2_salt_mean, " R2 Std: ", r2_salt_std)
    print("Salinity Metrics - Correlation Mean: ", corr_salt_mean, " Correlation Std: ", corr_salt_std)

    print("U Velocity Metrics - RMSE Mean: ", rmse_u_mean, " RMSE Std: ", rmse_u_std)
    print("U Velocity Metrics - MAE Mean: ", MAE_u_mean, " MAE Std: ", MAE_u_std)
    print("U Velocity Metrics - R2 Mean: ", r2_u_mean, " R2 Std: ", r2_u_std)
    print("U Velocity Metrics - Correlation Mean: ", corr_u_mean, " Correlation Std: ", corr_u_std)

    print("V Velocity Metrics - RMSE Mean: ", rmse_v_mean, " RMSE Std: ", rmse_v_std)
    print("V Velocity Metrics - MAE Mean: ", MAE_v_mean, " MAE Std: ", MAE_v_std)
    print("V Velocity Metrics - R2 Mean: ", r2_v_mean, " R2 Std: ", r2_v_std)
    print("V Velocity Metrics - Correlation Mean: ", corr_v_mean, " Correlation Std: ", corr_v_std)

    print("Height Metrics - RMSE Mean: ", rmse_height_mean, " RMSE Std: ", rmse_height_std)
    print("Height Metrics - MAE Mean: ", MAE_height_mean, " MAE Std: ", MAE_height_std)
    print("Height Metrics - R2 Mean: ", r2_height_mean, " R2 Std: ", r2_height_std)
    print("Height Metrics - Correlation Mean: ", corr_height_mean, " Correlation Std: ", corr_height_std)

    filename_results = "results.txt"

    with open(filename_results, "a") as file:
        file.write(f"\nExperiment Results\n")
        if config.per:
            file.write(f"Persistance\n\n")
        else:
            file.write(f"Model Filename: {filename}\n\n")  # Add the model filename as a header
        file.write(f"RMSE DAY: {config.rmse}\n")
        file.write(f"Temperature Metrics - RMSE Mean: {rmse_temp_mean} RMSE Std: {rmse_temp_std}\n")
        file.write(f"Temperature Metrics - MAE Mean: {MAE_temp_mean} MAE Std: {MAE_temp_std}\n")
        file.write(f"Temperature Metrics - R2 Mean: {r2_temp_mean} R2 Std: {r2_temp_std}\n")
        file.write(f"Temperature Metrics - Correlation Mean: {corr_temp_mean} Correlation Std: {corr_temp_std}\n\n")

        file.write(f"Salinity Metrics - RMSE Mean: {rmse_salt_mean} RMSE Std: {rmse_salt_std}\n")
        file.write(f"Salinity Metrics - MAE Mean: {MAE_salt_mean} MAE Std: {MAE_salt_std}\n")
        file.write(f"Salinity Metrics - R2 Mean: {r2_salt_mean} R2 Std: {r2_salt_std}\n")
        file.write(f"Salinity Metrics - Correlation Mean: {corr_salt_mean} Correlation Std: {corr_salt_std}\n\n")

        file.write(f"U Velocity Metrics - RMSE Mean: {rmse_u_mean} RMSE Std: {rmse_u_std}\n")
        file.write(f"U Velocity Metrics - MAE Mean: {MAE_u_mean} MAE Std: {MAE_u_std}\n")
        file.write(f"U Velocity Metrics - R2 Mean: {r2_u_mean} R2 Std: {r2_u_std}\n")
        file.write(f"U Velocity Metrics - Correlation Mean: {corr_u_mean} Correlation Std: {corr_u_std}\n\n")

        file.write(f"V Velocity Metrics - RMSE Mean: {rmse_v_mean} RMSE Std: {rmse_v_std}\n")
        file.write(f"V Velocity Metrics - MAE Mean: {MAE_v_mean} MAE Std: {MAE_v_std}\n")
        file.write(f"V Velocity Metrics - R2 Mean: {r2_v_mean} R2 Std: {r2_v_std}\n")
        file.write(f"V Velocity Metrics - Correlation Mean: {corr_v_mean} Correlation Std: {corr_v_std}\n\n")

        file.write(f"Height Metrics - RMSE Mean: {rmse_height_mean} RMSE Std: {rmse_height_std}\n")
        file.write(f"Height Metrics - MAE Mean: {MAE_height_mean} MAE Std: {MAE_height_std}\n")
        file.write(f"Height Metrics - R2 Mean: {r2_height_mean} R2 Std: {r2_height_std}\n")
        file.write(f"Height Metrics - Correlation Mean: {corr_height_mean} Correlation Std: {corr_height_std}\n")
        file.write("\n" + "-" * 100 + "\n")

    print(f"Results saved to '{filename_results}'")


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

    # rmse
    rmse(config)