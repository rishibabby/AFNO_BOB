
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
    input_day = 9860 #+ 60 #+ 92 #+ 122
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

    # Load actual data
    sst_ostia = xr.open_dataset('data/ostia_2020.nc')['analysed_sst'] # (366, 380, 440)

    for day in range(365 -rmse):

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
        
        # Post-process temp
        temp_output = data_postprocess(data=output_data[:,0:1], mask=mask, mean=mean_temp, variable="thetao")
        time = str(temp_date)
        time = time[0:10]
        print("Predicting Date: ", time)

           
        # Align dates
        sst_ostia_aligned = sst_ostia.sel(time=time)
        sst_ostia_aligned = sst_ostia_aligned - 273.15  # Convert to Celsius
        sst_ostia_aligned = torch.tensor(sst_ostia_aligned.values)
           
        # Interpolate temp_output to match the shape of sst_ostia_aligned
        temp_output_aligned = F.interpolate(temp_output, size=(380, 440), mode='bilinear', align_corners=False)[0,0]

        # Append to lists
        rmse_temp_list.append(calculate_rmse(temp_output_aligned, sst_ostia_aligned))
        MAE_temp_list.append(calculate_mae(temp_output_aligned, sst_ostia_aligned))
        r2_temp_list.append(calculate_r2(temp_output_aligned, sst_ostia_aligned))
        corr_temp_list.append(calculate_correlation(temp_output_aligned, sst_ostia_aligned))

        input_day += 1

    # print(rmse_temp_list)
    # exit()
    # Calculate metrics
    rmse_temp = np.mean(rmse_temp_list)
    MAE_temp = np.mean(MAE_temp_list)
    r2_temp = np.mean(r2_temp_list)
    corr_temp = np.mean(corr_temp_list)

    # Store results in results_ostia.txt
    with open('results_ostia.txt', 'a') as f:
        f.write(f'RMSE: {rmse_temp}\n')
        f.write(f'MAE: {MAE_temp}\n')
        f.write(f'R2: {r2_temp}\n')
        f.write(f'Correlation: {corr_temp}\n')
        f.write("\n" + "-" * 100 + "\n")

        
        
        

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