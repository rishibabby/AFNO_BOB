## ------------------------------Data Preprocesing while training AFNO------------------------------------------#

import xarray as xr
import numpy as np
import pandas as pd 

import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, Subset

from pathlib import Path 
import scipy.interpolate as interpolate

# Transform input ( subtract by mean and interpolation to (224,224))
class preprocessTransform(torch.nn.Module):
    def __init__(self, config):
        super(preprocessTransform, self).__init__()

    def forward(self, sample, mean=None, variable=None, type=None, variance=None):

        # sub by mean only for thetao
        if type == 'atm':
            if variable == 'ssr' or variable == 'tp' or variable == 'msl':
                sample = sample - mean
                sample = sample/variance
        else: 
            if variable == 'thetao' or variable == 'so':
                sample = sample - mean
            if variable == 'zos':
                sample = np.expand_dims(sample, axis =1)
            sample = sample 

        # numpy to torch
        sample = torch.tensor(sample, dtype=torch.float32)

        # interpolate data
        if type == 'atm':
            sample = F.interpolate(sample, size=(224,224), mode='bilinear', align_corners=False)
            return sample
        
        
        # Replace Nan values
        sample_mean = torch.nanmean(sample)
        sample = torch.where(torch.isnan(sample), sample_mean, sample)
        sample = F.interpolate(sample, size=(224,224), mode='bilinear', align_corners=False)
        
        return sample

# DataLoader
class NetCDFDataset(Dataset):
    def __init__(self, config, mean, transform=None, indices_ocean=None, indices_atm=None):
        self.data_dir = config.data.data_dir
        
        # Ocean data
        self.file_path_ocean = Path(self.data_dir) / f'{config.data.file_prefix}.nc'
        self.data_ocean = xr.open_dataset(self.file_path_ocean)
        self.variable = config.data.variable
        self.out_variables = config.data.out_variable


        # Atmospheric data
        self.file_path_atm = Path(self.data_dir) / f'{config.data.file_prefix_atm}.nc'
        self.data_atm = xr.open_dataset(self.file_path_atm)
        self.atm_variable = config.data.atm_variable
        self.ssr_std = np.sqrt(np.load('data/mean/ssr_var_1993_2018_all_months.npy'))
        self.tp_std = np.sqrt(np.load('data/mean/tp_var_1993_2018_all_months.npy'))
        self.msl_std = np.sqrt(np.load('data/mean/msl_var_1993_2018_all_months.npy'))

        # Indices
        self.indices_ocean = indices_ocean
        self.indices_atm = indices_atm

        # Transform and mean normalization
        self.transform = transform
        self.mean = mean
        self.rows = 20

    def __len__(self):
        return len(self.indices_atm)  # Use atmospheric indices length

    def __getitem__(self, idx):
        # Get ocean and atmospheric indices
        ocean_idx = self.indices_ocean[idx]
        atm_idx = self.indices_atm[idx]

        var_inputs = []
        var_outputs = []

        # Atmospheric data
        for atm_var in self.atm_variable:
            atm_input = self.data_atm[atm_var][atm_idx+1:atm_idx+2].values
            if self.transform:
                if atm_var == 'ssr':
                    atm_input = self.transform(atm_input, self.mean[atm_var], variable=atm_var, type='atm', variance=self.ssr_std)
                elif atm_var == 'tp':
                    atm_input = self.transform(atm_input, self.mean[atm_var], variable=atm_var, type='atm', variance=self.tp_std)
                elif atm_var == 'msl':
                    atm_input = self.transform(atm_input, self.mean[atm_var], variable=atm_var, type='atm', variance=self.msl_std)
                else:
                    atm_input = self.transform(atm_input, self.mean[atm_var], variable=atm_var, type='atm')
            var_inputs.append(atm_input)
        
        # Ocean data
        for variable in self.variable:
            var_input = self.data_ocean[variable][ocean_idx:ocean_idx+1].values
            # print(var_input.shape)
            var_output = self.data_ocean[variable][ocean_idx+1:ocean_idx+2].values

            if self.transform:
                var_input = self.transform(var_input, self.mean[variable], variable=variable)
                var_output = self.transform(var_output, self.mean[variable], variable=variable)
                var_output[:, :, -self.rows:, :] = 0.0
                var_input[:, :, -self.rows:, :] = 0.0
            var_outputs.append(var_output)
            var_inputs.append(var_input)

        
        var_input = torch.cat([var for var in var_inputs], dim=0)
        var_output = torch.cat([var for var in var_outputs], dim=0)

        return var_input.squeeze(1), var_output.squeeze(1)




def split_indices(config, start_year=1993, end_year=2020):
    # Generate a complete date range for your sequential data
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
    
    # Filter indices for each target year 
    train_years = [1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    val_year = 2019
    test_year = 2020
    
    # You can change for specific months
    train_mask = (dates.year.isin(train_years)) & (dates.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12]))
    val_mask = (dates.year == val_year) & (dates.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12]))
    test_mask = (dates.year == test_year) & (dates.month.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12]))

    # Convert the boolean masks to indices
    train_indices = train_mask.nonzero()[0].tolist()
    val_indices = val_mask.nonzero()[0].tolist()
    test_indices = test_mask.nonzero()[0].tolist()

    # Atmospheric data indices (sequential)
    train_indices_atm = list(range(len(train_indices)))
    val_indices_atm = list(range(len(val_indices)))
    test_indices_atm = list(range(len(test_indices)))


    return train_indices, val_indices, test_indices, train_indices_atm, val_indices_atm, test_indices_atm

    #return train_indices, val_indices, test_indices

def create_mask(data):
    # Get the first timestep to create the mask
    dataset_length, _, _, _ = data.shape
    first_timestep = data[0]  # Shape: (1, height, width)
    
    # Create mask: 1 for non-nan values, 0 for nan values
    mask = ~np.isnan(first_timestep)  # ~ operator inverts True/False
    mask = mask.astype(np.float32)    # Convert boolean to float
    
    # Ensure the mask has shape (1, 1, height, width)
    if mask.ndim == 3:
        mask = mask[np.newaxis, :, :, :]
        
    return dataset_length, mask

def load_and_prepare_data(config):
    
    # Load mean values
    # later ...............########
    print("\n### Mean LOADED ###\n")

    # Ocean mean
    mean_temp = np.load("data/mean/mean_thetao_1993_2018_all_months.npy")
    mean_salt= np.load("data/mean/mean_so_1993_2018_all_months.npy")
    mean_uo = None
    mean_vo = None
    mean_zos = None 

    # Atmosphere mean
    mean_ssr = np.load("data/mean/ssr_mean_1993_2018_all_months.npy")
    mean_tp  = np.load("data/mean/tp_mean_1993_2018_all_months.npy")
    mean_u10 = None
    mean_v10 = None
    mean_msl = np.load("data/mean/msl_mean_1993_2018_all_months.npy")
    mean_tcc = None
    

    mean = {'thetao': mean_temp, 'so': mean_salt, 'uo': mean_uo, 'vo': mean_vo, 'ssr': mean_ssr, 'tp': mean_tp, 'u10': mean_u10, 'v10': mean_v10, 'zos': mean_zos, 'msl': mean_msl, 'tcc': mean_tcc}

    # Load Msk
    data_dir = config.data.data_dir
    file_path = Path(data_dir)/f'{config.data.file_prefix}.nc'
    mask = xr.open_dataset(file_path)['thetao'].values
    dataset_length, mask = create_mask(mask)
    mask = torch.tensor(mask)
    mask = F.interpolate(mask, size=(224,224), mode='bilinear', align_corners=False)

    # transform
    print("\n### TRANSFORM ###\n")
    transform = preprocessTransform(config)

    print("\n### INDICES SPLIT ###\n")
    train_indices, val_indices, test_indices, train_indices_atm, val_indices_atm, test_indices_atm= split_indices(config)
    
    
    train_dataset = NetCDFDataset(config, mean, transform, train_indices, train_indices_atm)
    val_dataset = NetCDFDataset(config, mean, transform, val_indices, val_indices_atm)
    test_dataset = NetCDFDataset(config, mean, transform, test_indices, test_indices_atm)

    print("\n### DATA LOADER ###\n")
    train_loader = DataLoader(train_dataset, batch_size = config.data.batch_size, shuffle=config.data.shuffle)
    val_loader = DataLoader(val_dataset, batch_size = config.data.batch_size, shuffle=config.data.shuffle)
    test_loader = DataLoader(test_dataset, batch_size = config.data.batch_size, shuffle=config.data.shuffle)

    return train_loader, val_loader, test_loader, mask



def extract_all_points(test_dataset, name):
    """
    Iterate through the entire NetCDF dataset and extract all points.
    
    Args:
        test_dataset (NetCDFDataset): The dataset to iterate through
    
    Returns:
        numpy.ndarray: Array containing all extracted points
    """
    # Create a list to store all points
    all_points = []
    
    # Iterate through the entire dataset
    for i in range(len(test_dataset)):
        # Extract the point from the dataset
        point = test_dataset[i]
        
        # If point is a tuple (common in PyTorch datasets), 
        # we typically want the first element
        if isinstance(point, tuple):
            point = point[0]
        
        # Convert to numpy if it's a torch tensor
        if torch.is_tensor(point):
            point = point.numpy()
        
        # Append to the list
        all_points.append(point)
    
    # Convert list to numpy array
    np.save(f'{name}.npy', extracted_points)
    
    # return np.array(all_points)


if __name__ == "__main__":

    from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./salt_config.yaml", config_name='default', config_folder='cfg/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='cfg/')
        ]
    )
    config = pipe.read_conf()
    
    load_and_prepare_data(config)



