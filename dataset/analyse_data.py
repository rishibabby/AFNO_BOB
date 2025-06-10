import xarray as xr
from pathlib import Path 


file_path_ostia = 'data/ostia_2020.nc'
data = xr.open_dataset(file_path_ostia)
data = data['analysed_sst']
print(data.shape)
