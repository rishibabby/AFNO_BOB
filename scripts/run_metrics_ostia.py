import xarray as xr
import numpy as np

from utils.metrics import *

# Load datasets
sst_ostia = xr.open_dataset('data/ostia_2020.nc')['analysed_sst']  # (366, 380, 440)
sst_glorys = xr.open_dataset('data/ocean.nc')['thetao']  # (10227, 1, 229, 265)

# Select 2020 data from GLORYS
sst_glorys = sst_glorys.sel(time=slice("2020-01-01", "2020-12-31")).squeeze()  # Remove singleton dimension
sst_ostia = sst_ostia - 273.15  # Convert to Celsius
# print(sst_glorys.shape)
# exit()
# Check coordinate names
ostia_lon = 'longitude' if 'longitude' in sst_ostia.coords else 'lon'
ostia_lat = 'latitude' if 'latitude' in sst_ostia.coords else 'lat'
glorys_lon = 'longitude' if 'longitude' in sst_glorys.coords else 'lon'
glorys_lat = 'latitude' if 'latitude' in sst_glorys.coords else 'lat'

# Interpolate GLORYS to OSTIA grid
sst_glorys_interp = sst_glorys.interp(
    {glorys_lon: sst_ostia[ostia_lon], glorys_lat: sst_ostia[ostia_lat]},
    method="linear"
)

# Ensure temporal alignment
common_times = np.intersect1d(sst_ostia.time.values, sst_glorys_interp.time.values)
sst_ostia_aligned = sst_ostia.sel(time=common_times)
sst_glorys_interp_aligned = sst_glorys_interp.sel(time=common_times)

print("OSTIA shape:", sst_ostia_aligned.shape)
print("GLORYS shape:", sst_glorys_interp_aligned.shape)


def evaluate_all_metrics(pred, true):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred.values if hasattr(pred, "values") else pred, dtype=torch.float32)
    if not isinstance(true, torch.Tensor):
        true = torch.tensor(true.values if hasattr(true, "values") else true, dtype=torch.float32)

    rmse_list, mae_list, r2_list, corr_list = [], [], [], []
    for i in range(pred.shape[0]):
        pred_slice = pred[i]
        true_slice = true[i]
        
        rmse_list.append(calculate_rmse(pred_slice, true_slice))
        mae_list.append(calculate_mae(pred_slice, true_slice))
        r2_list.append(calculate_r2(pred_slice, true_slice))
        corr_list.append(calculate_correlation(pred_slice, true_slice))
        print(i)
    
    metrics = {
        "Mean RMSE": np.nanmean(rmse_list), #sum(rmse_list) / len(rmse_list),
        "Mean MAE": np.nanmean(mae_list), #sum(mae_list) / len(mae_list),
        "Mean R²": np.nanmean(r2_list), #sum(r2_list) / len(r2_list),
        "Mean Correlation": np.nanmean(corr_list), #sum(corr_list) / len(corr_list),
    }
    return metrics

# Call the evaluation
results = evaluate_all_metrics(sst_glorys_interp_aligned, sst_ostia_aligned)
print(results)

# Save results to file
with open('results_ostia.txt', 'a') as f:
    f.write(f"Overall RMSE: {results['Mean RMSE']:.4f}\n")
    f.write(f"Cross Correlation: {results['Mean Correlation']:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {results['Mean MAE']:.4f}\n")
    f.write(f"R-squared: {results['Mean R²']:.4f}\n")
    f.write("\n" + "-" * 100 + "\n")


