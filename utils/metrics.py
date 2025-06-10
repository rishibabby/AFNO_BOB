import torch
import torch.nn.functional as F

from utils.utils import data_postprocess

def calculate_rmse(pred_data, true_data):
    squared_diff = (true_data - pred_data) ** 2
    mse = torch.nanmean(squared_diff)
    return torch.sqrt(mse).item()

def calculate_mae(pred_data, true_data):
    mask = ~torch.isnan(pred_data) & ~torch.isnan(true_data)
    pred_data = pred_data[mask]
    true_data = true_data[mask]
    return F.l1_loss(pred_data, true_data).item()

def calculate_r2(pred_data, true_data):
    ss_res = torch.nansum((true_data - pred_data) ** 2)
    ss_tot = torch.nansum((true_data - torch.nanmean(true_data)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score.item()

def nanvar(tensor, dim=None, keepdim=False):
    """Computes variance while ignoring NaN values."""
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    squared_diff = (tensor - mean) ** 2
    return torch.nanmean(squared_diff, dim=dim, keepdim=keepdim)

def calculate_correlation(pred_data, true_data):
    """
    Calculate the correlation between predicted data and true data.

    Args:
        pred_data (torch.Tensor): The predicted data tensor.
        true_data (torch.Tensor): The true data tensor.

    Returns:
        float: The correlation coefficient between the predicted and true data.

    The function computes the correlation coefficient by:
    1. Calculating the mean of the predicted and true data, ignoring NaNs.
    2. Computing the covariance between the predicted and true data.
    3. Calculating the standard deviation of the predicted and true data, ignoring NaNs.
    4. Dividing the covariance by the product of the standard deviations to get the correlation coefficient.
    """
    pred_mean = torch.nanmean(pred_data, dim=(-2,-1), keepdim=True)
    true_mean = torch.nanmean(true_data, dim=(-2,-1), keepdim=True)
    covariance = torch.nanmean((pred_data - pred_mean) * (true_data - true_mean), dim=(-2,-1))
    pred_std = nanvar(pred_data, dim=(-2, -1), keepdim=True)
    true_std = nanvar(true_data, dim=(-2, -1), keepdim=True)
    deno = torch.sqrt(pred_std * true_std + 1e-8)
    return (covariance / deno).item()

# Calculate metrics for each variable separately
def calculate_metrics_for_variable(variable_index, true_data, output_data, mask, mean, variable_name):
    pred_data = data_postprocess(data=output_data[:, variable_index:variable_index+1], mask=mask, mean=mean, variable=variable_name)
    rmse = calculate_rmse(pred_data, true_data)
    mape = calculate_mae(pred_data, true_data)
    r2 = calculate_r2(pred_data, true_data)
    corr = calculate_correlation(pred_data, true_data)
    return rmse, mape, r2, corr