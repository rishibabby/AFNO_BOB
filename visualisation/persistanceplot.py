import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Sea Surface Temperature (SST)
sst_persistence_rmse = [0.149, 0.223, 0.280, 0.327, 0.366, 0.402, 0.433, 0.462, 0.489]
sst_persistence_cc = [0.973, 0.941, 0.910, 0.883, 0.857, 0.834, 0.812, 0.792, 0.773]
sst_afno_rmse = [0.127, 0.172, 0.207, 0.236, 0.261, 0.283, 0.302, 0.320, 0.338]
sst_afno_cc = [0.979, 0.962, 0.946, 0.930, 0.915, 0.900, 0.886, 0.872, 0.857]

# Sea Surface Salinity (SSS)
sss_persistence_rmse = [0.288, 0.426, 0.530, 0.611, 0.675, 0.727, 0.772, 0.812, 0.846]
sss_persistence_cc = [0.988, 0.976, 0.964, 0.952, 0.942, 0.933, 0.925, 0.918, 0.911]
sss_afno_rmse = [0.241, 0.325, 0.402, 0.470, 0.531, 0.589, 0.645, 0.701, 0.760]
sss_afno_cc = [0.991, 0.985, 0.978, 0.970, 0.964, 0.954, 0.945, 0.936, 0.926]

# Sea Surface Height (SSH)
ssh_persistence_rmse = [0.018, 0.024, 0.030, 0.035, 0.040, 0.044, 0.048, 0.051, 0.054]
ssh_persistence_cc = [0.982, 0.968, 0.951, 0.932, 0.913, 0.893, 0.874, 0.857, 0.840]
ssh_afno_rmse = [0.017, 0.022, 0.027, 0.031, 0.036, 0.041, 0.046, 0.050, 0.053]
ssh_afno_cc = [0.983, 0.974, 0.962, 0.947, 0.930, 0.911, 0.892, 0.873, 0.853]

# Zonal
zonal_persistence_rmse = [0.086, 0.118, 0.137, 0.150, 0.163, 0.176, 0.185, 0.193, 0.201]
zonal_persistence_cc = [0.927, 0.867, 0.821, 0.786, 0.749, 0.707, 0.673, 0.642, 0.611]
zonal_afno_rmse = [0.053, 0.077, 0.095, 0.111, 0.127, 0.143, 0.162, 0.174, 0.193]
zonal_afno_cc = [0.970, 0.937, 0.904, 0.869, 0.828, 0.782, 0.728, 0.666, 0.612]

# Meridional
meridional_persistence_rmse = [0.088, 0.120, 0.146, 0.156, 0.171, 0.185, 0.197, 0.206, 0.216]
meridional_persistence_cc = [0.920, 0.854, 0.798, 0.756, 0.698, 0.642, 0.594, 0.550, 0.508]
meridional_afno_rmse = [0.054, 0.078, 0.098, 0.114, 0.128, 0.141, 0.154, 0.166, 0.175]
meridional_afno_cc = [0.970, 0.933, 0.893, 0.853, 0.814, 0.775, 0.733, 0.691, 0.652]

# Create directory for plots if it doesn't exist
os.makedirs('plots/persistence', exist_ok=True)

# Function to create and save plots
def create_plot(days, persistence_data, afno_data, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(days, persistence_data, 'b-o', label='Persistence', linewidth=2)
    plt.plot(days, afno_data, 'r-o', label='AFNO', linewidth=2)
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(days)
    plt.savefig(f'plots/persistence/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# Create days array
days = range(1, 10)  # 1 to 9

# Plot RMSE and CC for SST
create_plot(days, sst_persistence_rmse, sst_afno_rmse, 'RMSE', '9-Day Comparison for SST RMSE', 'sst_rmse.pdf')
create_plot(days, sst_persistence_cc, sst_afno_cc, 'Correlation Coefficient', '9-Day Comparison for SST CC', 'sst_cc.pdf')

# Plot RMSE and CC for SSS
create_plot(days, sss_persistence_rmse, sss_afno_rmse, 'RMSE', '9-Day Comparison for SSS RMSE', 'sss_rmse.pdf')
create_plot(days, sss_persistence_cc, sss_afno_cc, 'Correlation Coefficient', '9-Day Comparison for SSS CC', 'sss_cc.pdf')

# Plot RMSE and CC for SSH
create_plot(days, ssh_persistence_rmse, ssh_afno_rmse, 'RMSE', '9-Day Comparison for SSH RMSE', 'ssh_rmse.pdf')
create_plot(days, ssh_persistence_cc, ssh_afno_cc, 'Correlation Coefficient', '9-Day Comparison for SSH CC', 'ssh_cc.pdf')

# Plot RMSE and CC for Zonal
create_plot(days, zonal_persistence_rmse, zonal_afno_rmse, 'RMSE', '9-Day Comparison for Zonal RMSE', 'zonal_rmse.pdf')
create_plot(days, zonal_persistence_cc, zonal_afno_cc, 'Correlation Coefficient', '9-Day Comparison for Zonal CC', 'zonal_cc.pdf')

# Plot RMSE and CC for Meridional
create_plot(days, meridional_persistence_rmse, meridional_afno_rmse, 'RMSE', '9-Day Comparison for Meridional RMSE', 'meridional_rmse.pdf')
create_plot(days, meridional_persistence_cc, meridional_afno_cc, 'Correlation Coefficient', '9-Day Comparison for Meridional CC', 'meridional_cc.pdf')


