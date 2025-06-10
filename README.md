# 🌊 AFNO for Sea Surface Prediction over the Bay of Bengal

## Project Title

Physics-Guided Adaptive Fourier Neural Operator Model for Synoptic Forecasting of the Sea Surface Dynamics in the Bay of Bengal

[Bay of Bengal](bay_of_bengal_map.pdf)

## 📌 Project Description

Sea surface dynamics such as **Sea Surface Temperature (SST)**, **Sea Surface Salinity (SSS)**, **Sea Surface Height (SSH)** and **zonal/meridional currents(SSC)** are crucial indicators for understanding ocean circulation, climate variability, and marine ecosystems. This project applies **AFNO** to learn the temporal evolution of these variables from historical atmospheric and oceanic reanalysis datasets.


## 🚀 How to Run the Code


## Installation 🛠️

```bash
# Clone the repository
git clone https://github.com/rishibabby/AFNO_BOB.git
cd AFNO_BOB
```

## Data Preparation

This project uses data from two primary sources:

1. **Atmospheric Data (ERA5)**
   - Source: [Copernicus Climate Data Store - ERA5 Daily Single Levels](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview)
   - Description: Provides daily-averaged atmospheric variables from the ERA5 reanalysis dataset.
   - Access: Registration is required. After logging in, select the desired variables and time range to download the data.

2. **Oceanographic Data (GLORYS)**
   - Source: [Copernicus Marine Environment Monitoring Service - GLOBAL_MULTIYEAR_PHY_001_030](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
   - Description: Offers multi-year global ocean physical reanalysis including sea surface temperature, salinity, and currents.
   - Access: Registration is required. Use the provided GUI or API to download relevant data for your study region and period.


## Run the Code 🚀

Run the AFNO training script with:

```bash
python3 -m scripts.run_afno 
```

* `--config` points to a YAML file defining data paths, model hyper‑parameters and training schedule.
* Trained checkpoint is stored in saved_models folder
## Visualisation 📊

After training, generate:

```bash
# Visualisation of one day head plot
python3 -m visualisation.onedayplot
```




