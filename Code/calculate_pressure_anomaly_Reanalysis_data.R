# ===============================================================
# Title: NCEP Ocean Surface Pressure Anomaly Processing
# Author: Patrick McLoughlin
# Date: October 2025
# Description:
#   Compute ocean surface pressure anomalies from NCEP Reanalysis NetCDF files
#   and extract values near specified target tide gauge locations. 
#   These anomalies are used to correct sea level records for atmospheric pressure effects 
#   and to assist in analyzing seasonal cycles in water level data. 
#   The script also calculates wind stress components and fits a linear model including 
#   annual and semiannual harmonics to separate atmospheric and seasonal contributions.
#
# Input files (too large for GitHub; available on Zenodo):
#   - Hisland.nc              : Land-sea mask (land = 1, sea = 0)
#   - pres.sfc.mon.mean.nc    : NCEP monthly mean surface pressure
#
# Output:
#   - ocean_surf_pressure_anomaly_NCEP.Rdata : Contains lat/lon grids, time, and anomaly array
#
# Instructions:
#   1. Download the NCEP NetCDF files from Zenodo.
#   2. Place them in a folder and update the 'data_dir' variable below.
# ===============================================================

# -------------------------------
# Load required libraries
# -------------------------------
library(ncdf4)      # For working with NetCDF files
library(pracma)     # For deg2rad
library(reshape2)   # For data reshaping (optional utility)

# -------------------------------
# Define data directory (update after download)
# -------------------------------
data_dir <- "Data/NCEP_files"  # Placeholder path — update to your local folder after downloading from Zenodo
fn1 <- file.path(data_dir, "Hisland.nc")
fn2 <- file.path(data_dir, "pres.sfc.mon.mean.nc")

# -------------------------------
# Check that files exist
# -------------------------------
stopifnot(file.exists(fn1), file.exists(fn2))

# -------------------------------
# Open NetCDF files
# -------------------------------
nc1 <- nc_open(fn1)
nc2 <- nc_open(fn2)
cat("Files opened successfully!\n")

# -------------------------------
# Extract latitude, longitude, and time data
# -------------------------------
lat <- ncvar_get(nc1, "lat")
lon <- ncvar_get(nc1, "lon")
time <- ncvar_get(nc2, "time")
time <- as.POSIXct(time * 3600, origin = "1800-01-01", tz = "UTC")  # Convert time to POSIXct

# -------------------------------
# Extract data
# -------------------------------
pres <- ncvar_get(nc2, "pres")  # Surface pressure (Pa)
land <- ncvar_get(nc1, "land")  # Land mask (1 = land, 0 = sea)

# -------------------------------
# Apply land mask (convert land to NA)
# -------------------------------
land[land == 1] <- NA  # NA = land, 0 = ocean

# Initialize output arrays
opres <- array(NA, dim = dim(pres))
opresa <- opres

# -------------------------------
# Cosine-latitude weighting
# -------------------------------
lat_weights <- cos(deg2rad(lat))
lat_weights_matrix <- matrix(rep(lat_weights, each = length(lon)),
                             nrow = length(lon), byrow = TRUE)

# Apply land mask to weight matrix
lat_weights_matrix <- lat_weights_matrix + land

# -------------------------------
# Compute anomalies for each timestep
# -------------------------------
for (k in 1:length(time)) {
  cat("Processing timestep:", k, "/", length(time), "\n")
  
  # Apply land mask to pressure
  opres[,,k] <- pres[,,k] + land
  
  # Compute cosine-weighted global mean pressure
  gpres <- sum(opres[,,k] * lat_weights_matrix, na.rm = TRUE) /
           sum(lat_weights_matrix, na.rm = TRUE)
  
  # Compute pressure anomaly
  opresa[,,k] <- opres[,,k] - gpres
}

# -------------------------------
# Save output
# -------------------------------
save(lat, lon, time, opresa,
     file = file.path(data_dir, "ocean_surf_pressure_anomaly_NCEP.Rdata"))
cat("Processing complete. Anomaly data saved.\n")

# -------------------------------
# Close NetCDF files
# -------------------------------
nc_close(nc1)
nc_close(nc2)
