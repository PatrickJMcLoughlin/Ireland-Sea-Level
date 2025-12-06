# Title: Ocean Surface Pressure Anomaly Processing
# Author: Patrick McLoughlin
# Date: October 2025
# Description:
#   Compute ocean surface pressure anomalies from high-resolution Copernicus ERA NetCDF files
#   and extract values near specified target tide gauge locations. 
#   These anomalies are used to correct sea level records for atmospheric pressure effects 
#   and to assist in analyzing seasonal cycles in water level data. 
#   The script also calculates wind stress components and fits a linear model including 
#   annual and semiannual harmonics to separate atmospheric and seasonal contributions.
#
# Input files (too large for GitHub; available on Zenodo):
#   - lsm.nc         : Land-sea mask
#   - data_0.nc      : ERA surface pressure
#
# Output:
#   - ocean_surf_pressure_anomaly.Rdata : Contains lat/lon grids, time, and anomaly array
#
# Instructions:
#   1. Download NetCDF files from Zenodo.
#   2. Place them in a folder and update the 'data_dir' variable below.

# -------------------------------
# Load required libraries
# -------------------------------
library(ncdf4)   # Reading NetCDF
library(pracma)  # deg2rad
library(fields)  # interp.surface.grid

# -------------------------------
# Define data directory (update after download)
# -------------------------------
data_dir <- "Data/ERA_files"  # Change to your local folder; this is only a placeholder example
fn1 <- file.path(data_dir, "lsm.nc")
fn2 <- file.path(data_dir, "data_0.nc")

# -------------------------------
# Check files exist
# -------------------------------
stopifnot(file.exists(fn1), file.exists(fn2))

# -------------------------------
# Open NetCDF files
# -------------------------------
nc1 <- nc_open(fn1)
nc2 <- nc_open(fn2)
cat("Files opened successfully!\n")

# -------------------------------
# Extract grids
# -------------------------------
lat_land <- ncvar_get(nc1, "latitude")
lon_land <- ncvar_get(nc1, "longitude")
lat_pres <- ncvar_get(nc2, "latitude")
lon_pres <- ncvar_get(nc2, "longitude")

# -------------------------------
# Extract data
# -------------------------------
land <- ncvar_get(nc1, "lsm")
pres <- ncvar_get(nc2, "sp")
time <- as.POSIXct(ncvar_get(nc2, "valid_time"), origin = "1970-01-01", tz = "UTC")

# -------------------------------
# Interpolate land mask to pressure grid
# -------------------------------
land_interp <- interp.surface.grid(
  list(x = lon_land, y = lat_land, z = land),
  list(x = lon_pres, y = lat_pres)
)$z

land_interp[land_interp > 0.5] <- NA
land_interp[land_interp <= 0.5] <- 1

# -------------------------------
# Cosine-latitude weighting
# -------------------------------
lat_weights <- cos(deg2rad(lat_pres))
lat_weights <- lat_weights / sum(lat_weights, na.rm = TRUE)
lat_weights_matrix <- matrix(rep(lat_weights, each = length(lon_pres)),
                             nrow = length(lon_pres), byrow = TRUE)

# -------------------------------
# Compute anomalies per timestep
# -------------------------------
opresa <- array(NA, dim = dim(pres))

for (k in 1:dim(pres)[3]) {
    pres_time <- pres[,,k]
    opres <- pres_time * land_interp
    gpres <- sum(opres * lat_weights_matrix, na.rm = TRUE) / sum(lat_weights_matrix * (!is.na(opres)), na.rm = TRUE)
    opresa[,,k] <- opres - gpres
}

# -------------------------------
# Save output
# -------------------------------
save(lat_pres, lon_pres, time, opresa,
     file = file.path(data_dir, "ocean_surf_pressure_anomaly.Rdata"))
cat("Processing complete. Anomaly data saved.\n")

# -------------------------------
# Close NetCDF files
# -------------------------------
nc_close(nc1)
nc_close(nc2)
