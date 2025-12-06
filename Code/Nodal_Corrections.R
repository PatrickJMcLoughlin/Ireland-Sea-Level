# Title: Irish Tide Gauge Nodal Cycle Correction
# Author: Patrick McLoughlin
# Date: October 2025
# Description:
#   This script reads monthly sea-level data for Irish tide gauge stations
#   (Belfast, Malin, Dublin Port), fits an 18.6-year nodal cycle sinusoid,
#   calculates amplitude, phase, and applies nodal correction to produce
#   monthly-corrected series.
#
#   Outputs:
#     - Station-level nodal-corrected CSVs: *_nodal_corrected.csv
#     - Diagnostic plots: *_nodal_fit.png, *_nodal_corrected.png, *_nodal_resid.png
#     - Combined nodal-corrected dataset: all_stations_nodal_corrected_combined_uncapped.csv
#
# Notes:
#   - File paths assume a relative directory: Data/18.6_Year_Cycle_Files
#   - Minimum required data length: 15 years and 12 valid months
#   - Amplitude confidence intervals are calculated using the delta method
#   - Phase is standardized to (-pi, pi] in radians and degrees

# -------------------------------
# Load necessary libraries
# -------------------------------
library(tidyverse)   # For data manipulation & ggplot2
library(lubridate)   # For date handling
library(broom)       # For tidy model output

# -------------------------------
# Project root assumed as script location
data_dir <- "Data/18.6_Year_Cycle_Files"  # relative path

# -------------------------------
# List of station files (use relative paths)
# -------------------------------
files <- c("Belfast.txt", "Malin.txt", "Dublin_Port.txt") %>%
  file.path(data_dir, .)  # prepend folder path

# Nodal period in years
nodal_period <- 18.612

# -------------------------------
# Helper function: standardize phase to (-pi, pi]
# -------------------------------
wrap_phase <- function(rad) {
  ((rad + pi) %% (2*pi)) - pi
}

# -------------------------------
# Function: compute amplitude CI using delta method
# -------------------------------
amplitude_ci <- function(fit, conf_level = 0.95) {
  coefs <- coef(fit)
  a_sin <- coefs["sinN"]
  a_cos <- coefs["cosN"]
  
  amplitude <- sqrt(a_sin^2 + a_cos^2)
  cov_mat <- vcov(fit)[c("sinN", "cosN"), c("sinN", "cosN")]
  grad <- c(a_sin / amplitude, a_cos / amplitude)
  var_amp <- t(grad) %*% cov_mat %*% grad
  se_amp <- sqrt(var_amp)
  
  z <- qnorm((1 + conf_level) / 2)
  lower <- amplitude - z * se_amp
  upper <- amplitude + z * se_amp
  
  list(amplitude = amplitude, lower = lower, upper = upper)
}

# -------------------------------
# Function: fit nodal sinusoid and apply correction
# -------------------------------
fit_and_apply_nodal <- function(file) {
  station <- tools::file_path_sans_ext(basename(file))
  
  df <- read_delim(file, delim = "\t", col_types = cols(
    year = col_integer(),
    month = col_integer(),
    monthly_avg = col_double()
  )) %>%
    mutate(decimal_year = year + (month - 0.5)/12) %>%
    arrange(decimal_year)
  
  df_complete <- df %>% filter(!is.na(monthly_avg))
  
  span_years <- max(df_complete$decimal_year) - min(df_complete$decimal_year)
  if (nrow(df_complete) < 12 || span_years < 15) {
    warning(station, ": insufficient data length (", round(span_years, 1), " years). Skipping.")
    return(NULL)
  }
  
  df_complete <- df_complete %>%
    mutate(
      omega = 2 * pi / nodal_period,
      sinN = sin(omega * decimal_year),
      cosN = cos(omega * decimal_year)
    )
  
  fit <- lm(monthly_avg ~ decimal_year + sinN + cosN, data = df_complete)
  
  amp_info <- amplitude_ci(fit)
  amplitude <- amp_info$amplitude
  amplitude_mm <- amplitude * 1000
  amp_ci_lower <- amp_info$lower * 1000
  amp_ci_upper <- amp_info$upper * 1000
  
  coefs <- coef(fit)
  a_sin <- coefs["sinN"]
  a_cos <- coefs["cosN"]
  
  phase_rad <- wrap_phase(atan2(a_sin, a_cos))   
  phase_deg <- phase_rad * 180/pi
  
  df <- df %>%
    mutate(
      omega = 2 * pi / nodal_period,
      nodal_comp = a_sin * sin(omega * decimal_year) + a_cos * cos(omega * decimal_year),
      monthly_corrected = monthly_avg - nodal_comp
    )
  
  # -------------------------------
  # Plots
  # -------------------------------
  p1 <- ggplot(df, aes(x = decimal_year)) +
    geom_line(aes(y = monthly_avg), color = "steelblue", na.rm = TRUE) +
    geom_line(aes(y = nodal_comp), color = "red", linetype = "dashed", na.rm = TRUE) +
    labs(title = paste0(station, " — original & fitted nodal"),
         subtitle = paste0("Fitted nodal amplitude ≈ ", round(amplitude_mm, 2), " mm"),
         x = "Year", y = "Monthly avg (m)") +
    theme_light()
  
  p2 <- ggplot(df, aes(x = decimal_year)) +
    geom_line(aes(y = monthly_corrected), color = "darkgreen", na.rm = TRUE) +
    labs(title = paste0(station, " — nodal-corrected monthly series"),
         x = "Year", y = "Monthly corrected (m)") +
    theme_light()
  
  residuals <- resid(fit)
  p3 <- ggplot(tibble(decimal_year = df_complete$decimal_year, resid = residuals),
               aes(x = decimal_year, y = resid)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste0(station, " — model residuals"),
         x = "Year", y = "Residual (m)") +
    theme_light()
  
  ggsave(file.path(data_dir, paste0(station, "_nodal_fit.png")), p1, width = 10, height = 3)
  ggsave(file.path(data_dir, paste0(station, "_nodal_corrected.png")), p2, width = 10, height = 3)
  ggsave(file.path(data_dir, paste0(station, "_nodal_resid.png")), p3, width = 10, height = 3)
  
  diagnostics <- list(
    station = station,
    n_obs = nrow(df_complete),
    span_years = span_years,
    amplitude_m = amplitude,
    amplitude_mm = amplitude_mm,
    amp_ci_lower = amp_ci_lower,
    amp_ci_upper = amp_ci_upper,
    phase_rad = phase_rad,
    phase_deg = phase_deg
  )
  
  write_csv(df, file.path(data_dir, paste0(station, "_nodal_corrected.csv")))
  
  list(data = df, diagnostics = diagnostics)
}

# -------------------------------
# Run for all files
# -------------------------------
results_uncapped <- lapply(files, fit_and_apply_nodal)

# -------------------------------
# Combine diagnostic summary
# -------------------------------
diag_summary_uncapped <- map_df(results_uncapped, function(x) {
  if (is.null(x)) return(NULL)
  tibble(
    station       = x$diagnostics$station,
    n_obs         = x$diagnostics$n_obs,
    span_years    = round(x$diagnostics$span_years, 2),
    amplitude_mm  = round(x$diagnostics$amplitude_mm, 3),
    amp_ci_lower  = round(x$diagnostics$amp_ci_lower, 3),
    amp_ci_upper  = round(x$diagnostics$amp_ci_upper, 3),
    phase_rad     = round(x$diagnostics$phase_rad, 3),
    phase_deg     = round(x$diagnostics$phase_deg, 1)
  )
})

print(diag_summary_uncapped)

# -------------------------------
# Save combined corrected data
# -------------------------------
all_corrected <- bind_rows(map(results_uncapped, "data"))
write_csv(all_corrected, file.path(data_dir, "all_stations_nodal_corrected_combined_uncapped.csv"))
