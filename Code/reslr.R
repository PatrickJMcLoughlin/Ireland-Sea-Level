# =========================================================
# Title: Irish Tide Gauge Relative Sea Level Analysis
# Author: Patrick McLoughlin
# Date: October 2025
# =========================================================

# =========================================================
# Description:
#   This script processes tide gauge data from Irish sites:
#     - Malin Head, Belfast, Dublin Port, Galway, Cork, Tarbert
#   
#   Main workflow:
#     1. Load tide gauge CSVs and standardize site names
#     2. Apply ICE6G_BC_hybrid_71p320 GIA corrections
#     3. Run NI-GAM decomposition using the reslr package
#     4. Extract yearly site-specific RSL rates (GIA-adjusted and GIA-free)
#     5. Compute raw tide gauge trends (fully GIA-independent)
#     6. Generate plots for model input, model fit, rates, and NI-GAM components
#     7. Save CSV outputs per site for both GIA-adjusted and GIA-free rates
#     8. Save full NI-GAM output for reproducibility (RDS)
#     9. Perform cross-validation and compute error metrics
# =========================================================

# =========================================================
# Outputs:
#   - Site-level RSL rates (GIA-adjusted): rate_plot_exact_from_model/*.csv
#   - Site-level RSL rates (GIA-free): rate_plot_exact_from_model_noGIA/*.csv
#   - Plots:
#       *_input_plot.png
#       model_fit_plot.png
#       rate_plot.png
#       regional_plot.png
#       regional_rate_plot.png
#       linear_local_plot.png
#       nigam_component_plot.png
#       plots_cross_validation/cross_validation_true_vs_predicted.png
#   - NI-GAM R output for reproducibility: output_ICE6G_BC_hybrid_71p320.rds
# =========================================================

# =========================================================
# Notes:
#   - File paths assume a GitHub folder structure: Data/reslr/
#   - Use set.seed() for reproducibility control
#   - Modify date ranges for modern or 2000s analyses in `modern_ranges` or `ranges_2000s`
#   - GIA corrections use ICE6G_BC_hybrid_71p320 values (mm/yr)
# =========================================================

# =========================================================
# 📦 Load libraries
# =========================================================
library(reslr)
library(tidyverse)
library(ncdf4)
library(coda)

# =========================================================
# 🏠 Setup
# =========================================================
data_dir <- "Data/reslr"
set.seed(1234)

# =========================================================
# 🔧 Load helper functions
# =========================================================
source("match.closest.R")
source("cross_val_check.R")
source("internal_functions.R")

# =========================================================
# 🛠 Define NI-GAM helper function
# =========================================================
run_ni_gam <- function(df, model_type = "ni_gam_decomp", include_tg = FALSE) {
  input <- reslr_load(
    df,
    include_linear_rate = TRUE,
    include_tide_gauge = include_tg,
    prediction_grid_res = 1
  )
  output <- reslr_mcmc(
    input,
    model_type = model_type,
    spline_nseg_t = 3,
    spline_nseg_st = 2
  )
  return(output)
}

# =========================================================
# 🔁 Load site CSVs and standardize names
# =========================================================
site_files <- c(
  malin   = "Malin_Head_reslr.csv",
  belfast = "Belfast_reslr.csv",
  dp      = "DP_reslr.csv",
  galway  = "Galway_High_resdata_reslr.csv",
  cork    = "Cork_reslr.csv",
  tarbert = "Tarbert_reslr.csv"
)

site_list <- lapply(site_files, read.csv)

# Standardize site names
site_list <- lapply(site_list, function(df) {
  df %>% mutate(Site = case_when(
    grepl("Dublin Port|DP", Site)  ~ "Dublin_Port",
    grepl("Belfast", Site)         ~ "Belfast",
    grepl("Malin Head", Site)      ~ "Malin_Head",
    grepl("Galway", Site)          ~ "Galway",
    grepl("Cork", Site)            ~ "Cork",
    grepl("Tarbert", Site)         ~ "Tarbert",
    TRUE ~ Site
  ))
})

site_combined <- bind_rows(site_list)

# =========================================================
# 🌍 Define ICE6G_BC_hybrid_71p320 GIA rates
# =========================================================
ICE6G_BC_hybrid_71p320 <- c(
  "Malin_Head"   = 0.03,
  "Belfast"      = -0.07,
  "Dublin_Port"  = 0.04,
  "Galway"       = 0.16,
  "Cork"         = 0.21,
  "Tarbert"      = 0.21
)

# =========================================================
# 🔢 Prepare model input
# =========================================================
df_input <- site_combined %>%
  filter(Age <= 2025) %>%
  mutate(
    linear_rate = sapply(Site, function(s) ICE6G_BC_hybrid_71p320[[s]]),
    linear_rate_err = 0.3
  )

# =========================================================
# ▶ Run NI-GAM
# =========================================================
model_name <- "ICE6G_BC_hybrid_71p320"

# Create plot directory
plot_dir <- paste0("plots_", model_name)
if(!dir.exists(plot_dir)) dir.create(plot_dir)

# Optional: Save input data plot before modeling
input_plot <- plot(reslr_load(df_input, include_linear_rate = TRUE), 
                   plot_caption = FALSE,
                   plot_tide_gauges = TRUE)
ggsave(filename = file.path(plot_dir, "input_data_plot.png"),
       plot = input_plot, width = 10, height = 6, dpi = 600)

# Run NI-GAM
output <- run_ni_gam(df_input)

# =========================================================
# 📊 Save all model plots
# =========================================================
plot_types <- c(
  "model_fit_plot", "rate_plot", "regional_plot", 
  "regional_rate_plot", "linear_local_plot", "nigam_component_plot"
)

for(pt in plot_types){
  p <- plot(output, plot_type = pt)
  ggsave(plot = p, filename = file.path(plot_dir, paste0(pt, ".png")))
}

# =========================================================
# ✅ Extract exact yearly site-specific rates (GIA-adjusted)
# =========================================================
df_rates <- output$output_dataframes$regional_component_df %>%
  mutate(
    Total_Rate = rate_pred + linear_rate,             
    Site_clean = gsub("[^A-Za-z0-9]", "_", SiteName)
  ) %>%
  select(Site = Site_clean, Year = Age, Total_Rate) %>%
  arrange(Site, Year)

# Save CSVs per site (GIA-adjusted)
csv_dir <- file.path(plot_dir, "rate_plot_exact_from_model")
if(!dir.exists(csv_dir)) dir.create(csv_dir)

for(site in unique(df_rates$Site)){
  df_site <- df_rates %>% filter(Site == site)
  write.csv(df_site, file.path(csv_dir, paste0(site, "_rates.csv")), row.names = FALSE)
}

cat("Saved CSVs for all sites (GIA-adjusted) to:", csv_dir, "\n")

# =========================================================
# ✅ Extract exact yearly site-specific rates (GIA-free)
# =========================================================
df_rates_noGIA <- output$output_dataframes$regional_component_df %>%
  mutate(
    Rate_noGIA = rate_pred,
    Site_clean = gsub("[^A-Za-z0-9]", "_", SiteName)
  ) %>%
  select(Site = Site_clean, Year = Age, Rate_noGIA) %>%
  arrange(Site, Year)

# Save CSVs per site (GIA-free)
csv_dir_noGIA <- file.path(plot_dir, "rate_plot_exact_from_model_noGIA")
if(!dir.exists(csv_dir_noGIA)) dir.create(csv_dir_noGIA)

for(site in unique(df_rates_noGIA$Site)){
  df_site <- df_rates_noGIA %>% filter(Site == site)
  write.csv(df_site, file.path(csv_dir_noGIA, paste0(site, "_rates_noGIA.csv")), row.names = FALSE)
}

cat("Saved CSVs for all sites (GIA-free) to:", csv_dir_noGIA, "\n")

# =========================================================
# 🔢 Compute raw Tide-Gauge trends (fully GIA-independent, mm/yr)
# =========================================================
tg_trends <- lapply(site_list, function(df){
  df %>%
    filter(Age <= 2025) %>%
    group_by(Site) %>%
    summarise(
      start_year = min(Age),
      end_year   = max(Age),
      n_years    = n(),
      tg_rate = coef(lm(RSL ~ Age, data = pick(everything())))[2] * 1000,  # mm/yr
      .groups = "drop"
    )
}) %>% bind_rows()

cat("\nRaw Tide-Gauge RSL rates (GIA-free) per site:\n")
print(tg_trends)

# Regional TG rate ± SE
regional_tg_rate <- mean(tg_trends$tg_rate)
tg_rate_se       <- sd(tg_trends$tg_rate) / sqrt(nrow(tg_trends))
cat("\nRegional Tide-Gauge RSL rate ± SE:", round(regional_tg_rate,2), "±", round(tg_rate_se,2), "mm/yr\n")

# =========================================================
# 🔢 Compute mean instantaneous rates per site (NI-GAM)
# =========================================================
mean_rates_per_site <- df_rates %>%
  group_by(Site) %>%
  summarise(
    start_year = min(Year),
    end_year   = max(Year),
    n_years    = n(),
    mean_rate  = mean(Total_Rate),
    .groups = "drop"
  ) %>% arrange(Site)

print(mean_rates_per_site)

# =========================================================
# 🔍 Regional component (1925–2024)
# =========================================================
regional_recent <- output$output_dataframes$regional_component_df %>%
  filter(Age >= 1925, Age <= 2024) %>%
  arrange(Age) %>%
  mutate(
    Total_Rate = rate_pred + linear_rate,            # GIA-adjusted
    RSL_cumulative_noGIA   = cumsum(rate_pred),      # GIA-free
    RSL_cumulative_withGIA = cumsum(Total_Rate)     # GIA-adjusted
  )

n_years <- nrow(regional_recent)
total_rise_noGIA   <- tail(regional_recent$RSL_cumulative_noGIA, 1)
total_rise_withGIA <- tail(regional_recent$RSL_cumulative_withGIA, 1)
avg_rate_noGIA     <- total_rise_noGIA / n_years
avg_rate_withGIA   <- total_rise_withGIA / n_years

cat("\nRegional NI-GAM RSL 1925–2024:\n")
cat("GIA-free (NI-GAM): ", round(total_rise_noGIA,2), "mm, Avg rate:", round(avg_rate_noGIA,3), "mm/yr\n")
cat("GIA-adjusted (NI-GAM): ", round(total_rise_withGIA,2), "mm, Avg rate:", round(avg_rate_withGIA,3), "mm/yr\n")

# Mean ± SE
mean_rate <- mean(regional_recent$Total_Rate)
sd_rate   <- sd(regional_recent$Total_Rate) / sqrt(n_years)
cat("Mean rate ± SE (GIA-adjusted):", round(mean_rate,1), "±", round(sd_rate,1), "mm/yr\n")
mean_rate_noGIA <- mean(regional_recent$rate_pred)
sd_rate_noGIA   <- sd(regional_recent$rate_pred) / sqrt(n_years)
cat("Mean rate ± SE (GIA-free):", round(mean_rate_noGIA,1), "±", round(sd_rate_noGIA,1), "mm/yr\n")

# =========================================================
# 💾 Save NI-GAM output for reproducibility
# =========================================================
saveRDS(output, file = paste0("output_", model_name, ".rds"))

# =========================================================
# Cross-validation
# =========================================================
df_cv <- site_combined %>%
  filter(Age <= 2025) %>%
  mutate(
    SiteName = Site,
    linear_rate = sapply(Site, function(s) ICE6G_BC_hybrid_71p320[[s]]),
    linear_rate_err = 0.3
  )

cv_all <- cross_val_check(
  data = df_cv,
  model_type = "ni_gam_decomp",
  n_iterations = 5000,
  n_burnin = 1000,
  n_thin = 5,
  n_chains = 3,
  spline_nseg_t = 3,
  spline_nseg_st = 2,
  seed = 1235,
  n_fold = 10,
  CI = 0.95
)

# Plot predicted vs observed
cv_plot <- plot(cv_all$true_pred_plot) + ggtitle("Cross-validation: True vs Predicted")
if(!dir.exists("plots_cross_validation")) dir.create("plots_cross_validation")
ggsave(file.path("plots_cross_validation", "cross_validation_true_vs_predicted.png"),
       plot = cv_plot, width = 10, height = 6, dpi = 600)

# Overall coverage
message("Overall coverage (95% PI): ", round(cv_all$total_coverage * 100, 2), "%")
print(cv_all$coverage_by_site)

# =========================================================
# Prediction interval widths
# =========================================================
prediction_interval_size <- cv_all$prediction_interval_size %>%
  mutate(PI_width = abs(PI_width))
print(prediction_interval_size)

# =========================================================
# Error metrics (ME, MAE, RMSE) - compute manually
# =========================================================
cv_data <- cv_all$true_pred_plot$data %>%
  mutate(error = true_RSL - pred_RSL)
# Overall
ME_overall   <- mean(cv_data$error)
MAE_overall  <- mean(abs(cv_data$error))
RMSE_overall <- sqrt(mean(cv_data$error^2))

# By site
site_metrics <- cv_data %>%
  group_by(SiteName) %>%
  summarise(
    ME   = mean(error),
    MAE  = mean(abs(error)),
    RMSE = sqrt(mean(error^2))
  )
print(site_metrics)
