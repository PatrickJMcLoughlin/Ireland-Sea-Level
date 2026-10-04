# =========================================================
# RESLR MODEL SETUP + FULL POSTERIOR RATE ANALYSIS
# =========================================================
# Title: A Century of Sea-Level Change in Ireland (1925–2024)
# Author: Patrick McLoughlin
# Date: October 2026
#
# Description:
# This script performs the RESLR modelling and posterior
# calculations for the paper:
#
# "A Century of Sea-Level Change in Ireland (1925–2024)"
#
# Part A performs the main RESLR modelling, posterior
# calculations, site-rate analysis, and associated plots.
#
# Part B calculates cumulative modelled sea-level change
# across the six Irish sites between 1925 and 2024.
#
# Part C compares the Ireland six-site mean of the total
# modelled RESLR RSL with global tide-gauge and satellite
# altimetry sea-level observations.
#
# Part D performs cross-validation of the RESLR model and
# produces the cross-validation figures.
#
# The file:
# observed_and_projected_change_in_global_mean_sea_level.csv
# is used in Part C to compare the RESLR results with
# global mean sea-level observations.
#
# Repository data structure:
#
# The main analysis script is stored in the Code folder.
#
# The six Irish tide-gauge station files and the global
# sea-level comparison dataset are stored directly in the
# Data folder.
#
# The Data/reslr subfolder contains the local R source files
# and JAGS model file required by the RESLR analysis.
#
# The Code and Data folders should be downloaded together
# from the GitHub repository. The repository root should be
# used as the working directory so that the relative file
# paths used below remain unchanged.
#
# The model output file model_output_setup2.RDS is generated
# by the analysis and is subsequently used to extend the
# prediction grids for the individual sites.
#
# This version retains all rate outputs and plots and applies
# consistent uncertainty handling. All displayed "95% CI"
# intervals are 95% Bayesian credible intervals calculated
# directly from posterior draws whenever posterior samples
# are available.
#
# IMPORTANT RESLR CONVENTION
# --------------------------
# In reslr's output data frames, the rate interval columns are
# stored in the opposite naming order:
#
#   rate_upr = TRUE LOWER posterior quantile
#   rate_lwr = TRUE UPPER posterior quantile
#
# The same naming issue exists for the ordinary RSL interval:
#
#   upr = TRUE LOWER posterior quantile
#   lwr = TRUE UPPER posterior quantile
#
# Therefore this script does NOT use those dataframe interval
# columns for the publication uncertainty ribbons. Instead it
# calculates all plotted posterior intervals directly from:
#
#   mu_pred
#   mu_pred_deriv
#   r_pred
#   r_pred_deriv
#   g_h_z_x_pred
#   g_z_x_pred_deriv
#   l_pred
#   l_pred_deriv
#
# IMPORTANT LINEAR-LOCAL RATE CONVENTION
# --------------------------------------
# g_h_z_x_pred is the posterior RSL component containing the
# site-specific vertical offset + linear local component.
#
# The actual posterior derivative of the linear local component
# is:
#
#   g_z_x_pred_deriv
#
# Therefore g_h_z_x_pred MUST NOT be used as a rate posterior.
#
# The RESLR posterior derivatives are already on the mm/yr scale
# because reslr internally works with age in thousands of years.
#
# mu_pred
#   Total modelled relative sea-level (RSL) posterior.
#   Units: metres (m).
#   Used in Part B and C to plot the six-site mean modelled RSL
#   against global tide-gauge and satellite-altimetry
#   sea-level series.
#
# mu_pred_deriv
#   Posterior derivative of the total modelled RSL.
#   Units: mm/yr.
#   Used in Part A for the site rates.
#
# r_pred
#   Regional component of modelled RSL.
#   Units: metres (m).
#
# r_pred_deriv
#   Posterior derivative of the regional component.
#   Units: mm/yr.
#
# IMPORTANT:
#   Part A -> mu_pred_deriv
#   Part B -> mu_pred
#   Part C -> mu_pred
#
# These objects are not interchangeable:
#
#   mu_pred       = RSL level (m)
#   mu_pred_deriv = RSL rate (mm/yr)
#
# =========================================================

# =========================================================
# PART A: RESLR MODELLING AND POSTERIOR RATE ANALYSIS
# =========================================================
# Runs the RESLR model, extracts posterior samples,
# calculates site-specific RSL rates and uncertainty,
# and generates the associated model and rate plots.
# =========================================================

library(reslr)
library(tidyverse)
library(ncdf4)
library(coda)
library(ggplot2)
library(dplyr)


# =========================================================
# WORKING DIRECTORY
# =========================================================
# =========================================================
# WORKING DIRECTORY + ZENODO MODEL OUTPUT
# =========================================================

# The large model-output file is stored on Zenodo rather than
# GitHub because the file is too large for the repository.
#
# Zenodo record:
# https://zenodo.org/records/23085723
#
# The script downloads the file automatically if it is not
# already present in the Data folder.
#
# IMPORTANT:
# Run this script with the GitHub repository root as the
# working directory.

MODEL_OUTPUT_FILE <- file.path(
  "Data",
  "model_output_setup2.RDS"
)

ZENODO_MODEL_URL <- paste0(
  "https://zenodo.org/records/23085723/files/",
  "model_output_setup2.RDS?download=1"
)

# Check that the Data folder exists
if (!dir.exists("Data")) {
  dir.create("Data", recursive = TRUE)
}

# Download the model output from Zenodo if it is not
# already present locally
if (!file.exists(MODEL_OUTPUT_FILE)) {

  message(
    "\nmodel_output_setup2.RDS was not found locally.\n",
    "Downloading the model output from Zenodo.\n",
    "The file is large and this may take some time.\n"
  )

  download.file(
    url = ZENODO_MODEL_URL,
    destfile = MODEL_OUTPUT_FILE,
    mode = "wb"
  )
}

# Check that the model output now exists
if (!file.exists(MODEL_OUTPUT_FILE)) {
  stop(
    "model_output_setup2.RDS could not be found or downloaded."
  )
}

message(
  "\nUsing model output: ",
  MODEL_OUTPUT_FILE,
  "\n"
)

# Reproducible random operations
set.seed(1234)


# =========================================================
# LOAD RESLR FUNCTIONS
# =========================================================

source("Data/reslr/match.closest.R")
source("Data/reslr/cross_val_check.R")
source("Data/reslr/internal_functions_old.R")
source("Data/reslr/reslr_load.R")
source("Data/reslr/reslr_mcmc_old.R")

# =========================================================
# GLOBAL SETTINGS
# =========================================================

# Credible interval (CI)
CI_LEVEL <- 0.95
CI_LOW_PROB <- (1 - CI_LEVEL) / 2
CI_HIGH_PROB <- 1 - CI_LOW_PROB

# Model period
MODEL_START <- 1925
MODEL_END <- 2024


# =========================================================
# GIA VALUES
# Units: mm/yr
# =========================================================

GIA <- c(
  Malin_Head  = 0.03,
  Belfast     = -0.07,
  Dublin_Port = 0.04,
  Galway      = 0.16,
  Cork        = 0.21,
  Tarbert     = 0.21
)


# =========================================================
# LOAD SITE DATA
# =========================================================

site_files <- c(
  Malin_Head = "Data/Malin_Head_reslr.csv",
  Belfast    = "Data/Belfast_reslr.csv",
  Dublin_Port = "Data/Dublin_Port_reslr.csv",
  Galway     = "Data/Galway_reslr.csv",
  Cork       = "Data/Cork_reslr.csv",
  Tarbert    = "Data/Tarbert_reslr.csv"
)

site_list <- lapply(
  site_files,
  read_csv,
  show_col_types = FALSE
)

# =========================================================
# COMBINE SITE DATA
# =========================================================

site_combined <- bind_rows(site_list) %>%
  mutate(
    Site = case_when(
      grepl("Dublin|DP", Site, ignore.case = TRUE) ~ "Dublin_Port",
      grepl("Belfast", Site, ignore.case = TRUE) ~ "Belfast",
      grepl("Malin", Site, ignore.case = TRUE) ~ "Malin_Head",
      grepl("Galway", Site, ignore.case = TRUE) ~ "Galway",
      grepl("Cork", Site, ignore.case = TRUE) ~ "Cork",
      grepl("Tarbert", Site, ignore.case = TRUE) ~ "Tarbert",
      TRUE ~ Site
    )
  )


# =========================================================
# CHECK SITE NAMES
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("SITE CHECK\n")
cat("=========================================================\n")

print(
  sort(unique(site_combined$Site))
)


# =========================================================
# MODEL INPUT
# SAFE GIA MAP
# =========================================================
#
# MODEL_END is used here so that observations after the intended
# analysis end date are not silently incorporated into the model.
#
# If the model is deliberately intended to include later data,
# change MODEL_END above rather than changing this filter.
# =========================================================

df_input <- site_combined %>%
  filter(
    Age <= MODEL_END
  ) %>%
  mutate(
    linear_rate = unname(GIA[Site]),
    linear_rate_err = 0.3
  )


# =========================================================
# HARD CHECKS
# =========================================================

stopifnot(
  !any(is.na(df_input$linear_rate))
)

stopifnot(
  all(df_input$Site %in% names(GIA))
)


# =========================================================
# PRINT INPUT SUMMARY
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("MODEL INPUT SUMMARY\n")
cat("=========================================================\n")

print(
  df_input %>%
    count(Site)
)


# =========================================================
# RESLR INPUT
# prediction_grid_res = 1
# =========================================================

input <- reslr_load(
  df_input,
  include_linear_rate = TRUE,
  include_tide_gauge = FALSE,
  prediction_grid_res = 1
)


# =========================================================
# OUTPUT PLOT DIRECTORY
# =========================================================
# Directory for saving model plots
# Modify this path if needed

plot_dir <- file.path(
  getwd(),
  "plots_gm_modelsetup2"
)

dir.create(
  plot_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# =========================================================
# INPUT DATA PLOT
# =========================================================

p_input <-
  plot(
    input,
    plot_tide_gauges = TRUE,
    plot_caption = FALSE
  ) +
  theme(
    strip.text = element_text(
      size = 12,
      face = "bold"
    ),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    panel.spacing = unit(
      1.2,
      "lines"
    )
  )

ggsave(
  filename = file.path(
    plot_dir,
    "Figure_3_input_plot.png"
  ),
  plot = p_input,
  width = 12,
  height = 8,
  dpi = 600
)


# =========================================================
# RESLR MODEL OUTPUT
# =========================================================

# Set to FALSE to use the archived model output from Zenodo.
#
# Set to TRUE if you want to rerun the RESLR model from
# the input data and generate a new model_output_setup2.RDS.
#
# FALSE is recommended for reproducing the published
# analysis using the archived model output.

RERUN_MODEL <- FALSE


if (RERUN_MODEL) {

  # =======================================================
  # RUN RESLR MODEL FROM SCRATCH
  # =======================================================

  message(
    "\nRunning RESLR model from input data.\n",
    "This may take a substantial amount of time.\n"
  )

  output <- reslr_mcmc_old(
    input,
    model_type = "ni_gam_decomp",
    spline_nseg_t = 2,
    spline_nseg_st = 1
  )

  # Save the newly generated model output locally
  saveRDS(
    output,
    MODEL_OUTPUT_FILE
  )

  message(
    "\nNew model output saved to: ",
    MODEL_OUTPUT_FILE,
    "\n"
  )


} else {

  # =======================================================
  # LOAD ARCHIVED MODEL OUTPUT FROM ZENODO
  # =======================================================

  message(
    "\nLoading archived RESLR model output.\n"
  )

  output <- readRDS(
    MODEL_OUTPUT_FILE
  )

}

# =========================================================
# EXTRACT TOTAL MODEL DATA
# =========================================================

data <- output$output_dataframes$total_model_df


# =========================================================
# EXTRACT MODEL COMPONENT DATA
# =========================================================

regional_component_df <-
  output$output_dataframes$regional_component_df

lin_loc_component_df <-
  output$output_dataframes$lin_loc_component_df

non_lin_loc_component_df <-
  output$output_dataframes$non_lin_loc_component_df


# =========================================================
# EXTRACT POSTERIOR SAMPLES
# =========================================================

post <-
  output$noisy_model_run_output$BUGSoutput$sims.list


# =========================================================
# REQUIRED POSTERIOR OBJECTS
# =========================================================
#
# mu_pred              = total RSL posterior
# mu_pred_deriv        = total rate posterior
# r_pred               = regional RSL posterior
# r_pred_deriv         = regional rate posterior
# g_h_z_x_pred         = vertical offset + linear local RSL
# g_z_x_pred_deriv     = linear local rate posterior
# l_pred               = non-linear local RSL
# l_pred_deriv         = non-linear local rate posterior
# =========================================================

mu <- post$mu_pred
mu_rate <- post$mu_pred_deriv

r_regional <- post$r_pred
r_regional_rate <- post$r_pred_deriv

lin_loc_post <- post$g_h_z_x_pred
lin_loc_rate_post <- post$g_z_x_pred_deriv

non_lin_loc_post <- post$l_pred
non_lin_loc_rate <- post$l_pred_deriv


# =========================================================
# HARD POSTERIOR CHECKS
# =========================================================

required_objects <- c(
  "mu_pred",
  "mu_pred_deriv",
  "r_pred",
  "r_pred_deriv",
  "g_h_z_x_pred",
  "g_z_x_pred_deriv",
  "l_pred",
  "l_pred_deriv"
)

missing_objects <- required_objects[
  !vapply(
    required_objects,
    function(x) !is.null(post[[x]]),
    logical(1)
  )
]

if (length(missing_objects) > 0) {
  stop(
    paste(
      "Required posterior object(s) missing:",
      paste(missing_objects, collapse = ", ")
    )
  )
}


# =========================================================
# POSTERIOR DIMENSION CHECKS
# =========================================================

posterior_objects <- list(
  mu_pred = mu,
  mu_pred_deriv = mu_rate,
  r_pred = r_regional,
  r_pred_deriv = r_regional_rate,
  g_h_z_x_pred = lin_loc_post,
  g_z_x_pred_deriv = lin_loc_rate_post,
  l_pred = non_lin_loc_post,
  l_pred_deriv = non_lin_loc_rate
)

for (nm in names(posterior_objects)) {

  x <- posterior_objects[[nm]]

  if (length(dim(x)) != 2) {
    stop(
      paste(
        nm,
        "must be a two-dimensional posterior matrix."
      )
    )
  }
}


# =========================================================
# TOTAL MODEL DATAFRAME / PREDICTION GRID
# =========================================================

grid <- output$output_dataframes$total_model_df %>%
  mutate(
    Site_clean = case_when(
      grepl("Belfast", SiteName, ignore.case = TRUE) ~ "Belfast",
      grepl("Cork", SiteName, ignore.case = TRUE) ~ "Cork",
      grepl("Dublin|DP", SiteName, ignore.case = TRUE) ~ "Dublin_Port",
      grepl("Malin", SiteName, ignore.case = TRUE) ~ "Malin_Head",
      grepl("Galway", SiteName, ignore.case = TRUE) ~ "Galway",
      grepl("Tarbert", SiteName, ignore.case = TRUE) ~ "Tarbert",
      TRUE ~ SiteName
    ),
    idx = row_number()
  )


# =========================================================
# POSTERIOR / GRID ALIGNMENT
# =========================================================

if (ncol(mu) != nrow(grid)) {
  stop(
    paste(
      "mu_pred has", ncol(mu),
      "prediction columns but total_model_df has",
      nrow(grid),
      "rows."
    )
  )
}

if (ncol(mu_rate) != nrow(grid)) {
  stop(
    "mu_pred_deriv and the prediction grid are not aligned."
  )
}

if (ncol(r_regional) != nrow(grid)) {
  stop(
    "r_pred and the prediction grid are not aligned."
  )
}

if (ncol(r_regional_rate) != nrow(grid)) {
  stop(
    "r_pred_deriv and the prediction grid are not aligned."
  )
}

if (ncol(lin_loc_post) != nrow(grid)) {
  stop(
    "g_h_z_x_pred and the prediction grid are not aligned."
  )
}

if (ncol(lin_loc_rate_post) != nrow(grid)) {
  stop(
    "g_z_x_pred_deriv and the prediction grid are not aligned."
  )
}

if (ncol(non_lin_loc_post) != nrow(grid)) {
  stop(
    "l_pred and the prediction grid are not aligned."
  )
}

if (ncol(non_lin_loc_rate) != nrow(grid)) {
  stop(
    "l_pred_deriv and the prediction grid are not aligned."
  )
}


# =========================================================
# SITE INDEX
# =========================================================

site_index <- split(
  seq_len(nrow(grid)),
  grid$Site_clean
)


# =========================================================
# CHECK SITE INDEX
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("PREDICTION GRID CHECK\n")
cat("=========================================================\n")

cat(
  "Prediction-grid rows: ",
  nrow(grid),
  "\n"
)

cat(
  "Posterior draws: ",
  nrow(mu),
  "\n"
)

cat(
  "Posterior prediction locations: ",
  ncol(mu),
  "\n"
)

cat(
  "Number of sites: ",
  length(site_index),
  "\n"
)

print(
  sapply(site_index, length)
)


# =========================================================
# ANALYSIS-WINDOW INDEX
# =========================================================
#
# The RESLR prediction grid may extend beyond the publication
# analysis window. All rate summaries intended to represent
# 1925-2024 are therefore explicitly restricted here.
# =========================================================

analysis_grid_idx <- which(
  grid$Age >= MODEL_START &
    grid$Age <= MODEL_END
)

if (length(analysis_grid_idx) == 0) {
  stop(
    paste0(
      "No prediction-grid locations fall within the analysis ",
      "window ",
      MODEL_START,
      "-",
      MODEL_END,
      "."
    )
  )
}

analysis_years <- sort(
  unique(
    grid$Age[analysis_grid_idx]
  )
)


# =========================================================
# CHECK ANALYSIS WINDOW
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("ANALYSIS WINDOW CHECK\n")
cat("=========================================================\n")

cat(
  "Requested analysis window: ",
  MODEL_START,
  "-",
  MODEL_END,
  "\n"
)

cat(
  "Prediction years available in analysis window: ",
  min(analysis_years),
  "-",
  max(analysis_years),
  "\n"
)

cat(
  "Number of analysis-grid rows: ",
  length(analysis_grid_idx),
  "\n"
)


# =========================================================
# POSTERIOR SUMMARY HELPER
# =========================================================
#
# This always returns:
#
#   mean
#   lower 95% credible limit
#   upper 95% credible limit
#
# directly from posterior draws.
# =========================================================

posterior_summary <- function(draws) {

  draws <- as.numeric(draws)
  draws <- draws[is.finite(draws)]

  if (length(draws) == 0) {
    return(
      tibble(
        mean = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_
      )
    )
  }

  tibble(
    mean = mean(draws),
    CI_low = unname(
      quantile(
        draws,
        probs = CI_LOW_PROB,
        names = FALSE
      )
    ),
    CI_high = unname(
      quantile(
        draws,
        probs = CI_HIGH_PROB,
        names = FALSE
      )
    )
  )
}


# =========================================================
# POSTERIOR GRID SUMMARY HELPER
# =========================================================

posterior_grid_summary <- function(
  posterior_matrix,
  grid_df
) {

  if (ncol(posterior_matrix) != nrow(grid_df)) {
    stop(
      "Posterior matrix columns do not match grid rows."
    )
  }

  tibble(
    SiteName = grid_df$SiteName,
    Site_clean = grid_df$Site_clean,
    Age = grid_df$Age,
    pred = apply(
      posterior_matrix,
      2,
      function(x) mean(x, na.rm = TRUE)
    ),
    CI_low = apply(
      posterior_matrix,
      2,
      function(x)
        unname(
          quantile(
            x,
            probs = CI_LOW_PROB,
            na.rm = TRUE
          )
        )
    ),
    CI_high = apply(
      posterior_matrix,
      2,
      function(x)
        unname(
          quantile(
            x,
            probs = CI_HIGH_PROB,
            na.rm = TRUE
          )
        )
    )
  )
}


# =========================================================
# CALCULATE ESTIMATED SLOPE FOR EACH SITE
# =========================================================
#
# This is retained as the diagnostic comparison with GIA.
# Because pred is RSL in metres and Age is years in the
# output dataframe, the fitted slope is converted to mm/yr.
#
# This diagnostic is NOT part of the Bayesian model fitting.
#
# g_h_z_x_pred contains a site-specific vertical offset plus
# the linear local component. The offset only affects the
# intercept, so the fitted slope remains the diagnostic local
# linear trend.
# =========================================================

slopes <-
  lin_loc_component_df %>%
  group_by(SiteName) %>%
  summarise(
    slope = coef(
      lm(pred ~ Age)
    )[["Age"]],
    .groups = "drop"
  )


# =========================================================
# PREPARE GIA TABLE
# =========================================================

gia_tbl <- enframe(
  GIA,
  name = "location",
  value = "prior"
)


# =========================================================
# MATCH SITE NAMES
# =========================================================

slopes_joined <- slopes %>%
  mutate(
    location = str_trim(
      str_remove(
        SiteName,
        ",[\\s\\S]*"
      )
    )
  ) %>%
  left_join(
    gia_tbl,
    by = "location"
  ) %>%
  select(-SiteName)


# =========================================================
# CHECK GIA MATCHING
# =========================================================

if (any(is.na(slopes_joined$prior))) {

  warning(
    "Some estimated slope sites did not match a GIA prior."
  )

  print(
    slopes_joined %>%
      filter(is.na(prior))
  )
}


# =========================================================
# GIA PRIOR VS ESTIMATED SLOPE PLOT
# =========================================================

gia_plot <-
  ggplot(
    slopes_joined,
    aes(
      x = prior,
      y = slope * 1000,
      colour = location
    )
  ) +
  geom_point(size = 4) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    x = "GIA prior (mm/yr)",
    y = "Estimated linear local slope (mm/yr)",
    colour = "Site"
  ) +
  theme_bw(base_size = 14) +
  coord_equal()

print(gia_plot)

ggsave(
  filename = file.path(
    plot_dir,
    "Figure_5_GIA_prior_vs_estimated_slope.png"
  ),
  plot = gia_plot,
  width = 7,
  height = 6,
  dpi = 600
)


# =========================================================
# POSTERIOR RSL SUMMARIES
# =========================================================

total_rsl_grid <- posterior_grid_summary(
  mu,
  grid
)

regional_rsl_grid <- posterior_grid_summary(
  r_regional,
  grid
)

linear_rsl_grid <- posterior_grid_summary(
  lin_loc_post,
  grid
)

nonlinear_rsl_grid <- posterior_grid_summary(
  non_lin_loc_post,
  grid
)


# =========================================================
# TOTAL MODEL + COMPONENTS PLOT
# =========================================================
#
# All ribbons here are calculated directly from posterior
# samples. No reslr dataframe lwr/upr fields are used.
# =========================================================

component_lines <- bind_rows(
  regional_rsl_grid %>%
    select(
      SiteName,
      Site_clean,
      Age,
      pred
    ) %>%
    mutate(
      Component = "Regional"
    ),

  linear_rsl_grid %>%
    select(
      SiteName,
      Site_clean,
      Age,
      pred
    ) %>%
    mutate(
      Component = "Linear local"
    ),

  nonlinear_rsl_grid %>%
    select(
      SiteName,
      Site_clean,
      Age,
      pred
    ) %>%
    mutate(
      Component = "Non-linear local"
    )
)


combined_plot <-
  ggplot() +

  geom_ribbon(
    data = total_rsl_grid,
    aes(
      x = Age,
      ymin = CI_low,
      ymax = CI_high
    ),
    fill = "#7B68EE",
    alpha = 0.25
  ) +

  geom_line(
    data = total_rsl_grid,
    aes(
      x = Age,
      y = pred,
      colour = "Mean modelled RSL"
    ),
    linewidth = 1.0
  ) +

  geom_line(
    data = subset(
      component_lines,
      Component == "Regional"
    ),
    aes(
      x = Age,
      y = pred,
      colour = Component
    ),
    linewidth = 0.6
  ) +

  geom_line(
    data = subset(
      component_lines,
      Component == "Linear local"
    ),
    aes(
      x = Age,
      y = pred,
      colour = Component
    ),
    linewidth = 0.8
  ) +

  geom_line(
    data = subset(
      component_lines,
      Component == "Non-linear local"
    ),
    aes(
      x = Age,
      y = pred,
      colour = Component
    ),
    linewidth = 0.8
  ) +

  facet_wrap(
    ~SiteName,
    ncol = 2,
    scales = "free_y"
  ) +

  scale_colour_manual(
    name = "Model component",
    values = c(
      "Mean modelled RSL" = "#0072B2",
      "Regional" = "black",
      "Linear local" = "#009E73",
      "Non-linear local" = "#D55E00"
    ),
    breaks = c(
      "Mean modelled RSL",
      "Regional",
      "Linear local",
      "Non-linear local"
    )
  ) +

  labs(
    x = "Year",
    y = "Relative sea-level (m)",
    colour = "Model component",
    title = paste0(
      "Mean modelled RSL and model components (",
      CI_LEVEL * 100,
      "% credible interval)"
    )
  ) +

  theme_bw(base_size = 13)

print(combined_plot)

ggsave(
  filename = file.path(
    plot_dir,
    "Figure_8_RESLR_mean_modelled_RSL_with_components_and_CI.png"
  ),
  plot = combined_plot,
  width = 12,
  height = 8,
  dpi = 600
)



# =========================================================
# 1. CHECK RATE POSTERIORS
# =========================================================

if (is.null(mu_rate)) {
  stop(
    "post$mu_pred_deriv was not found in the model output."
  )
}

if (is.null(r_regional_rate)) {
  stop(
    "post$r_pred_deriv was not found in the model output."
  )
}

if (is.null(lin_loc_rate_post)) {
  stop(
    "post$g_z_x_pred_deriv was not found in the model output."
  )
}

if (is.null(non_lin_loc_rate)) {
  stop(
    "post$l_pred_deriv was not found in the model output."
  )
}


# =========================================================
# 2. SITE TOTAL MEAN RATES
# =========================================================
#
# For each posterior draw:
#
#   average all prediction years for one site
#
# within the explicit analysis window.
#
# Then calculate the posterior mean and 95% credible interval
# from those draw-level site means.
# =========================================================

site_rates <-
  lapply(
    names(site_index),
    function(s) {

      idx <- site_index[[s]]

      idx <- idx[
        grid$Age[idx] >= MODEL_START &
          grid$Age[idx] <= MODEL_END
      ]

      if (length(idx) == 0) {
        return(
          tibble(
            Site = s,
            mean_rate = NA_real_,
            CI_low = NA_real_,
            CI_high = NA_real_,
            n_years = 0
          )
        )
      }

      draws <-
        rowMeans(
          mu_rate[
            ,
            idx,
            drop = FALSE
          ],
          na.rm = TRUE
        )

      summary <- posterior_summary(draws)

      tibble(
        Site = s,
        mean_rate = summary$mean,
        CI_low = summary$CI_low,
        CI_high = summary$CI_high,
        n_years = length(idx)
      )
    }
  ) %>%
  bind_rows()


# =========================================================
# PRINT SITE TOTAL MEAN RATES
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("SITE TOTAL MEAN RATES\n")
cat("=========================================================\n")

print(site_rates)


# =========================================================
# SAVE SITE TOTAL MEAN RATES
# =========================================================

write.csv(
  site_rates,
  file.path(
    plot_dir,
    "extended_grid_total_mean_site_rates_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 3. YEARLY TOTAL RATES PER SITE
# =========================================================
#
# One row per site-year.
# Every CI is calculated directly across posterior draws.
# =========================================================

yearly_site_rates <-
  lapply(
    names(site_index),
    function(s) {

      idx <- site_index[[s]]

      idx <- idx[
        grid$Age[idx] >= MODEL_START &
          grid$Age[idx] <= MODEL_END
      ]

      if (length(idx) == 0) {
        return(NULL)
      }

      site_mat <-
        mu_rate[
          ,
          idx,
          drop = FALSE
        ]

      tibble(
        Site = s,
        Year = grid$Age[idx],
        mean_rate = colMeans(
          site_mat,
          na.rm = TRUE
        ),
        CI_low = apply(
          site_mat,
          2,
          function(x)
            unname(
              quantile(
                x,
                CI_LOW_PROB,
                na.rm = TRUE
              )
            )
        ),
        CI_high = apply(
          site_mat,
          2,
          function(x)
            unname(
              quantile(
                x,
                CI_HIGH_PROB,
                na.rm = TRUE
              )
            )
        )
      )
    }
  ) %>%
  bind_rows() %>%
  arrange(
    Site,
    Year
  )


# =========================================================
# PRINT YEARLY RATES
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("YEARLY SITE TOTAL RATES\n")
cat("=========================================================\n")

print(yearly_site_rates)


# =========================================================
# SAVE YEARLY RATES
# =========================================================

write.csv(
  yearly_site_rates,
  file.path(
    plot_dir,
    "extended_grid_total_yearly_site_rates_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 4. REGIONAL MEAN RATE
# =========================================================
#
# Regional posterior rate field:
#
#   r_pred_deriv
#
# Restrict to the analysis window first.
#
# The regional component is a temporal regional signal and is
# represented on the prediction grid. The draw-level mean is
# therefore calculated across the regional prediction locations
# in the requested analysis window.
# =========================================================

regional_analysis_mat <-
  r_regional_rate[
    ,
    analysis_grid_idx,
    drop = FALSE
  ]

regional_draws <-
  rowMeans(
    regional_analysis_mat,
    na.rm = TRUE
  )

regional_summary <- posterior_summary(
  regional_draws
)

regional_rate <-
  tibble(
    mean_rate = regional_summary$mean,
    CI_low = regional_summary$CI_low,
    CI_high = regional_summary$CI_high
  )


# =========================================================
# PRINT REGIONAL RATE
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("REGIONAL POSTERIOR RATE\n")
cat("=========================================================\n")

print(regional_rate)


# =========================================================
# SAVE REGIONAL RATE
# =========================================================

write.csv(
  regional_rate,
  file.path(
    plot_dir,
    "regional_mean_rate_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 5. CENTURY MEAN TOTAL RATE -- ALL SITES
# =========================================================
#
# For every posterior draw:
#
#   average across ALL prediction locations and years
#   within 1925-2024.
#
# This is the total posterior rate from mu_pred_deriv.
# =========================================================

total_analysis_mat <-
  mu_rate[
    ,
    analysis_grid_idx,
    drop = FALSE
  ]

total_regional_draws <-
  rowMeans(
    total_analysis_mat,
    na.rm = TRUE
  )

total_summary <- posterior_summary(
  total_regional_draws
)

total_regional_rate <-
  tibble(
    mean_rate = total_summary$mean,
    CI_low = total_summary$CI_low,
    CI_high = total_summary$CI_high
  )


# =========================================================
# PRINT TOTAL RATE
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("TOTAL POSTERIOR RATE\n")
cat("=========================================================\n")

print(total_regional_rate)


# =========================================================
# SAVE TOTAL RATE
# =========================================================

write.csv(
  total_regional_rate,
  file.path(
    plot_dir,
    "total_mean_rate_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 6. PERIOD DEFINITIONS
# =========================================================

periods <-
  tibble(
    Period = c(
      "1925-2024",
      "1925-2000",
      "1925-1975",
      "1975-2024",
      "2000-2024",
      "2024"
    ),
    Start = c(
      1925,
      1925,
      1925,
      1975,
      2000,
      2024
    ),
    End = c(
      2024,
      2000,
      1975,
      2024,
      2024,
      2024
    ),
    Instantaneous = c(
      FALSE,
      FALSE,
      FALSE,
      FALSE,
      FALSE,
      TRUE
    )
  )


# =========================================================
# 7. PERIOD DRAW HELPER
# =========================================================
#
# IMPORTANT:
#
# Period means are calculated within EACH posterior draw first.
# The posterior CI is then taken across those draw-level period
# means. This preserves posterior uncertainty.
# =========================================================

get_period_draws <- function(
  posterior_matrix,
  ages,
  start_year,
  end_year,
  instantaneous = FALSE
) {

  idx <- which(
    ages >= start_year &
      ages <= end_year
  )

  if (length(idx) == 0) {
    stop(
      paste(
        "No prediction years found for period:",
        start_year,
        "-",
        end_year
      )
    )
  }

  if (instantaneous) {

    # For an instantaneous rate such as 2024, there may be
    # multiple site prediction locations at the same year.
    #
    # First average those prediction locations within each
    # posterior draw. This preserves posterior covariance.

    rowMeans(
      posterior_matrix[
        ,
        idx,
        drop = FALSE
      ],
      na.rm = TRUE
    )

  } else {

    # For a period mean, average all prediction years/locations
    # within each posterior draw first.

    rowMeans(
      posterior_matrix[
        ,
        idx,
        drop = FALSE
      ],
      na.rm = TRUE
    )
  }
}


# =========================================================
# 8. REGIONAL PERIOD RATES
# =========================================================

regional_period_rates <-
  lapply(
    seq_len(nrow(periods)),
    function(i) {

      draws <- get_period_draws(
        posterior_matrix = r_regional_rate,
        ages = grid$Age,
        start_year = periods$Start[i],
        end_year = periods$End[i],
        instantaneous = periods$Instantaneous[i]
      )

      summary <- posterior_summary(draws)

      tibble(
        Component = "Regional",
        Period = periods$Period[i],
        mean_rate = summary$mean,
        CI_low = summary$CI_low,
        CI_high = summary$CI_high
      )
    }
  ) %>%
  bind_rows()


# =========================================================
# PRINT REGIONAL PERIOD RATES
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("REGIONAL PERIOD RATES\n")
cat("=========================================================\n")

print(regional_period_rates)


# =========================================================
# 9. TOTAL MODEL PERIOD RATES
# =========================================================

total_period_rates <-
  lapply(
    seq_len(nrow(periods)),
    function(i) {

      draws <- get_period_draws(
        posterior_matrix = mu_rate,
        ages = grid$Age,
        start_year = periods$Start[i],
        end_year = periods$End[i],
        instantaneous = periods$Instantaneous[i]
      )

      summary <- posterior_summary(draws)

      tibble(
        Component = "Total (Regional + Local + GIA)",
        Period = periods$Period[i],
        mean_rate = summary$mean,
        CI_low = summary$CI_low,
        CI_high = summary$CI_high
      )
    }
  ) %>%
  bind_rows()


# =========================================================
# PRINT TOTAL PERIOD RATES
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("TOTAL PERIOD RATES\n")
cat("=========================================================\n")

print(total_period_rates)


# =========================================================
# 10. COMBINE REGIONAL + TOTAL PERIOD OUTPUTS
# =========================================================

period_rate_summary <-
  bind_rows(
    regional_period_rates,
    total_period_rates
  )


# =========================================================
# PRINT COMBINED PERIOD SUMMARY
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("REGIONAL + TOTAL PERIOD SUMMARY\n")
cat("=========================================================\n")

print(period_rate_summary)


# =========================================================
# SAVE PERIOD SUMMARY
# =========================================================

write.csv(
  period_rate_summary,
  file.path(
    plot_dir,
    "regional_total_period_rate_summary.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 11. COMPONENT DATA FOR RATE PLOTS
# =========================================================
#
# IMPORTANT:
#
# The linear local component rate MUST use
# g_z_x_pred_deriv.
#
# The previous version incorrectly passed g_h_z_x_pred into
# posterior_grid_summary() and thereby labelled an RSL
# component as a rate.
# =========================================================

linear_rate_grid <- posterior_grid_summary(
  lin_loc_rate_post,
  grid
)

nonlinear_rate_grid <- posterior_grid_summary(
  non_lin_loc_rate,
  grid
)

regional_rate_grid <- posterior_grid_summary(
  r_regional_rate,
  grid
)

total_rate_grid <- posterior_grid_summary(
  mu_rate,
  grid
)


# =========================================================
# ORIGINAL RATE PLOT
# =========================================================
#
# Uses the posterior total rate field and posterior-derived
# 95% credible intervals.
# =========================================================

plot_result <-
  ggplot2::ggplot(
    total_rate_grid
  ) +
  ggplot2::geom_line(
    ggplot2::aes(
      x = Age,
      y = pred,
      colour = SiteName
    )
  ) +
  ggplot2::geom_ribbon(
    ggplot2::aes(
      x = Age,
      ymin = CI_low,
      ymax = CI_high,
      fill = SiteName
    ),
    alpha = 0.3
  ) +
  ggplot2::coord_cartesian(
    xlim = c(
      MODEL_START,
      MODEL_END
    )
  )

print(plot_result)

ggsave(
  filename = file.path(
    plot_dir,
    "mu_tot_pred_rate.pdf"
  ),
  plot = plot_result,
  width = 10,
  height = 7
)


# =========================================================
# 12. REGIONAL COMPONENT RATE PLOT
# =========================================================

p <-

  ggplot2::ggplot(
    regional_rate_grid %>%
      filter(
        Age >= MODEL_START,
        Age <= MODEL_END
      )
  ) +

  ggplot2::geom_ribbon(
    ggplot2::aes(
      x = Age,
      ymin = CI_low,
      ymax = CI_high,
      fill = "95% Credible Interval"
    ),
    alpha = 0.3,
    colour = NA
  ) +

  ggplot2::geom_line(
    ggplot2::aes(
      x = Age,
      y = pred,
      colour = "Posterior Fit"
    ),
    linewidth = 1
  ) +

  ggplot2::scale_colour_manual(
    name = "",
    values = c(
      "Posterior Fit" = "#4C6EF5"
    )
  ) +

  ggplot2::scale_fill_manual(
    name = "",
    values = c(
      "95% Credible Interval" = "#7B68EE"
    )
  ) +

  ggplot2::labs(
    x = "Year",
    y = expression(
      "Rate of relative sea-level change (mm " *
        yr^{-1} *
        ")"
    ),
    title = "Regional component rate through time"
  ) +

  ggplot2::theme_bw() +

  ggplot2::theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

print(p)


# =========================================================
# SAVE REGIONAL COMPONENT
# =========================================================

ggplot2::ggsave(
  filename = file.path(
    plot_dir,
    "regional_component.png"
  ),
  plot = p,
  width = 6,
  height = 6,
  dpi = 300,
  device = "png"
)


# =========================================================
# 13. REGIONAL RATE PLOT
# =========================================================

p_regional_rate <-
  ggplot(
    regional_rate_grid %>%
      filter(
        Age >= MODEL_START,
        Age <= MODEL_END
      ),
    aes(
      Age,
      pred
    )
  ) +

  geom_ribbon(
    aes(
      ymin = CI_low,
      ymax = CI_high
    ),
    fill = "#7B68EE",
    alpha = 0.3
  ) +

  geom_line(
    colour = "#4C6EF5",
    linewidth = 1
  ) +

  labs(
    title = "Regional rate",
    x = "Year",
    y = "Rate (mm/yr)"
  ) +

  theme_bw()

ggsave(
  filename = file.path(
    plot_dir,
    "03_regional_rate.png"
  ),
  plot = p_regional_rate,
  width = 10,
  height = 6,
  dpi = 300
)


# =========================================================
# 14. TOTAL MODEL -- TRUE POSTERIOR ONLY
# =========================================================
#
# This section deliberately does NOT use total_model_df's
# rate_lwr/rate_upr fields because those fields are reversed
# by RESLR's internal output naming.
# =========================================================

total_df <- total_rate_grid %>%
  select(
    SiteName,
    Age,
    rate_pred = pred,
    rate_lwr = CI_low,
    rate_upr = CI_high
  )


# =========================================================
# 15. CREATE FIGURE 9 DATA FROM TOTAL POSTERIOR RATE
# =========================================================
#
# Figure 9 is calculated directly from mu_pred_deriv.
# The CI is therefore a genuine posterior credible interval.
# =========================================================

figure9_rates <-
  lapply(
    names(site_index),
    function(s) {

      idx <- site_index[[s]]

      idx <- idx[
        grid$Age[idx] >= MODEL_START &
          grid$Age[idx] <= MODEL_END
      ]

      if (length(idx) == 0) {
        return(NULL)
      }

      site_mat <- mu_rate[
        ,
        idx,
        drop = FALSE
      ]

      tibble(
        SiteName = s,
        Age = grid$Age[idx],
        rate_pred = colMeans(
          site_mat,
          na.rm = TRUE
        ),
        rate_lwr = apply(
          site_mat,
          2,
          function(x)
            unname(
              quantile(
                x,
                CI_LOW_PROB,
                na.rm = TRUE
              )
            )
        ),
        rate_upr = apply(
          site_mat,
          2,
          function(x)
            unname(
              quantile(
                x,
                CI_HIGH_PROB,
                na.rm = TRUE
              )
            )
        )
      )
    }
  ) %>%
  bind_rows()


# =========================================================
# CHECK FIGURE 9 CREDIBLE INTERVALS
# =========================================================

if (
  any(
    figure9_rates$rate_lwr >
      figure9_rates$rate_pred,
    na.rm = TRUE
  )
) {
  stop(
    "Figure 9 contains a lower credible limit above the posterior mean."
  )
}

if (
  any(
    figure9_rates$rate_pred >
      figure9_rates$rate_upr,
    na.rm = TRUE
  )
) {
  stop(
    "Figure 9 contains a posterior mean above the upper credible limit."
  )
}


# =========================================================
# 16. PLOT AND SAVE FIGURE 9
# =========================================================

p_total <-
  ggplot(
    figure9_rates,
    aes(
      x = Age
    )
  ) +

  geom_ribbon(
    aes(
      ymin = rate_lwr,
      ymax = rate_upr,
      fill = "95% Credible Interval"
    ),
    alpha = 0.3,
    colour = NA
  ) +

  geom_line(
    aes(
      y = rate_pred,
      colour = "Posterior Fit"
    ),
    linewidth = 1
  ) +

  facet_wrap(
    ~SiteName,
    ncol = 2
  ) +

  scale_colour_manual(
    name = "",
    values = c(
      "Posterior Fit" = "#4C6EF5"
    )
  ) +

  scale_fill_manual(
    name = "",
    values = c(
      "95% Credible Interval" = "#7B68EE"
    )
  ) +

  labs(
    x = "Year",
    y = expression(
      "Rate of relative sea-level change (mm " *
        yr^{-1} *
        ")"
    ),
    title = paste0(
      "Total Model Posterior Predictions (",
      MODEL_START,
      "-",
      MODEL_END,
      ")"
    )
  ) +

  theme_bw() +

  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text = element_text(
      face = "bold"
    )
  )

print(p_total)


ggsave(
  filename = file.path(
    plot_dir,
    "Figure_9_total_rate_posterior.png"
  ),
  plot = p_total,
  width = 12,
  height = 8,
  dpi = 300
)


# =========================================================
# 17. SAVE TOTAL MODEL PLOTS
# =========================================================

ggsave(
  filename = file.path(
    plot_dir,
    "total_model_plot.png"
  ),
  plot = p_total,
  width = 10,
  height = 6,
  dpi = 300
)


# =========================================================
# 18. SITE-AVERAGED TOTAL RATE
# =========================================================
#
# For each year and posterior draw:
#
#   average the posterior total rate across sites
#
# and only then calculate the posterior mean and credible
# interval across those draw-level site averages.
#
# The calculation below explicitly performs the site average
# within each posterior draw and each year.
# =========================================================

years_for_site_average <-
  sort(
    unique(
      grid$Age[
        grid$Age >= MODEL_START &
          grid$Age <= MODEL_END
      ]
    )
  )

total_rate_single_draws <- matrix(
  NA_real_,
  nrow = nrow(mu_rate),
  ncol = length(years_for_site_average)
)

colnames(total_rate_single_draws) <-
  as.character(years_for_site_average)


# =========================================================
# SITE-AVERAGED POSTERIOR RATE BY YEAR
# =========================================================

for (j in seq_along(years_for_site_average)) {

  year_j <- years_for_site_average[j]

  year_site_draws <- lapply(
    names(site_index),
    function(s) {

      idx_site <- site_index[[s]]

      idx_year <- idx_site[
        grid$Age[idx_site] == year_j
      ]

      if (length(idx_year) == 0) {
        return(NULL)
      }

      rowMeans(
        mu_rate[
          ,
          idx_year,
          drop = FALSE
        ],
        na.rm = TRUE
      )
    }
  )

  year_site_draws <- Filter(
    Negate(is.null),
    year_site_draws
  )

  if (length(year_site_draws) == 0) {
    next
  }

  year_site_matrix <- do.call(
    cbind,
    year_site_draws
  )

  total_rate_single_draws[, j] <-
    rowMeans(
      year_site_matrix,
      na.rm = TRUE
    )
}


# =========================================================
# SITE-AVERAGED TOTAL RATE SUMMARY
# =========================================================

total_rate_single <-
  tibble(
    Age = years_for_site_average,
    rate_lwr = apply(
      total_rate_single_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_LOW_PROB,
            na.rm = TRUE
          )
        )
    ),
    rate_pred = colMeans(
      total_rate_single_draws,
      na.rm = TRUE
    ),
    rate_upr = apply(
      total_rate_single_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_HIGH_PROB,
            na.rm = TRUE
          )
        )
    )
  )


# =========================================================
# CHECK SITE-AVERAGED TOTAL RATE
# =========================================================

total_rate_single_check <-
  total_rate_single %>%
  summarise(
    n_lower_above_mean = sum(
      rate_lwr > rate_pred,
      na.rm = TRUE
    ),
    n_mean_above_upper = sum(
      rate_pred > rate_upr,
      na.rm = TRUE
    ),
    n_lower_above_upper = sum(
      rate_lwr > rate_upr,
      na.rm = TRUE
    )
  )

cat("\n")
cat("=========================================================\n")
cat("SITE-AVERAGED TOTAL RATE INTERVAL CHECK\n")
cat("=========================================================\n")

print(total_rate_single_check)

stopifnot(
  all(
    total_rate_single$rate_lwr <=
      total_rate_single$rate_pred,
    na.rm = TRUE
  )
)

stopifnot(
  all(
    total_rate_single$rate_pred <=
      total_rate_single$rate_upr,
    na.rm = TRUE
  )
)


# =========================================================
# PRINT FIRST 20 ROWS
# =========================================================

print(
  total_rate_single %>%
    select(
      Age,
      rate_lwr,
      rate_pred,
      rate_upr
    ) %>%
    head(20)
)


# =========================================================
# SAVE SITE-AVERAGED RATE TABLE
# =========================================================

write.csv(
  total_rate_single,
  file.path(
    plot_dir,
    "site_averaged_total_rate_posterior_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# 19. PLOT SITE-AVERAGED TOTAL RATE
# =========================================================

p_total_single <-
  ggplot2::ggplot(
    total_rate_single,
    ggplot2::aes(
      x = Age,
      y = rate_pred
    )
  ) +

  ggplot2::geom_ribbon(
    ggplot2::aes(
      ymin = rate_lwr,
      ymax = rate_upr,
      fill = "95% Credible Interval"
    ),
    alpha = 0.3,
    colour = NA
  ) +

  ggplot2::geom_line(
    ggplot2::aes(
      colour = "Posterior fit"
    ),
    linewidth = 1
  ) +

  ggplot2::scale_colour_manual(
    name = "",
    values = c(
      "Posterior fit" = "#4C6EF5"
    )
  ) +

  ggplot2::scale_fill_manual(
    name = "",
    values = c(
      "95% Credible Interval" = "#7B68EE"
    )
  ) +

  ggplot2::labs(
    title = "Site-averaged total modelled RSL rate",
    x = "Year",
    y = expression(
      "Rate of relative sea-level change (mm " *
        yr^{-1} *
        ")"
    )
  ) +

  ggplot2::theme_bw() +

  ggplot2::theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

print(p_total_single)


# =========================================================
# SAVE SITE-AVERAGED TOTAL RATE
# =========================================================

ggplot2::ggsave(
  filename = file.path(
    plot_dir,
    "total_single_rate.png"
  ),
  plot = p_total_single,
  width = 10,
  height = 6,
  dpi = 300
)


# =========================================================
# 20. REGIONAL + MEAN MODELLED RSL RATE
# =========================================================
#
# CRITICAL FIX:
#
# Both regional and total uncertainty are calculated from
# posterior draws.
#
# Regional and total values are calculated at each year by
# averaging prediction locations within each posterior draw.
# =========================================================

regional_total_draws <- matrix(
  NA_real_,
  nrow = nrow(mu_rate),
  ncol = length(years_for_site_average)
)

regional_total_site_draws <- matrix(
  NA_real_,
  nrow = nrow(mu_rate),
  ncol = length(years_for_site_average)
)

colnames(regional_total_draws) <-
  as.character(years_for_site_average)

colnames(regional_total_site_draws) <-
  as.character(years_for_site_average)


# =========================================================
# REGIONAL + TOTAL POSTERIOR DRAW CALCULATIONS
# =========================================================

for (j in seq_along(years_for_site_average)) {

  year_j <- years_for_site_average[j]

  idx_year <- which(
    grid$Age == year_j
  )

  if (length(idx_year) == 0) {
    next
  }

  regional_total_draws[, j] <-
    rowMeans(
      r_regional_rate[
        ,
        idx_year,
        drop = FALSE
      ],
      na.rm = TRUE
    )

  # Explicitly average total rate across sites first.
  # This prevents sites with different numbers of prediction
  # rows from receiving unequal implicit weights.

  year_site_draws <- lapply(
    names(site_index),
    function(s) {

      idx_site <- site_index[[s]]

      idx_site_year <- idx_site[
        grid$Age[idx_site] == year_j
      ]

      if (length(idx_site_year) == 0) {
        return(NULL)
      }

      rowMeans(
        mu_rate[
          ,
          idx_site_year,
          drop = FALSE
        ],
        na.rm = TRUE
      )
    }
  )

  year_site_draws <- Filter(
    Negate(is.null),
    year_site_draws
  )

  if (length(year_site_draws) > 0) {

    year_site_matrix <- do.call(
      cbind,
      year_site_draws
    )

    regional_total_site_draws[, j] <-
      rowMeans(
        year_site_matrix,
        na.rm = TRUE
      )
  }
}


# =========================================================
# 21. REGIONAL + TOTAL SUMMARY
# =========================================================

regional_total_df <-
  tibble(
    Age = years_for_site_average,

    regional_lwr = apply(
      regional_total_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_LOW_PROB,
            na.rm = TRUE
          )
        )
    ),

    regional_pred = colMeans(
      regional_total_draws,
      na.rm = TRUE
    ),

    regional_upr = apply(
      regional_total_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_HIGH_PROB,
            na.rm = TRUE
          )
        )
    ),

    total_lwr = apply(
      regional_total_site_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_LOW_PROB,
            na.rm = TRUE
          )
        )
    ),

    total_pred = colMeans(
      regional_total_site_draws,
      na.rm = TRUE
    ),

    total_upr = apply(
      regional_total_site_draws,
      2,
      function(x)
        unname(
          quantile(
            x,
            CI_HIGH_PROB,
            na.rm = TRUE
          )
        )
    )
  )


# =========================================================
# 22. CHECK REGIONAL + TOTAL INTERVALS
# =========================================================

regional_total_interval_check <-
  regional_total_df %>%
  summarise(
    regional_bad_lower = sum(
      regional_lwr > regional_pred,
      na.rm = TRUE
    ),
    regional_bad_upper = sum(
      regional_pred > regional_upr,
      na.rm = TRUE
    ),
    total_bad_lower = sum(
      total_lwr > total_pred,
      na.rm = TRUE
    ),
    total_bad_upper = sum(
      total_pred > total_upr,
      na.rm = TRUE
    )
  )

cat("\n")
cat("=========================================================\n")
cat("REGIONAL + TOTAL INTERVAL CHECK\n")
cat("=========================================================\n")

print(regional_total_interval_check)

stopifnot(
  all(
    regional_total_df$regional_lwr <=
      regional_total_df$regional_pred,
    na.rm = TRUE
  )
)

stopifnot(
  all(
    regional_total_df$regional_pred <=
      regional_total_df$regional_upr,
    na.rm = TRUE
  )
)

stopifnot(
  all(
    regional_total_df$total_lwr <=
      regional_total_df$total_pred,
    na.rm = TRUE
  )
)

stopifnot(
  all(
    regional_total_df$total_pred <=
      regional_total_df$total_upr,
    na.rm = TRUE
  )
)


# =========================================================
# 23. PLOT REGIONAL + TOTAL
# =========================================================

p_regional_total <-
  ggplot2::ggplot(
    regional_total_df
  ) +

  ggplot2::geom_ribbon(
    ggplot2::aes(
      x = Age,
      ymin = regional_lwr,
      ymax = regional_upr,
      fill = "Regional 95% Credible Interval"
    ),
    alpha = 0.25,
    colour = NA
  ) +

  ggplot2::geom_ribbon(
    ggplot2::aes(
      x = Age,
      ymin = total_lwr,
      ymax = total_upr,
      fill = "Mean modelled RSL 95% Credible Interval"
    ),
    alpha = 0.25,
    colour = NA
  ) +

  ggplot2::geom_line(
    ggplot2::aes(
      x = Age,
      y = regional_pred,
      colour = "Regional component"
    ),
    linewidth = 1
  ) +

  ggplot2::geom_line(
    ggplot2::aes(
      x = Age,
      y = total_pred,
      colour = "Mean modelled RSL"
    ),
    linewidth = 1
  ) +

  ggplot2::scale_colour_manual(
    name = NULL,
    values = c(
      "Regional component" = "#4C6EF5",
      "Mean modelled RSL" = "black"
    )
  ) +

  ggplot2::scale_fill_manual(
    name = NULL,
    values = c(
      "Regional 95% Credible Interval" = "#7B68EE",
      "Mean modelled RSL 95% Credible Interval" = "grey70"
    )
  ) +

  ggplot2::labs(
    x = "Year",
    y = expression(
      "Rate of relative sea-level change (mm yr"^{-1} * ")"
    )
  ) +

  ggplot2::theme_bw(
    base_size = 12
  ) +

  ggplot2::theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "vertical",
    panel.grid.minor =
      ggplot2::element_blank(),
    axis.title =
      ggplot2::element_text(
        size = 12
      ),
    axis.text =
      ggplot2::element_text(
        size = 11
      )
  )

print(p_regional_total)


# =========================================================
# 24. SAVE PUBLICATION FIGURE
# =========================================================

ggplot2::ggsave(
  filename = file.path(
    plot_dir,
    "Figure_10_regional_vs_mean_modelled_RSL_rate_publication.png"
  ),
  plot = p_regional_total,
  width = 8,
  height = 6,
  dpi = 600
)


# =========================================================
# 25. SAVE IMPORTANT POSTERIOR TABLES
# =========================================================

write.csv(
  regional_period_rates,
  file.path(
    plot_dir,
    "regional_period_rates.csv"
  ),
  row.names = FALSE
)

write.csv(
  total_period_rates,
  file.path(
    plot_dir,
    "total_period_rates.csv"
  ),
  row.names = FALSE
)

write.csv(
  period_rate_summary,
  file.path(
    plot_dir,
    "regional_total_period_rate_summary.csv"
  ),
  row.names = FALSE
)


# =========================================================
# ADDITIONAL POSTERIOR GRID TABLES
# =========================================================

write.csv(
  total_rate_grid %>%
    filter(
      Age >= MODEL_START,
      Age <= MODEL_END
    ) %>%
    select(
      SiteName,
      Site_clean,
      Age,
      rate_pred = pred,
      rate_lwr = CI_low,
      rate_upr = CI_high
    ),
  file.path(
    plot_dir,
    "posterior_total_rate_grid_CI.csv"
  ),
  row.names = FALSE
)

write.csv(
  regional_rate_grid %>%
    filter(
      Age >= MODEL_START,
      Age <= MODEL_END
    ) %>%
    select(
      SiteName,
      Site_clean,
      Age,
      rate_pred = pred,
      rate_lwr = CI_low,
      rate_upr = CI_high
    ),
  file.path(
    plot_dir,
    "posterior_regional_rate_grid_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# LINEAR LOCAL RATE POSTERIOR GRID TABLE
# =========================================================

write.csv(
  linear_rate_grid %>%
    filter(
      Age >= MODEL_START,
      Age <= MODEL_END
    ) %>%
    select(
      SiteName,
      Site_clean,
      Age,
      rate_pred = pred,
      rate_lwr = CI_low,
      rate_upr = CI_high
    ),
  file.path(
    plot_dir,
    "posterior_linear_local_rate_grid_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# NON-LINEAR LOCAL RATE POSTERIOR GRID TABLE
# =========================================================

write.csv(
  nonlinear_rate_grid %>%
    filter(
      Age >= MODEL_START,
      Age <= MODEL_END
    ) %>%
    select(
      SiteName,
      Site_clean,
      Age,
      rate_pred = pred,
      rate_lwr = CI_low,
      rate_upr = CI_high
    ),
  file.path(
    plot_dir,
    "posterior_nonlinear_local_rate_grid_CI.csv"
  ),
  row.names = FALSE
)


# =========================================================
# FINAL SUMMARY CHECK
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("FINAL RATE ANALYSIS CHECK\n")
cat("=========================================================\n")

cat(
  "Posterior total RSL dimensions: ",
  paste(
    dim(mu),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior total rate dimensions: ",
  paste(
    dim(mu_rate),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior regional RSL dimensions: ",
  paste(
    dim(r_regional),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior regional rate dimensions: ",
  paste(
    dim(r_regional_rate),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior linear local RSL dimensions: ",
  paste(
    dim(lin_loc_post),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior linear local rate dimensions: ",
  paste(
    dim(lin_loc_rate_post),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior non-linear local RSL dimensions: ",
  paste(
    dim(non_lin_loc_post),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Posterior non-linear local rate dimensions: ",
  paste(
    dim(non_lin_loc_rate),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Prediction-grid rows: ",
  nrow(grid),
  "\n"
)

cat(
  "Number of sites: ",
  length(site_index),
  "\n"
)

cat(
  "Analysis years: ",
  min(analysis_years),
  "-",
  max(analysis_years),
  "\n"
)

cat(
  "Number of yearly site-rate rows: ",
  nrow(yearly_site_rates),
  "\n"
)

cat(
  "Number of period summaries: ",
  nrow(period_rate_summary),
  "\n"
)


# =========================================================
# SITE-AVERAGED TOTAL INTERVAL CHECK
# =========================================================

cat("\n")
cat("Site-averaged total rate interval check:\n")

print(
  total_rate_single_check
)


# =========================================================
# REGIONAL + TOTAL POSTERIOR INTERVAL CHECK
# =========================================================

cat("\n")
cat("Regional + total posterior interval check:\n")

print(
  regional_total_interval_check
)


# =========================================================
# POSTERIOR SUMMARY
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("POSTERIOR RATE SUMMARIES\n")
cat("=========================================================\n")

cat("\nRegional posterior rate:\n")

print(
  regional_rate
)

cat("\nTotal posterior rate:\n")

print(
  total_regional_rate
)

cat("\nSite mean rates:\n")

print(
  site_rates
)

cat("\nPeriod rates:\n")

print(
  period_rate_summary
)


# =========================================================
# FINAL OUTPUT LOCATION
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("OUTPUT DIRECTORY\n")
cat("=========================================================\n")

cat(
  normalizePath(
    plot_dir,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n"
)


# =========================================================
# FINAL COMPLETION MESSAGE
# =========================================================

cat("\n")
cat("=========================================================\n")
cat("RATE ANALYSIS COMPLETE\n")
cat("=========================================================\n")

cat(
  "All posterior rate calculations, credible intervals, figures, and tables have been completed.\n"
)

cat(
  "All reported rate CIs are calculated from posterior draws rather than by averaging interval endpoints.\n"
)

cat(
  "The linear local rate uses g_z_x_pred_deriv rather than g_h_z_x_pred.\n"
)

cat(
  "The analysis window is explicitly restricted to ",
  MODEL_START,
  "-",
  MODEL_END,
  ".\n",
  sep = ""
)

cat(
  "Output directory:\n"
)

cat(
  normalizePath(
    plot_dir,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n"
)
cat("=========================================================\n")


# =========================================================
# PART B: CUMULATIVE SEA-LEVEL CHANGE, 1925–2024
# =========================================================
# Calculates the cumulative change in the Ireland six-site
# mean of the total modelled RSL between 1925 and 2024.
#
# Uses:
#   mu_pred = total modelled RSL posterior
#
# Results are reported in centimetres with 95% credible
# intervals calculated directly from the posterior draws.
# ============================================================

idx_1925 <- which(grid$Age == 1925)
idx_2024 <- which(grid$Age == 2024)

stopifnot(length(idx_1925) == 6)
stopifnot(length(idx_2024) == 6)

mu_1925 <- rowMeans(
  mu[, idx_1925, drop = FALSE],
  na.rm = TRUE
)

mu_2024 <- rowMeans(
  mu[, idx_2024, drop = FALSE],
  na.rm = TRUE
)

cumulative_change_cm <- (mu_2024 - mu_1925) * 100

cumulative_change_summary <- tibble(
  start_year = 1925,
  end_year = 2024,
  total_change_cm = mean(cumulative_change_cm, na.rm = TRUE),
  CI_low_cm = quantile(cumulative_change_cm, 0.025, na.rm = TRUE),
  CI_high_cm = quantile(cumulative_change_cm, 0.975, na.rm = TRUE)
)

print(cumulative_change_summary)

# =========================================================
# PART C: IRELAND RSL VS GLOBAL SEA LEVEL
# =========================================================
# Compares the Ireland six-site mean of the total modelled
# RESLR RSL with global tide-gauge and satellite-altimetry
# sea-level observations.
#
# All series are referenced to 1993 = 0.
#
# The Ireland series is calculated from the posterior draws
# of mu_pred, with uncertainty calculated from the resulting
# six-site mean posterior distribution.
# =========================================================


# =========================================================
# 1. REFERENCE YEAR
# =========================================================

ref_year <- 1993


# =========================================================
# 2. LOAD GLOBAL DATA
# =========================================================

global_mean_sea_level <- read_csv(
  "observed_and_projected_change_in_global_mean_sea_level.csv",
  skip = 15,
  col_names = FALSE
) %>%
  rename(
    Year = X1,
    TG   = X2,
    SSH  = X3
  ) %>%
  mutate(
    Year = as.numeric(Year),
    TG   = as.numeric(TG),
    SSH  = as.numeric(SSH)
  ) %>%
  filter(
    Year >= MODEL_START,
    Year <= MODEL_END
  )


# =========================================================
# 3. GLOBAL DATA
#    mm -> m
# =========================================================

global_plot <- global_mean_sea_level %>%
  mutate(
    TG_m  = TG / 1000,
    SSH_m = SSH / 1000
  )


# =========================================================
# 4. REFERENCE GLOBAL SERIES TO 1993 = 0
# =========================================================

tg_ref <- global_plot %>%
  filter(
    Year == ref_year,
    !is.na(TG_m)
  ) %>%
  pull(TG_m)

ssh_ref <- global_plot %>%
  filter(
    Year == ref_year,
    !is.na(SSH_m)
  ) %>%
  pull(SSH_m)


stopifnot(
  length(tg_ref) == 1
)

stopifnot(
  length(ssh_ref) == 1
)


global_plot <- global_plot %>%
  mutate(
    TG_m_ref  = TG_m - tg_ref,
    SSH_m_ref = SSH_m - ssh_ref
  )


# =========================================================
# 5. SELECT TOTAL MODELLED RSL POSTERIOR
# =========================================================
#
# mu_pred = total modelled relative sea level (RSL)
#
# This is the full RESLR modelled RSL prediction at each
# site and year.
#
# The Ireland series below is constructed as the equal-weighted
# six-site mean of the mu_pred posterior within each draw.
# =========================================================

reslr_posterior <- post$mu_pred

reslr_label <-
  "Ireland six-site mean of RESLR total modelled RSL"


# =========================================================
# 6. CHECK POSTERIOR / GRID ALIGNMENT
# =========================================================

stopifnot(
  ncol(reslr_posterior) == nrow(grid)
)


# =========================================================
# 7. SELECT ANALYSIS YEARS
# =========================================================

comparison_idx <- which(
  grid$Age >= MODEL_START &
    grid$Age <= MODEL_END
)

comparison_years <- sort(
  unique(
    grid$Age[comparison_idx]
  )
)


# =========================================================
# 8. BUILD IRELAND-WIDE POSTERIOR SERIES
# =========================================================
#
# For each year:
#
#   1. Take the total modelled RSL posterior prediction
#      at the six Irish sites.
#
#   2. Average those six site predictions WITHIN EACH
#      posterior draw.
#
# This produces:
#
#     posterior draws x 100 years
#
# representing the equal-weighted six-site mean of the
# total modelled RESLR RSL.
#
# IMPORTANT:
#
# Credible intervals are calculated later from these
# resulting posterior draws.
# =========================================================

reslr_comparison_draws <- matrix(
  NA_real_,
  nrow = nrow(reslr_posterior),
  ncol = length(comparison_years)
)

colnames(reslr_comparison_draws) <-
  as.character(comparison_years)


for (j in seq_along(comparison_years)) {

  year_j <- comparison_years[j]

  year_idx <- which(
    grid$Age == year_j
  )

  stopifnot(
    length(year_idx) == 6
  )

  reslr_comparison_draws[, j] <-
    rowMeans(
      reslr_posterior[
        ,
        year_idx,
        drop = FALSE
      ],
      na.rm = TRUE
    )
}


# =========================================================
# 9. FIND 1993
# =========================================================

ref_idx <- which(
  comparison_years == ref_year
)

stopifnot(
  length(ref_idx) == 1
)


# =========================================================
# 10. REFERENCE TOTAL MODELLED RSL TO 1993
# =========================================================
#
# The posterior MEAN at 1993 is subtracted from the complete
# posterior distribution.
#
# This:
#
#   - sets the Ireland posterior mean to 0 in 1993
#   - preserves the natural posterior uncertainty
#   - does NOT collapse the 1993 CI to zero
#
# Therefore the uncertainty remains the uncertainty of the
# six-site mean of the total modelled RSL posterior.
# =========================================================

reslr_ref_value <- mean(
  reslr_comparison_draws[, ref_idx],
  na.rm = TRUE
)

reslr_comparison_draws_ref <-
  reslr_comparison_draws -
  reslr_ref_value


# =========================================================
# 11. SUMMARISE TOTAL MODELLED RSL POSTERIOR
# =========================================================

reslr_comparison_df <- tibble(

  Age = comparison_years,

  level_mean =
    apply(
      reslr_comparison_draws_ref,
      2,
      mean,
      na.rm = TRUE
    ),

  level_lwr =
    apply(
      reslr_comparison_draws_ref,
      2,
      quantile,
      probs = CI_LOW_PROB,
      na.rm = TRUE
    ),

  level_upr =
    apply(
      reslr_comparison_draws_ref,
      2,
      quantile,
      probs = CI_HIGH_PROB,
      na.rm = TRUE
    )

)


# =========================================================
# 12. CHECK 1993 MEAN
# =========================================================

ref_check <- reslr_comparison_df %>%
  filter(
    Age == ref_year
  )


stopifnot(
  nrow(ref_check) == 1
)


# The posterior mean is zero at 1993 after referencing.
# The CI retains its natural posterior width.

print(
  ref_check
)


# =========================================================
# 13. CHECK CREDIBLE INTERVAL
# =========================================================

stopifnot(
  all(
    reslr_comparison_df$level_lwr <=
      reslr_comparison_df$level_mean,
    na.rm = TRUE
  )
)

stopifnot(
  all(
    reslr_comparison_df$level_mean <=
      reslr_comparison_df$level_upr,
    na.rm = TRUE
  )
)


# =========================================================
# 14. PLOT
# =========================================================

p_ireland_global <- ggplot() +

  # -------------------------------------------------------
  # Ireland 95% credible interval
  # -------------------------------------------------------

  geom_ribbon(
    data = reslr_comparison_df,
    aes(
      x = Age,
      ymin = level_lwr,
      ymax = level_upr,
      fill = "95% Credible Interval"
    ),
    alpha = 0.30,
    colour = NA
  ) +

  # -------------------------------------------------------
  # Ireland six-site mean
  # -------------------------------------------------------

  geom_line(
    data = reslr_comparison_df,
    aes(
      x = Age,
      y = level_mean,
      colour = reslr_label
    ),
    linewidth = 1.2
  ) +

  # -------------------------------------------------------
  # Global tide gauges
  # -------------------------------------------------------

  geom_line(
    data = global_plot %>%
      filter(!is.na(TG_m_ref)),
    aes(
      x = Year,
      y = TG_m_ref,
      colour = "Global tide gauges"
    ),
    linewidth = 0.9
  ) +

  # -------------------------------------------------------
  # Global satellite altimetry
  # -------------------------------------------------------

  geom_line(
    data = global_plot %>%
      filter(!is.na(SSH_m_ref)),
    aes(
      x = Year,
      y = SSH_m_ref,
      colour = "Global satellite altimetry"
    ),
    linewidth = 0.9
  ) +

  # -------------------------------------------------------
  # LINE COLOURS
  # -------------------------------------------------------

  scale_colour_manual(
    name = NULL,
    values = c(
      "Ireland six-site mean of RESLR total modelled RSL" =
        "black",
      "Global tide gauges" = "blue",
      "Global satellite altimetry" = "red"
    )
  ) +

  # -------------------------------------------------------
  # 95% CREDIBLE INTERVAL
  # -------------------------------------------------------

  scale_fill_manual(
    name = NULL,
    values = c(
      "95% Credible Interval" = "#7B68EE"
    )
  ) +

  # -------------------------------------------------------
  # LEGEND
  # -------------------------------------------------------

  guides(
    colour = guide_legend(
      order = 1,
      nrow = 1
    ),
    fill = guide_legend(
      order = 2,
      nrow = 1,
      override.aes = list(
        alpha = 0.30
      )
    )
  ) +

  # -------------------------------------------------------
  # LABELS
  # -------------------------------------------------------

  labs(
    x = "Year",
    y = "Sea-level change relative to 1993 (m)",
    title =
      "Ireland six-site mean of RESLR total modelled RSL compared with global mean sea level",
    subtitle =
      "All series referenced to 1993 = 0"
  ) +

  # -------------------------------------------------------
  # THEME
  # -------------------------------------------------------

  theme_bw(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    panel.grid.minor = element_blank()
  )


# =========================================================
# 15. DISPLAY
# =========================================================

print(
  p_ireland_global
)


# =========================================================
# 16. SAVE
# =========================================================

ggsave(
  filename = file.path(
    plot_dir,
    "Figure_11_Ireland_six_site_mean_total_modelled_RSL_vs_global_sea_level_1993.png"
  ),
  plot = p_ireland_global,
  width = 10,
  height = 6,
  units = "in",
  dpi = 600,
  bg = "white"
)

# =========================================================
# PART D: CROSS-VALIDATION
# =========================================================
# Tests the predictive performance of the RESLR model using
# cross-validation.
#
# Calculates overall and site-level error metrics and
# generates the cross-validation and residual plots used
# in the manuscript.
# Figures 6 and 7 in the manuscript correspond to the
# cross-validation outputs generated in this section.
# Figure numbering follows the order of figures in the manuscript.
# =========================================================

df_cv <- df_input
# alternatively:
# df_cv <- site_combined %>%
#   filter(Age <= 2025) %>%
#   mutate(
#     linear_rate = unname(GIA[Site]),
#     linear_rate_err = 0.3
#   )

cv_all <- cross_val_check(
  data = df_cv,
  model_type = "ni_gam_decomp",
  n_iterations = 15000,  
  n_burnin = 2000,   
  n_thin = 5,
  n_chains = 3,
  spline_nseg_t = 2,
  spline_nseg_st = 1,
  seed = 35124,
  n_fold = 3,
  CI = 0.95
)


# =========================================================
# CROSS-VALIDATION OUTPUT DIRECTORY
# =========================================================

if (!dir.exists("plots_cross_validation"))
  dir.create("plots_cross_validation")


# =========================================================
# OVERALL CROSS-VALIDATION COVERAGE
# =========================================================

message(
  "Overall coverage (95% PI): ",
  round(cv_all$total_coverage * 100, 2),
  "%"
)

print(cv_all$coverage_by_site)


# =========================================================
# PREDICTION INTERVAL WIDTHS
# =========================================================

prediction_interval_size <- cv_all$prediction_interval_size %>%
  mutate(PI_width = abs(PI_width))

print(prediction_interval_size)


# =========================================================
# ERROR METRICS
# =========================================================

cv_data <- cv_all$true_pred_plot$data %>%
  mutate(error = true_RSL - pred_RSL)

ME_overall   <- mean(cv_data$error)
MAE_overall  <- mean(abs(cv_data$error))
RMSE_overall <- sqrt(mean(cv_data$error^2))

cat("\nOverall Error Metrics\n")
cat("---------------------\n")
cat("ME   :", round(ME_overall, 3), "\n")
cat("MAE  :", round(MAE_overall, 3), "\n")
cat("RMSE :", round(RMSE_overall, 3), "\n\n")


# =========================================================
# SITE-LEVEL ERROR METRICS
# =========================================================

site_metrics <- cv_data %>%
  group_by(SiteName) %>%
  summarise(
    ME   = mean(error),
    MAE  = mean(abs(error)),
    RMSE = sqrt(mean(error^2)),
    .groups = "drop"
  )

print(site_metrics)


# =========================================================
# CROSS-VALIDATION PLOT
# FIXED AXES + 95% PREDICTION INTERVALS
# =========================================================
#
# Figure 6 uses the prediction intervals returned by
# cross_val_check().
#
# These are prediction intervals (PI), not posterior
# credible intervals.
# =========================================================

cv_plot <- ggplot(
  cv_data,
  aes(
    x = true_RSL,
    y = pred_RSL
  )
) +

  # -------------------------------------------------------
  # 95% prediction intervals
  # -------------------------------------------------------

  geom_errorbar(
    aes(
      ymin = lwr_PI,
      ymax = upr_PI
    ),
    width = 0,
    colour = "grey40",
    alpha = 0.6
  ) +

  # -------------------------------------------------------
  # predicted points
  # -------------------------------------------------------

  geom_point(
    colour = "red",
    size = 2,
    alpha = 0.8
  ) +

  # -------------------------------------------------------
  # 1:1 line
  # -------------------------------------------------------

  geom_abline(
    intercept = 0,
    slope = 1,
    colour = "black",
    linewidth = 0.6
  ) +

  # -------------------------------------------------------
  # same axes for every site
  # -------------------------------------------------------

  facet_wrap(
    ~SiteName,
    ncol = 3,
    scales = "fixed"
  ) +

  coord_fixed(
    xlim = c(-0.3, 0.1),
    ylim = c(-0.3, 0.1)
  ) +

  labs(
    title = "Cross-validation: True vs Predicted Relative Sea Level",
    x = "True Relative Sea Level (m)",
    y = "Predicted Relative Sea Level (m)"
  ) +

  theme_bw() +

  theme(
    strip.text = element_text(
      size = 11,
      face = "bold"
    ),
    axis.text = element_text(
      size = 10
    ),
    axis.title = element_text(
      size = 12,
      face = "bold"
    ),
    plot.title = element_text(
      size = 14,
      face = "bold",
      hjust = 0.5
    ),
    panel.grid.minor = element_blank()
  )


# =========================================================
# DISPLAY FIGURE 6
# =========================================================

print(cv_plot)


# =========================================================
# SAVE FIGURE 6
# =========================================================

ggsave(
  filename = file.path(
    "plots_cross_validation",
    "Figure_6_cross_validation_true_vs_predicted.png"
  ),
  plot = cv_plot,
  width = 12,
  height = 9,
  dpi = 600
)


# =========================================================
# PUBLICATION RESIDUAL PLOT
# =========================================================

cv_residuals <- cv_data %>%
  mutate(
    residual = true_RSL - pred_RSL
  )


residual_plot_pub <- ggplot(
  cv_residuals,
  aes(
    x = Age,
    y = residual,
    colour = SiteName
  )
) +

  geom_point(
    size = 2.2,
    alpha = 0.8
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.6
  ) +

  facet_wrap(
    ~SiteName,
    scales = "fixed"
  ) +

  labs(
    title = "Cross-validation residuals",
    x = "Year",
    y = "Residual (Observed RSL - Predicted RSL, m)"
  ) +

  theme_bw() +

  theme(
    legend.position = "none",
    strip.text = element_text(
      size = 11,
      face = "bold"
    ),
    axis.text = element_text(
      size = 10
    ),
    axis.title = element_text(
      size = 12,
      face = "bold"
    ),
    plot.title = element_text(
      size = 14,
      face = "bold",
      hjust = 0.5
    ),
    panel.grid.minor = element_blank()
  )


# =========================================================
# DISPLAY FIGURE 7
# =========================================================

print(residual_plot_pub)


# =========================================================
# SAVE FIGURE 7
# =========================================================

ggsave(
  filename = file.path(
    "plots_cross_validation",
    "Figure_7_cross_validation_residuals.png"
  ),
  plot = residual_plot_pub,
  width = 12,
  height = 8,
  dpi = 600
)

