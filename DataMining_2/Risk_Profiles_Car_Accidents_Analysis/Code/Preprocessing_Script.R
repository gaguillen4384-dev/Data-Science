if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("lubridate", quietly = TRUE)) install.packages("lubridate")
if (!requireNamespace("fastDummies", quietly = TRUE)) install.packages("fastDummies")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("uwot", quietly = TRUE)) install.packages("uwot")
library(data.table)
library(dplyr)
library(lubridate)
library(fastDummies)
library(jsonlite)
library(uwot)

# --- PREPROCESSING FUNCTIONS ---

preprocess_raw_data <- function(file_path, id_col) {
  message("Loading and Engineering Features...")
  df_raw <- fread(file_path)
  
  # Record Initial State
  initial_row_count <- nrow(df_raw)
  
  # Feature Engineering 
  df_clean <- df_raw %>%
    mutate(
      Start_Time = as.POSIXct(Start_Time, format="%Y-%m-%dT%H:%M:%SZ", tz="UTC"),
      End_Time = as.POSIXct(End_Time, format="%Y-%m-%dT%H:%M:%SZ", tz="UTC"),
      Duration_Min = as.numeric(difftime(End_Time, Start_Time, units = "mins")),
      Start_Hour = hour(Start_Time)
    ) %>%
    select(-Start_Time, -End_Time) %>%
    na.omit() 
  
  # Encoding
  logical_cols <- c("Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
                    "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
                    "Traffic_Calming", "Traffic_Signal", "Turning_Loop")
  binary_cols <- c("Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight")
  
  df_clean <- df_clean %>%
    mutate(across(all_of(logical_cols), as.numeric)) %>%
    mutate(across(all_of(binary_cols), ~ ifelse(.x == "Day", 1, 0))) %>%
    dummy_cols(select_columns = c("Wind_Direction", "Weather_Condition"),
               remove_first_dummy = TRUE, 
               remove_selected_columns = TRUE)
  
  # Separate IDs from Features
  ids_vector <- df_clean[[id_col]]
  df_features <- df_clean %>% select(-all_of(id_col))
  
  # Clean zero variance
  var_check <- sapply(df_features, function(x) var(x, na.rm=TRUE) > 0)
  dropped_cols <- names(var_check)[!var_check]
  df_features <- df_features[, var_check, with = FALSE] # Ensure data.table syntax
  
  # Scaling
  mat <- as.matrix(df_features)
  scaled_mat <- scale(mat)
  
  # Metadata Collection
  final_row_count <- nrow(df_features)
  
  # Convert scaling attributes to named lists (Key-Value)
  scaling_center <- as.list(attr(scaled_mat, "scaled:center"))
  scaling_scale  <- as.list(attr(scaled_mat, "scaled:scale"))
  
  metadata <- list(
    rows = list(
      initial = initial_row_count,
      final = final_row_count,
      dropped = initial_row_count - final_row_count,
      retention_rate = round(final_row_count / initial_row_count, 4)
    ),
    columns = list(
      dropped_zero_variance = dropped_cols
    ),
    scaling = list(
      center = scaling_center,
      scale  = scaling_scale
    )
  )
  
  return(list(matrix = scaled_mat, ids = ids_vector, metadata = metadata))
}
run_pca_pipeline <- function(mat, variance_threshold = 0.90) {
  message("Running PCA...")
  pca_res <- prcomp(mat, center = FALSE, scale. = FALSE)
  cum_var <- cumsum(pca_res$sdev^2) / sum(pca_res$sdev^2)
  n_comp <- which(cum_var >= variance_threshold)[1]
  
  return(list(data = pca_res$x[, 1:n_comp], rotation = pca_res$rotation))
}

run_umap_pipeline <- function(mat, n_comp = 2) {
  message("Running UMAP & Saving Model...")
  umap_res <- uwot::umap(mat, n_neighbors = 15, min_dist = 0.1, 
                         n_components = n_comp, fast_sgd = TRUE, 
                         ret_model = TRUE, n_threads = parallel::detectCores() - 1)
  return(umap_res)
}

# --- EXECUTION FLOW ---

input_path <- "./Datasets/Cleaning/FL_with_imputation.csv"
output_dir <- "./Datasets/Preprocessing/"
id_col_name <- "ID" 

# Preprocess
prep <- preprocess_raw_data(input_path, id_col_name)

# Save Standardized Full with IDs
#full_output <- cbind(ID = prep$ids, as.data.table(prep$matrix))
#fwrite(full_output, paste0(output_dir, "Standardized_Full.csv"))

# Save Metadata
write_json(prep$metadata, "./Output/Stats/Preprocessing/KEY_metadata_report.json", pretty = TRUE)

# PCA
pca_results <- run_pca_pipeline(prep$matrix)
pca_output <- cbind(ID = prep$ids, as.data.table(pca_results$data))
fwrite(pca_output, paste0(output_dir, "PCA_for_Clustering.csv"))
fwrite(as.data.table(pca_results$rotation, keep.rownames = "Feature"), 
       "./Output/Stats/Preprocessing/KEY_pca_rotation.csv")

# UMAP
umap_list <- run_umap_pipeline(prep$matrix)
umap_coords <- as.data.table(umap_list$embedding)
colnames(umap_coords) <- c("UMAP1", "UMAP2")
umap_output <- cbind(ID = prep$ids, umap_coords)
fwrite(umap_output, paste0(output_dir, "UMAP_for_HDBSCAN.csv"))
uwot::save_uwot(umap_list, "./Output/Models/Dimension_Reduction/KEY_umap_model.uwot")

message("Pipeline Complete. Review metadata for row/column changes.")