library(data.table)
library(jsonlite)
library(lubridate)

# --- UTILITY FUNCTIONS ---
save_summary_to_json <- function(results_list, file_path) {
  remaining_columns <- ncol(results_list$data)
  
  # Create the summary subset (excluding the large data frame)
  summary_data <- results_list[names(results_list) != "data"]
  
  # Add the new metadata fields
  summary_data$remaining_column_count <- remaining_columns
  summary_data$processed_at <- Sys.time()
  
  # 4. Write to JSON
  write_json(
    summary_data, 
    path = file_path, 
    pretty = TRUE, 
    auto_unbox = TRUE
  )
  cat("Summary metadata saved to:", file_path, "\n")
}

# --- PREPROCESSING FUNCTIONS ---

select_clustering_features <- function(df) {
  # Columns that describe 'what' and 'where'
  original_column_count = ncol(df)
  helpful_features <- c( "ID",
    "Start_Lat", "Start_Lng","Start_Time", 
    "Severity",
    "RANK", 
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
    "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
    "Traffic_Calming", "Traffic_Signal"
  )
  
  # Subset the data
  clean_df <- df[, ..helpful_features]
  
  # Calculate dropped rows
  initial_count <- nrow(clean_df)
  clean_df <- na.omit(clean_df)
  dropped_count <- initial_count - nrow(clean_df)
  
  cat("Clustering dataset prepared:", nrow(clean_df), "rows kept,", dropped_count, "rows dropped.\n")
  
  # Return both objects as a named list
  return(list(
    data = clean_df,
    initial_count = initial_count,
    original_column_count = original_column_count,
    dropped = dropped_count
  ))
}

extract_time_features <- function(input_list) {
  library(lubridate)
  df <- input_list$data
  
  # Ensure Start_Time is POSIXct
  df[, Start_Time := as.POSIXct(Start_Time)]
  
  # Create new features
  df[, `:=`(
    Hour = hour(Start_Time),
    DayOfWeek = wday(Start_Time),
    Month = month(Start_Time)
  )]
  
  # Remove the original column and check for NAs introduced by parsing
  df[, c("Start_Time", "Month") := NULL]
  
  initial_count <- nrow(df)
  df <- na.omit(df)
  
  # Update list
  input_list$data <- df
  input_list$dropped <- input_list$dropped + (initial_count - nrow(df))
  
  return(input_list)
}

convert_binary_flags <- function(input_list) {
  df <- input_list$data
  
  # Identify logical columns
  logical_cols <- names(which(sapply(df, is.logical)))
  
  # Convert TRUE/FALSE to 1/0
  for (col in logical_cols) {
    set(df, j = col, value = as.integer(df[[col]]))
  }
  
  # Dropping rows is rare here, but check for NAs just in case 
  # of unexpected data types during conversion
  initial_count <- nrow(df)
  df <- na.omit(df)
  
  # Update list
  input_list$data <- df
  input_list$dropped <- input_list$dropped + (initial_count - nrow(df))
  
  return(input_list)
}

# --- STANDARDIZATION FUNCTIONS ---

apply_scaling <- function(input_list) {

  df_scaled <- copy(input_list$data)
  
  # Identify numeric columns only
  numeric_cols <- names(which(sapply(df_scaled, is.numeric)))
  
  # Identify Zero-Variance columns (where all values are the same)
  # If SD is 0, the column provides no info for clustering and causes errors
  variances <- sapply(df_scaled[, ..numeric_cols], var, na.rm = TRUE)
  zero_var_cols <- names(variances[variances == 0 | is.na(variances)])
  
  if(length(zero_var_cols) > 0) {
    df_scaled[, (zero_var_cols) := NULL]
    # Update numeric_cols list after dropping
    numeric_cols <- setdiff(numeric_cols, zero_var_cols)
  }
  
  # Apply Z-score scaling (Standardization)
  # scale() centers at 0 and scales to SD of 1
  df_scaled[, (numeric_cols) := lapply(.SD, scale), .SDcols = numeric_cols]
  
  # Update the list
  input_list$data <- df_scaled
  input_list$scaled_cols_count <- length(numeric_cols)
  input_list$dropped_cols_count <- length(zero_var_cols)
  
  return(input_list)
}

# --- EXECUTION FLOW ---
message("Execution flow completed.")
enriched_cfl_dataset <- "./Datasets/Enriched_Central_Florida_Combined_Data.csv"
preprocessing_output_path <- "./Datasets/Preprocessing/clean_enriched_cfl.csv"
standardize_output_path <- "./Datasets/Preprocessing/standardized_enriched_cfl.csv"
metadata_path <- "./Output/Stats/Preprocessing/preprocessing_stats.json"

raw_df <- fread(enriched_cfl_dataset)

results <- select_clustering_features(raw_df)

results <- extract_time_features(results)

results <- convert_binary_flags(results)

fwrite(results$data, preprocessing_output_path)
save_summary_to_json(results,metadata_path)
message("Preprocessing flow completed...")

apply_scaling(results)
fwrite(results$data, standardize_output_path)
message("Scaling flow completed...")

message("Execution flow completed.")