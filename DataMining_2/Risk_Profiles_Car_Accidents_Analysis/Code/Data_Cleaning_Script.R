if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("lubridate", quietly = TRUE)) install.packages("lubridate")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")

library(data.table)
library(dplyr)
library(lubridate)
library(jsonlite)

# --- UTILITY FUNCTIONS ---

calc_mode <- function(x) {
  ux <- unique(na.omit(x))
  if(length(ux) == 0) return(NA)
  ux[which.max(tabulate(match(x, ux)))]
}

get_na_rows <- function(data, type_check) {
  sub_data <- data %>% select(where(type_check))
  if (ncol(sub_data) == 0) return(logical(nrow(data)))
  rowSums(is.na(sub_data)) > 0
}

# Row-Level Logging Utility
log_unique_id_imputations <- function(data, cols_to_impute, type, id_col, file_path) {
  if (length(cols_to_impute) == 0) return()
  
  # Identify which rows have NAs for the specific columns being processed
  na_map <- which(is.na(data %>% select(all_of(cols_to_impute))), arr.ind = TRUE)
  
  if (nrow(na_map) > 0) {
    entries <- data.frame(
      ID = data[[id_col]][na_map[, 1]],
      col_name = cols_to_impute[na_map[, 2]]
    )
    
    # Group by ID and collapse column names into a single string
    # Also record the type (Numeric or Categorical)
    summary_log <- entries %>%
      group_by(ID) %>%
      summarise(
        imputed_columns = paste(col_name, collapse = ", "),
        imputation_type = type,
        .groups = 'drop'
      ) %>%
      mutate(timestamp = Sys.time())
    
    # Append to CSV
    if (!file.exists(file_path)) {
      fwrite(summary_log, file_path)
    } else {
      fwrite(summary_log, file_path, append = TRUE)
    }
  }
}

save_imputation_log <- function(numeric_cols, categorical_cols, file_path = "imputation_metadata.json") {
  metadata <- list(
    imputed_numeric_columns = numeric_cols,
    imputed_categorical_columns = categorical_cols,
    total_unique_columns = length(unique(c(numeric_cols, categorical_cols)))
  )
  write_json(metadata, file_path, pretty = TRUE)
}

# --- CORE PROCESS FUNCTIONS ---

drop_all_missing <- function(data, output_path = "./Output/dropped_na_summary.json") {
  initial_count <- nrow(data)
  df_dropped <- na.omit(data)
  final_count <- nrow(df_dropped)
  
  drop_stats <- list(
    initial_rows = initial_count,
    rows_dropped = initial_count - final_count,
    rows_remaining = final_count,
    timestamp = Sys.time()
  )
  
  if(!dir.exists(dirname(output_path))) dir.create(dirname(output_path), recursive = TRUE)
  write_json(drop_stats, output_path, pretty = TRUE)
  return(df_dropped)
}

step_column_selection <- function(data) {
  data %>% select(
    -any_of(c("Start_Lat", "Start_Lng", "End_Lat", "End_Lng", 
              "Description", "Street", "City", "County", "State", 
              "Zipcode", "Country", "Timezone", "Airport_Code", 
              "Source", "Weather_Timestamp", "Wind_Chill(F)", "Precipitation(in)"))
  )
}

step_numeric_imputation <- function(data, id_col, log_file) {
  cols_to_impute <- data %>% 
    select(where(is.numeric)) %>% 
    summarise(across(everything(), ~any(is.na(.)))) %>% 
    tidyr::pivot_longer(everything()) %>% 
    filter(value == TRUE) %>% 
    pull(name)
  
  rows_changed <- sum(get_na_rows(data, is.numeric))
  
  if(length(cols_to_impute) > 0) {
    # Calculate median for each affected column
    subset_data <- data %>% select(all_of(cols_to_impute))
    impute_values <- sapply(subset_data, median, na.rm = TRUE)
    
    log_unique_id_imputations(data, cols_to_impute, "Numeric", id_col, log_file)
    # Perform Imputation
    data <- data %>%
      mutate(across(all_of(cols_to_impute), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
  }
  
  return(list(data = data, rows_changed = rows_changed, imputed_cols = cols_to_impute))
}

step_categorical_imputation <- function(data, id_col, log_file) {
  # Standardize empty strings to NA
  data <- data %>% mutate(across(where(is.character), ~na_if(trimws(.), "")))
  
  cols_to_impute <- data %>% 
    select(where(is.character) | where(is.factor)) %>% 
    summarise(across(everything(), ~any(is.na(.)))) %>% 
    tidyr::pivot_longer(everything()) %>% 
    filter(value == TRUE) %>% 
    pull(name)
  
  rows_changed <- sum(get_na_rows(data, ~is.character(.) | is.factor(.)))
  
  if(length(cols_to_impute) > 0) {
    subset_data <- data %>% select(all_of(cols_to_impute))
    impute_values <- sapply(subset_data, calc_mode)

    log_unique_id_imputations(data, cols_to_impute, "Categorical", id_col, log_file)
    
    # Perform Imputation
    data <- data %>%
      mutate(across(all_of(cols_to_impute), ~ifelse(is.na(.), calc_mode(.), .)))
  }
  
  return(list(data = data, rows_changed = rows_changed, imputed_cols = cols_to_impute))
}

# --- MAIN EXECUTION PROCESS ---

main <- function(file_path) {
  log_file <- "./Output/Stats/Cleaning/detailed_imputation_log.csv"
  if(file.exists(log_file)) file.remove(log_file)
  
  df <- fread(file_path)

  stats <- list(
    initial_rows = nrow(df),
    initial_cols = ncol(df),
    steps = list()
  )
  
  # Column Selection
  df_cleaned <- step_column_selection(df)
  stats$steps$column_removal <- list(remaining_cols = ncol(df_cleaned))
  
  # Sensitivity Analysis (No-imputation)
  df_no_imp <- drop_all_missing(df_cleaned, "./Output/Stats/Cleaning/drop_summary_no_imputation.json")
  fwrite(df_no_imp, "./Datasets/Cleaning/FL_with_no_imputation.csv")
  
  # Numeric Imputation
  num_res <- step_numeric_imputation(df_cleaned, "ID", log_file)
  df_cleaned <- num_res$data
  stats$steps$numeric_imputation <- list(
    rows_changed = num_res$rows_changed,
    imputed_columns = num_res$imputed_cols
  )
  
  # Categorical Imputation
  cat_res <- step_categorical_imputation(df_cleaned, "ID", log_file)
  df_cleaned <- cat_res$data
  stats$steps$character_imputation <- list(
    rows_changed = cat_res$rows_changed,
    imputed_columns = cat_res$imputed_cols
  )
  
  # Save Metadata
  save_imputation_log(
    numeric_cols = num_res$imputed_cols, 
    categorical_cols = cat_res$imputed_cols,
    file_path = "./Output/Stats/Cleaning/imputation_metadata.json"
  )
  
  # Final NA Drop
  df_cleaned <- drop_all_missing(df_cleaned, "./Output/Stats/Cleaning/final_drop_summary.json")
  
  # Export Logs and Final Data
  write_json(stats, "./Output/Stats/Cleaning/cleaning_log_imputation.json", pretty = TRUE)
  fwrite(df_cleaned, "./Datasets/Cleaning/FL_with_imputation.csv")
  
  print(paste("Process complete. Detailed row log saved to:", log_file))
}

main("./Datasets/Florida_Combined_Data.csv")