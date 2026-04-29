if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("lubridate", quietly = TRUE)) install.packages("lubridate")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("fastDummies", quietly = TRUE)) install.packages("fastDummies")

library(tidyverse)
library(jsonlite)
library(lubridate)
library(fastDummies)


# --- Analysis Function ---
analyze_traffic_data <- function(df, metadata_path = "./Output/Stats/Preprocessing/KEY_metadata_report.json") {
  meta <- fromJSON(metadata_path)
  
  # --- 2. Data Integrity & Descriptive Analysis (Raw Data) ---
  row_count <- nrow(df)
  col_count <- ncol(df)
  duplicate_count <- sum(duplicated(df))
  missing_summary <- colSums(is.na(df))
  
  # Categorical Cardinality
  categorical_info <- df %>%
    select(where(is.character), where(is.factor)) %>%
    summarise(across(everything(), n_distinct)) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Unique_Values")
  
  # Outlier Analysis (IQR Method)
  numeric_vars_raw <- df %>% select(where(is.numeric))
  outlier_summary <- sapply(numeric_vars_raw, function(x) {
    x <- x[!is.na(x)]
    q <- quantile(x, probs = c(0.25, 0.75))
    iqr <- diff(q)
    sum(x < (q[1] - 1.5 * iqr) | x > (q[2] + 1.5 * iqr))
  })
  
  # Temporal Features
  df$Start_Time <- as.POSIXct(df$Start_Time, format="%Y-%m-%dT%H:%M:%SZ", tz="UTC")
  time_stats <- list(
    hourly = df %>%
      mutate(hour = hour(Start_Time)) %>%
      filter(!is.na(hour)) %>%
      group_by(hour) %>%
      summarise(count = n(), .groups = 'drop'),
    daily_seasonal = df %>%
      mutate(weekday = wday(Start_Time, label = TRUE)) %>%
      filter(!is.na(weekday)) %>%
      group_by(weekday) %>%
      summarise(count = n(), .groups = 'drop')
  )
  
  # Distributions
  severity_dist <- df %>% group_by(Severity) %>% summarise(count = n(), .groups = 'drop')
  weather_dist <- df %>% 
    group_by(Weather_Condition) %>% 
    summarise(count = n(), .groups = 'drop') %>%
    arrange(desc(count)) %>% 
    slice(1:15)
  
  # --- 3. Focused Correlation Processing (Severity, Temp, Vis, Weather) ---
  # Define targets based on your request and JSON naming conventions
  top_15_weather <- weather_dist$Weather_Condition
  
  df_proc <- df %>%
    # Select only the target columns
    select(all_of(intersect(c("Severity", "Temperature(F)", "Visibility(mi)", "Weather_Condition"), names(.)))) %>%
    # Group rare weather as "Other" to keep the matrix small
    mutate(Weather_Condition = ifelse(Weather_Condition %in% top_15_weather, Weather_Condition, "Other")) %>%
    # Create OHE
    dummy_cols(select_columns = "Weather_Condition",
               remove_first_dummy = TRUE, 
               remove_selected_columns = TRUE)
  
  # Apply Z-Normalization using the scaling block from your JSON
  center_list <- meta$scaling$center
  scale_list <- meta$scaling$scale
  
  for (col in names(df_proc)) {
    if (col %in% names(center_list)) {
      cntr <- center_list[[col]][1]
      scl  <- scale_list[[col]][1]
      if (!is.na(scl) && scl > 0) {
        df_proc[[col]] <- (df_proc[[col]] - cntr) / scl
      }
    }
  }
  
  # Generate Correlation Matrix
  numeric_subset <- df_proc %>% 
    select(where(is.numeric)) %>%
    select(where(~sd(.x, na.rm = TRUE) > 0))
  
  cor_matrix <- cor(numeric_subset, use = "pairwise.complete.obs")
  
  # --- 4. Return Structured List ---
  list(
    data_integrity = list(
      dimensions = c(rows = row_count, cols = col_count),
      duplicates = duplicate_count,
      missing_values = as.list(missing_summary)
    ),
    categorical_cardinality = categorical_info,
    outlier_counts = as.list(outlier_summary),
    severity_distribution = severity_dist,
    weather_distribution = weather_dist,
    temporal_distributions = time_stats,
    correlation_matrix = as.data.frame(cor_matrix)
  )
}

# --- Sensitivity Analysis Function ---
run_sensitivity_analysis <- function(df_dropped, df_imputed) {
  
  # --- Numeric Sensitivity (Your existing logic) ---
  numeric_cols <- intersect(
    names(df_dropped %>% select(where(is.numeric))),
    names(df_imputed %>% select(where(is.numeric)))
  )
  
  num_sensitivity <- data.frame(
    Variable = numeric_cols,
    Mean_Dropped = colMeans(df_dropped[, numeric_cols], na.rm = TRUE),
    Mean_Imputed = colMeans(df_imputed[, numeric_cols], na.rm = TRUE)
  ) %>%
    mutate(Pct_Change = ((Mean_Imputed - Mean_Dropped) / Mean_Dropped) * 100)
  
  # --- Categorical Sensitivity (Distributional Shift) ---
  cat_cols <- intersect(
    names(df_dropped %>% select(where(is.character), where(is.factor))),
    names(df_imputed %>% select(where(is.character), where(is.factor)))
  )
  
  # Helper function to get proportions for a column
  get_prop <- function(data, col) {
    data %>%
      count(!!sym(col)) %>%
      mutate(prop = n / sum(n)) %>%
      select(-n)
  }
  
  cat_sensitivity_list <- lapply(cat_cols, function(col) {
    p_dropped <- get_prop(df_dropped, col)
    p_imputed <- get_prop(df_imputed, col)
    
    full_join(p_dropped, p_imputed, by = col, suffix = c("_dropped", "_imputed")) %>%
      mutate(
        Variable = col,
        Abs_Diff = abs(prop_imputed - prop_dropped)
      ) %>%
      rename(Category = !!sym(col))
  })
  
  cat_sensitivity_df <- bind_rows(cat_sensitivity_list)
  
  return(list(
    numeric = num_sensitivity,
    categorical = cat_sensitivity_df
  ))
}

# --- Export Function ---
save_eda_json <- function(eda_no_impute, eda_imputed, sensitivity_data, output_path) {
  # Combining all three major components
  final_output <- list(
    metrics_no_imputation = eda_no_impute,
    metrics_with_imputation = eda_imputed,
    sensitivity_comparison = sensitivity_data
  )
  
  write_json(final_output, output_path, pretty = TRUE)
  cat("Comprehensive Analysis saved to:", output_path, "\n")
}

# --- Execution ---

# Load the dataframes
df_no_impute <- read_csv("./Datasets/Cleaning/FL_with_no_imputation.csv")
df_imputed   <- read_csv("./Datasets/Cleaning/FL_with_imputation.csv")

df_no_impute_clean <- df_no_impute %>% select(-contains("ID"), -matches("^ID$"))
results_no_impute <- analyze_traffic_data(df_no_impute_clean)
df_imputed_clean <- df_imputed %>% select(-contains("ID"), -matches("^ID$"))
results_imputed   <- analyze_traffic_data(df_imputed_clean)

sensitivity_results <- run_sensitivity_analysis(df_no_impute_clean, df_imputed_clean)

save_eda_json(
  results_no_impute, 
  results_imputed, 
  sensitivity_results, 
  "./Output/Stats/EDA/eda_comprehensive_results.json"
)