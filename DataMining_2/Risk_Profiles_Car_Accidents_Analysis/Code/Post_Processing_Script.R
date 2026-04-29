if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")

library(dplyr)
library(data.table)
library(jsonlite)

denormalize <- function(vals, feature_name, metadata) {
  # Check if the feature exists in metadata
  if (!(feature_name %in% names(metadata$scaling$center))) {
    return(vals)
  }
  
  # Extract scaling parameters (using [[1]] to get the numeric value)
  center <- metadata$scaling$center[[feature_name]][[1]]
  scale  <- metadata$scaling$scale[[feature_name]][[1]]
  
  # Standard Z-score reversal applied to the whole vector
  # (Z * SD) + Mean
  result <- (vals * scale) + center
  return(result)
}

get_time_of_day <- function(hour_val) {
  if (is.na(hour_val)) return("Unknown")
  
  if (hour_val >= 5  && hour_val < 8)  return("Early Morning")
  if (hour_val >= 8  && hour_val < 12) return("Morning")
  if (hour_val >= 12 && hour_val < 14) return("Midday")
  if (hour_val >= 14 && hour_val < 17) return("Afternoon")
  if (hour_val >= 17 && hour_val < 20) return("Evening")
  if (hour_val >= 20 && hour_val < 23) return("Night")
  
  return("Late Night")
}

get_dominant_cat <- function(row, prefix) {
  cols <- grep(paste0("^", prefix), names(row), value = TRUE)
  if(length(cols) == 0) return("None")
  vals <- unlist(row[, ..cols])
  max_col <- names(which.max(vals))
  return(gsub(paste0("^", prefix, "(_|\\.|\\s+|\\/)"), "", max_col))
}

# --- MAIN UPDATED FUNCTION ---
interpret_clusters <- function(profile_path, rotation_path, metadata_path, model_name) {
  profiles <- fread(profile_path)
  rotation <- fread(rotation_path)
  metadata <- fromJSON(metadata_path)
  
  top_drivers <- 5
  top_pcs <- 3
  
  raw_profiles <- copy(profiles)
  cols_to_denormalize <- intersect(names(raw_profiles), names(metadata$scaling$center))
  
  for(col in cols_to_denormalize) {
    raw_profiles[[col]] <- denormalize(raw_profiles[[col]], col, metadata)
  }
  
  # Strictly PC1-PC8
  pc_cols <- paste0("PC", 1:8)
  pc_cols <- intersect(pc_cols, names(rotation))
  
  # PCA Projection
  available_features <- intersect(rotation$Feature, names(profiles))
  rot_mat <- as.matrix(rotation[Feature %in% available_features, ..pc_cols])
  prof_mat <- as.matrix(profiles[, ..available_features])
  pc_projection <- prof_mat %*% rot_mat
  
  # Driver discovery logic (Z-score importance)
  exclude_patterns <- "Cluster|Weather_Condition|Wind_Direction|Start_Hour|Severity|Sunrise|Twilight"
  potential_driver_cols <- names(profiles)[!grepl(exclude_patterns, names(profiles))]
  col_means <- colMeans(profiles[, ..potential_driver_cols], na.rm = TRUE)
  col_sds <- sapply(profiles[, ..potential_driver_cols], sd, na.rm = TRUE)
  
  artifact <- lapply(1:nrow(profiles), function(i) {
    # 'row' has normalized Z-scores (good for math/importance)
    # 'raw_row' has actual units (good for the final report)
    row <- profiles[i, ]
    raw_row <- raw_profiles[i, ]
    
    # Driver Importance (calculated on normalized values)
    cluster_vals <- unlist(row[, ..potential_driver_cols])
    importance <- (cluster_vals - col_means) / (col_sds + 0.001)
    
    # Get the names of the top N drivers
    driver_names <- names(sort(importance, decreasing = TRUE)[1:top_drivers])
    
    # Create a named list of the drivers with their REAL (denormalized) values
    driver_values <- setNames(
      lapply(driver_names, function(d) unname(unlist(raw_row[, ..d]))), 
      driver_names
    )
    
    # Build the Item
    item <- list(
      Cluster = row$Cluster,
      Severity_Level = as.integer(round(unname(unlist(raw_row$Severity)))), 
      Time_of_Day = get_time_of_day(unname(unlist(raw_row$Start_Hour))),
      Top_PC_Alignment = names(sort(abs(pc_projection[i,]), decreasing = TRUE)[1:top_pcs])
    )
    
    item$Weather <- get_dominant_cat(row, "Weather_Condition")
    item$Wind    <- get_dominant_cat(row, "Wind_Direction")
    
    # This now contains both the feature names and their real-world numbers
    item$Main_Drivers <- driver_values
    
    return(item)
  })
  
  write_json(artifact, 
             paste0("./Output/Stats/Postprocessing/interpretation_",model_name,".json"),
             pretty = TRUE)
  
  df_csv <- as.data.frame(do.call(rbind, artifact))
  df_csv[] <- lapply(df_csv, function(col) {
    if (is.list(col)) return(vapply(col, function(x) paste(unlist(x), collapse = ", "), character(1)))
    return(unlist(col))
  })
  
  write.csv(df_csv, paste0("./Output/Stats/Postprocessing/interpretation_", model_name, ".csv"), row.names = FALSE)
  message(sprintf("Success: %s interpreted using metadata for accurate severity/time.", model_name))
}

# --- Execution Flow ---

profile_file <- "./Output/Stats/Model_Engineering/gmm_cluster_profiles.csv"
rotation_file <- "./Output/Stats/Preprocessing/KEY_pca_rotation.csv"
preprocessing_metadata_file <-"./Output/Stats/Preprocessing/KEY_metadata_report.json"

# Run processing
interpret_clusters(profile_file, rotation_file,preprocessing_metadata_file ,"gmm")
