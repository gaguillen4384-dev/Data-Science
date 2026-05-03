library(dbscan)
library(jsonlite)
library(dplyr)
library(cluster)
library(data.table)

# --- Functions ---

prepare_data <- function(file_path) {
  data <- fread(file_path)
  
  # Store IDs for re-attachment
  ids <- data$ID
  
  # Remove ID and convert to matrix for dbscan performance
  analysis_matrix <- data %>% 
    select(-ID) %>% 
    as.matrix()
  
  return(list(ids = ids, features = analysis_matrix))
}

generate_k_distance_plot <- function(data_matrix, k_val = 50, output_file = "./Output/Graphs/Model_Engineering/k_distance_elbow.pdf") {
  message("Calculating k-nearest neighbor distances for 140k rows...")
  
  # Initialize the PDF device
  pdf(file = output_file, width = 8, height = 6)
    tryCatch({
    # Generate the plot
    kNNdistplot(data_matrix, k = k_val)
    
    # Add a grid for easier 'elbow' identification
    grid()
    
    # Add a title for professional reporting
    title(main = paste("K-Distance Plot (k =", k_val, ")"),
          sub = "Look for the knee/elbow point to determine Epsilon")
    
  }, error = function(e) {
    message("Error during plotting: ", e$message)
  }, finally = {
    invisible(dev.off())
  })
  
  message("K-Distance plot saved to: ", output_file)
}

# Fit DBSCAN Model (Optimized for ~140k rows)
run_dbscan_large <- function(data_matrix, eps_val = 0.5, minPts_val = 25) {
  # For 140k rows, minPts should generally be higher (e.g., ln(n) or higher)
  # to avoid capturing tiny slivers of noise as clusters.
  message("Starting DBSCAN fit on 140k rows...")
  
  # dbscan::dbscan uses a fast KD-tree search (O(n log n))
  db_model <- dbscan(data_matrix, eps = eps_val, minPts = minPts_val)
  
  return(db_model)
}

calculate_shadow_silhouette <- function(db_model, feature_matrix, sample_size = 10000) {
  
  # Filter out noise - Silhouette is typically calculated only for clustered points
  clustered_idx <- which(db_model$cluster != 0)
  
  # For 140k rows, we MUST sample to avoid memory overflow
  # (Calculating a 140k x 140k distance matrix requires ~150GB of RAM)
  if(length(clustered_idx) > sample_size) {
    set.seed(123) # For reproducibility
    sample_idx <- sample(clustered_idx, sample_size)
  } else {
    sample_idx <- clustered_idx
  }
  
  # Calculate distance matrix for the sample
  dist_matrix <- dist(feature_matrix[sample_idx, ])
  
  # Calculate Silhouette
  sil_obj <- silhouette(db_model$cluster[sample_idx], dist_matrix)
  
  # Return the average silhouette width
  # Ranges from -1 to 1. Closer to 1 means excellent separation.
  avg_width <- mean(sil_obj[, 3])
  
  return(avg_width)
}

# Export Metrics to JSON
export_metrics_json <- function(db_model, feature_matrix, output_file = "./Output/Stats/Model_Engineering/dbscan_metrics.json") {
  
  cluster_counts <- table(db_model$cluster)
  actual_clusters <- cluster_counts[names(cluster_counts) != "0"]
  
  # --- Calculate Density Contrast ---
  # Get K-NN distances for all points
  knndist_values <- kNNdist(feature_matrix, k = db_model$minPts)
  
  # Mean distance for points in clusters vs noise
  avg_dist_in_clusters <- mean(knndist_values[db_model$cluster != 0])
  avg_dist_in_noise <- mean(knndist_values[db_model$cluster == 0])
  
  # Ratio: Higher is better (means clusters are much tighter than noise)
  density_contrast_ratio <- avg_dist_in_noise / avg_dist_in_clusters
  
  # --- Existing Logic ---
  max_size <- if(length(actual_clusters) > 0) max(actual_clusters) else 0
  max_id <- if(length(actual_clusters) > 0) names(actual_clusters)[which.max(actual_clusters)] else NA
  
  metrics <- list(
    observation_count = length(db_model$cluster),
    eps = db_model$eps,
    minPts = db_model$minPts,
    total_clusters = length(actual_clusters),
    largest_cluster_id = if(!is.na(max_id)) as.integer(max_id) else NULL,
    largest_cluster_size = as.integer(max_size),
    noise_points_count = as.integer(sum(db_model$cluster == 0)),
    noise_percentage = round((sum(db_model$cluster == 0) / length(db_model$cluster)) * 100, 4),
    density_contrast_ratio = round(density_contrast_ratio, 2),
    shadow_sil_score = calculate_shadow_silhouette(db_model, feature_matrix)
  )
  
  write_json(metrics, output_file, auto_unbox = TRUE, pretty = TRUE)
  message("Metrics exported to JSON.")
}

# Save Cluster Mapping for Future Mapping
save_mapping_file <- function(ids, clusters, output_file = "./Datasets/Model_Engineering/cluster_mapping.csv") {
  mapping_df <- data.frame(
    ID = ids,
    Cluster = clusters
  )
  
  write.csv(mapping_df, output_file, row.names = FALSE)
  message("Mapping file created: ", output_file)
}

# --- Execution Flow ---
message("Execution Flow Started")
# Verbatim file reference
target_file <- "./Datasets/Preprocessing/standardized_enriched_cfl.csv"

processed <- prepare_data(target_file)

#Note: This helps identify the eps for DBSCAN, k_val = minPts_val
generate_k_distance_plot(processed$features, k_val = 40)

# Fit Model
# Note: With 140k rows, if eps is too large, it may crash your RAM. 
db_results <- run_dbscan_large(processed$features, eps_val = .35, minPts_val = 40)
saveRDS(db_results, file = "./Output/Rds/dbscan_model.rds")

# JSON Metrics
export_metrics_json(db_results,processed$features)

# Map IDs to Clusters
save_mapping_file(processed$ids, db_results$cluster)

message("Execution Flow Completed")