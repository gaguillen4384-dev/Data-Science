# --- Package Management ---
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
if (!requireNamespace("factoextra", quietly = TRUE)) install.packages("factoextra")
if (!requireNamespace("gridExtra", quietly = TRUE)) install.packages("gridExtra")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("GGally", quietly = TRUE)) install.packages("GGally")
if (!requireNamespace("clusterSim", quietly = TRUE)) install.packages("clusterSim")
if (!requireNamespace("fpc", quietly = TRUE)) install.packages("fpc")

library(tidyverse)
library(cluster)
library(factoextra)
library(gridExtra)
library(jsonlite)
library(data.table)
library(GGally)
library(clusterSim) # Required for Davies-Bouldin
library(fpc)        # Required for Calinski-Harabasz

# --- Load Data & Filter to first 8 PCs ---
load_pca_data <- function(file_path) {
  df <- fread(file_path)
  
  # CAPTURE IDs: This keeps the original row identifiers for the CSV export later
  ids <- df[[1]]
  id_name <- colnames(df)[1] # Store the original column name (e.g., "RowID")
  
  # Process Numeric Data
  df_numeric <- df %>%
    dplyr::select(-1) %>%               
    dplyr::select(1:8) %>%              
    filter(across(everything(), ~ is.finite(.))) %>%
    as.data.frame()              
  
  # Attach IDs as row names for internal tracking
  if(nrow(df_numeric) == nrow(df)) {
    rownames(df_numeric) <- ids
  }
  
  message(paste("Loaded", nrow(df_numeric), "observations and 8 numeric PCs."))
  # Return both the data and the ID name for the final CSV header
  return(list(data = df_numeric, id_name = id_name))
}

load_rotation_data <- function(file_path) {
  rot <- fread(file_path)
  
  # Extract the feature names to use as labels later
  feature_names <- rot$Feature
  
  # Select only the PC columns used (PC1 through PC8)
  # use 'all_of' to be safe or index by position if names are standard
  rot_matrix <- as.matrix(rot[, .(PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8)])
  
  # Attach the names back to the matrix rows
  rownames(rot_matrix) <- feature_names
  
  return(rot_matrix)
}

# --- Determine Optimal K (Visual Evaluation) ---
generate_diagnostic_plots <- function(df, output_file = "diagnostic_plots.pdf", sample_size = 10000) {
  set.seed(42)
  df_sample <- if(nrow(df) > sample_size) sample_n(df, sample_size) else df
  
  p_elbow <- fviz_nbclust(df_sample, kmeans, method = "wss") + labs(title = "Elbow Method (Sampled)")
  p_sil <- fviz_nbclust(df_sample, kmeans, method = "silhouette") + labs(title = "Average Silhouette Method (Sampled)")
  
  diag_combined <- marrangeGrob(list(p_elbow, p_sil), nrow = 1, ncol = 2, top = NULL)
  ggsave(output_file, diag_combined, width = 11, height = 8.5)
  
  return(list(elbow = p_elbow, silhouette = p_sil))
}

# --- Execute K-Means Modeling ---
run_kmeans_model <- function(df, k_val, seed = 123) {
  set.seed(seed)
  km_model <- kmeans(df, centers = k_val, nstart = 25, iter.max = 20)
  return(km_model)
}

map_clusters_to_features <- function(km_model, rotation_matrix) {
  # Matrix Math: (Clusters x PCs) %*% (PCs x Features)
  # This projects cluster centers back into the original feature space
  feature_profiles <- km_model$centers %*% t(rotation_matrix)
  
  profiles_df <- as.data.frame(feature_profiles) %>%
    mutate(Cluster = paste0("Cluster_", row_number())) %>%
    dplyr::select(Cluster, everything())
  
  return(profiles_df)
}

# --- Extract Metrics & Build Metadata (Assignments Removed from JSON) ---
get_model_metadata <- function(df, km_model, k_val, sample_size = 10000) {
  set.seed(42)
  # Sampling is used here to keep distance matrix calculations efficient
  idx <- sample(1:nrow(df), min(nrow(df), sample_size))
  df_sample <- df[idx, ]
  cluster_sample <- km_model$cluster[idx]
  
  # Silhouette Width
  sil_data <- silhouette(cluster_sample, dist(df_sample))
  avg_sil_width <- mean(sil_data[, 3])
  
  # Davies-Bouldin Index
  # Measures average similarity between clusters (Lower = better separation)
  db_index <- index.DB(df_sample, cluster_sample)$DB
  
  # Calinski-Harabasz Index
  # Variance ratio criterion (Higher = better defined clusters)
  ch_stats <- cluster.stats(dist(df_sample), cluster_sample)
  ch_index <- ch_stats$ch
  
  metadata <- list(
    model_type = "K-Means",
    timestamp = Sys.time(),
    obs_count = nrow(df),
    parameters = list(k = k_val, nstart = 25, max_iter = 20),
    metrics = list(
      totss = km_model$totss,
      tot_withinss = km_model$tot.withinss,
      betweenss = km_model$betweenss,
      ratio_bss_tss = (km_model$betweenss / km_model$totss) * 100,
      avg_silhouette_sampled = avg_sil_width,
      davies_bouldin = db_index,
      calinski_harabasz = ch_index
    ),
    cluster_sizes = as.vector(km_model$size)
  )
  
  return(list(metadata = metadata, sil_data = sil_data, df_sample = df_sample, cluster_sample = cluster_sample))
}

# --- Save Outputs (Including CSV for Assignments) ---
save_results <- function(km_model, metadata, id_name, row_ids, 
                         model_file = "final_kmeans_model.rds", 
                         json_file = "kmeans_comparison_metrics.json",
                         csv_file = "cluster_assignments.csv") {
  
  # Save RDS
  saveRDS(km_model, model_file)
  
  # Save JSON (Metadata only)
  write_json(metadata, json_file, pretty = TRUE, auto_unbox = TRUE)
  
  # Save CSV (Assignments with IDs)
  assignments_df <- data.frame(
    ID = row_ids,
    Cluster = km_model$cluster
  )
  colnames(assignments_df)[1] <- id_name # Use original ID column name
  fwrite(assignments_df, csv_file)
  
  message(paste("Files exported: JSON (metrics), RDS (model), and CSV (assignments)."))
}

# --- Final Visualizations ---
generate_final_plots <- function(km_model, results_data, k_val = NULL, output_path = "./Output/Graphs/Model_Engineering/") {  
  # Prepare data for the first 4 PCs (consistent with GMM logic)
  # Assumes results_data$df_sample contains the PC components
  sample_idx <- seq_len(nrow(results_data$df_sample)) 
  plot_df <- as.data.frame(results_data$df_sample[, 1:4])
  colnames(plot_df) <- paste0("PC", 1:4)
  plot_df$Cluster <- as.factor(results_data$cluster_sample)
  
  # PC1 vs PC2 
  p1 <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.3, size = 0.5) +
    theme_minimal() + 
    labs(title = paste0("K-Means (k=", k_val, "): PC1 vs PC2 (Primary Dimensions)"),
         subtitle = "Visualizing cluster centroids and point density")
  
  ggsave(paste0(output_path, "k-means_pca_PC1_PC2_k", k_val, ".pdf"), 
         plot = p1, width = 10, height = 7)
  
  # Pairs Plot (PC1 through PC4)
  p_pairs <- ggpairs(plot_df, 
                     columns = 1:4, 
                     mapping = aes(color = Cluster, alpha = 0.3),
                     upper = list(continuous = wrap("cor", size = 3)),
                     lower = list(continuous = wrap("points", size = 0.1))) +
    theme_minimal() +
    labs(title = paste0("K-Means (k=", k_val, ") Cluster Separation across PC1-PC4"))
  
  ggsave(paste0(output_path, "k-means_pca_pairs_plot_k", k_val, ".pdf"), 
         plot = p_pairs, width = 12, height = 12)
  
  # Cluster Size Distribution
  # Pulling from the full model object to represent the 800k rows correctly
  p3 <- ggplot(data.frame(Cluster = as.factor(km_model$cluster)), aes(x = Cluster)) +
    geom_bar(fill = "darkred") + 
    theme_minimal() +
    labs(title = paste0("K-Means Cluster Size Distribution (k=", k_val, " - Full Data)"),
         caption = paste("Total Observations:", length(km_model$cluster)))
  
  ggsave(paste0(output_path, "k-means_cluster_distribution_k", k_val, ".pdf"), 
         plot = p3, width = 8, height = 6)
  
  # Silhouette Plot (Keeping your existing fviz_silhouette logic for profile analysis)
  viz_sil <- fviz_silhouette(results_data$sil_data) + 
    theme_minimal() + 
    labs(title = paste0("K-Means Silhouette Profile (Sampled, k=", k_val, ")"))
  
  ggsave(paste0(output_path, "k-means_silhouette_k", k_val, ".pdf"), 
         plot = viz_sil, width = 11, height = 8.5)
  
  message(paste0("Success: K-Means graphs for k=", k_val, " saved to ", output_path))
}

# --- EXECUTION FLOW ---

# Data Intake
pca_output <- load_pca_data("./Datasets/Preprocessing/PCA_for_Clustering.csv")
path_rotation <- "./Output/Stats/Preprocessing/KEY_pca_rotation.csv"
df_pca     <- pca_output$data
rot_matrix <- load_rotation_data(path_rotation)

# Run this to see the Elbow and Silhouette plots in Output folder
# These help confirm if k_val should be 2, 3, 4, etc.
#diag_plots <- generate_diagnostic_plots(
#  df_pca, 
#  output_file = "./Output/Graphs/Model_Engineering/k-means_diagnostic_plots.pdf")

# Model Execution
k_val <- 2  # <--- Change this based on the diagnostic plots
model_path <- paste0("./Output/Models/Model_Engineering/k-means_model_", k_val, ".rds")
if (file.exists(model_path)) {
  message(paste("Existing model found for k =", k_val, "- Loading from RDS..."))
  km_res <- readRDS(model_path)
} else {
  message(paste("No existing model found for k =", k_val, "- Running K-Means..."))
  km_res <- run_kmeans_model(df_pca, k_val)
}

# Export Results
# Map Back to Original Features
feature_interpretation <- map_clusters_to_features(km_res, rot_matrix)

# Feature Profiles (What does each cluster mean?)
fwrite(feature_interpretation, "./Output/Stats/Model_Engineering/k-means_cluster_profiles.csv")

# Metric Extraction
results_data <- get_model_metadata(df_pca, km_res, k_val)

# File Export
save_results(km_res, 
             results_data$metadata, 
             id_name = pca_output$id_name,
             row_ids = rownames(df_pca),
             model_file=paste0("./Output/Models/Model_Engineering/k-means_model_",k_val,".rds"),
             json_file=paste0("./Output/Stats/Model_Engineering/k-means_comparison_metrics_",k_val,".json"),
             csv_file=paste0("./Datasets/Stats/Model_Engineering/cluster_k-means_assignments_",k_val,".csv"))

# Final Visualization
generate_final_plots(km_res, results_data)

message("Success: Diagnostic plots, model, and CSV assignments exported.")