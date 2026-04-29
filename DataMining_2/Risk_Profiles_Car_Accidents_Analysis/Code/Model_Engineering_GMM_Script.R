# --- Package Management ---
if (!requireNamespace("ClusterR", quietly = TRUE)) install.packages("ClusterR")
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("GGally", quietly = TRUE)) install.packages("GGally")


# Load necessary libraries
library(ClusterR)  
library(jsonlite)
library(cluster)
library(ggplot2)
library(dplyr)
library(data.table) 
library(GGally)

# --- DATA LOADING FUNCTIONS ---
load_pca_data <- function(pca_path, rotation_path) {
  # Load PCA Components (800k rows)
  pca_full <- as.data.frame(data.table::fread(pca_path))
  row_ids <- pca_full[[1]]  # Save IDs
  pca_subset <- as.matrix(pca_full[, 2:9]) # Take only PC1-PC8
  storage.mode(pca_subset) <- "double"
  
  # Load Rotations (110 rows/features)
  # row.names = 1 handles the variable names in the first column automatically
  rotations_full <- read.csv(rotation_path, row.names = 1)
  rotations_subset <- as.matrix(rotations_full[, 1:8]) # Take PC1-PC8
  storage.mode(rotations_subset) <- "double"
  
  message("Data loaded: ", nrow(pca_subset), " rows. Rotations: ", nrow(rotations_subset), " features.")
  
  return(list(
    data = pca_subset, 
    ids = row_ids, 
    rotations = rotations_subset
  ))
}

# --- HELPERS ---
map_gmm_to_features <- function(gmm_model, rotation_matrix) {
  # ClusterR stores centroids as (Clusters x PCs)
  pca_centers <- gmm_model$centroids 
  
  # Project back to original feature space: (K x 8) %*% (8 x 110)
  feature_profiles <- pca_centers %*% t(rotation_matrix)
  
  profiles_df <- as.data.frame(feature_profiles) %>%
    mutate(Cluster = paste0("GMM_Cluster_", row_number())) %>%
    dplyr::select(Cluster, everything())
  
  return(profiles_df)
}

# --- MODEL SELECTION (Finding K) ---
find_optimal_k <- function(data, max_k = 10) {
  message("Searching for optimal K using BIC...")
  
  # sample for the search phase to save time
  set.seed(42)
  search_idx <- sample(1:nrow(data), min(nrow(data), 50000))
  data_search <- data[search_idx, ]
  
  # This function calculates BIC for a range of clusters
  # criterion = "BIC" is standard for GMM
  opt_run <- Optimal_Clusters_GMM(
    data = data_search, 
    max_clusters = max_k, 
    criterion = "BIC", 
    plot_data = FALSE,
    seed_mode = "random_subset",
    km_iter = 10,
    em_iter = 10
  )
  
  # In ClusterR, Optimal_Clusters_GMM returns a vector of BIC values
  # The index of the minimum value is the optimal K
  optimal_k <- which.min(opt_run)
  
  message("Optimal K suggested by BIC: ", optimal_k)
  
  # Optional: Plot the BIC curve to verify manually
  plot_df <- data.frame(K = 1:max_k, BIC = opt_run)
  p <- ggplot(plot_df, aes(x = K, y = BIC)) +
    geom_line() + geom_point() +
    theme_minimal() +
    labs(title = "GMM Model Selection", subtitle = "BIC vs Number of Clusters")
  ggsave("./Output/Graphs/Model_Engineering/gmm_bic_selection_plot.pdf", plot = p)
  
  return(optimal_k)
}

# --- FIT FUNCTION ---
fit_gmm <- function(data, clusters, model_path) {
  message("Fitting GMM model with K = ", clusters, "...")
  set.seed(42)
  train_idx <- sample(1:nrow(data), min(nrow(data), 100000)) # Larger sample for final fit
  
  gmm_model <- GMM(
    data = data[train_idx, ], 
    gaussian_comps = clusters, 
    dist_mode = "eucl_dist", 
    seed_mode = "random_subset", 
    km_iter = 20, # Increased for precision
    em_iter = 50  # Increased for convergence
  )
  
  if (!dir.exists(dirname(model_path))) dir.create(dirname(model_path), recursive = TRUE)
  saveRDS(gmm_model, model_path)
  return(gmm_model)
}

# --- PREDICT FUNCTION ---
predict_gmm_labels <- function(data, gmm_model) {
  message("Predicting clusters for full 800k dataset...")

  full_assignments <- predict_GMM(
    data = data, 
    CENTROIDS = gmm_model$centroids, 
    COVARIANCE = gmm_model$covariance_matrices, 
    WEIGHTS = gmm_model$weights
  )
  
  return(full_assignments$cluster_labels)
}

# --- ENHANCED METRICS ---
calculate_gmm_metrics <- function(gmm_out, data, rotations) {
  clusters <- gmm_out$clusters
  set.seed(42)
  sample_idx <- sample(1:nrow(data), 10000)
  pca_sample <- data[sample_idx, ]
  cluster_sample <- clusters[sample_idx]
  
  # Internal Validity Metrics
  sil <- silhouette(cluster_sample, dist(pca_sample))
  db_val <- clusterSim::index.DB(pca_sample, cluster_sample)$DB
  ch_val <- clusterSim::index.G1(pca_sample, cluster_sample)
  
  # Generate profiles
  profiles_df <- map_gmm_to_features(gmm_out$model, rotations)
  
  list(
    summary = list(
      model_type = "GMM_Optimized",
      timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      metrics = list(
        avg_silhouette_sampled = mean(sil[, 3]),
        davies_bouldin_sampled = db_val,
        calinski_harabasz_sampled = ch_val,
        cluster_sizes = as.vector(table(clusters))
      )
    ),
    sample_idx = sample_idx,
    sil_obj = sil,
    profiles = profiles_df 
  )
}

# --- VISUALIZATION ---
generate_gmm_plots <- function(clusters, data, metrics_results) {
  sample_idx <- metrics_results$sample_idx
  
  # Prepare data for the first 4 PCs
  # use 4 instead of 8 to keep the plot readable
  plot_df <- as.data.frame(data[sample_idx, 1:4])
  colnames(plot_df) <- paste0("PC", 1:4)
  plot_df$Cluster <- as.factor(clusters[sample_idx])
  
  # PC1 vs PC2 (The standard view)
  p1 <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.3, size = 0.5) +
    theme_minimal() + 
    labs(title = "GMM: PC1 vs PC2 (Primary Dimensions)")
  
  ggsave("./Output/Graphs/Model_Engineering/gmm_pca_PC1_PC2.pdf", plot = p1, width = 10, height = 7)
  
  # Pairs Plot (PC1 through PC4)
  # This shows how clusters separate across higher dimensions
  p_pairs <- ggpairs(plot_df, 
                     columns = 1:4, 
                     mapping = aes(color = Cluster, alpha = 0.3),
                     upper = list(continuous = wrap("cor", size = 3)),
                     lower = list(continuous = wrap("points", size = 0.1))) +
    theme_minimal() +
    labs(title = "GMM Cluster Separation across PC1-PC4")
  
  ggsave("./Output/Graphs/Model_Engineering/gmm_pca_pairs_plot.pdf", plot = p_pairs, width = 12, height = 12)
  
  # Cluster Size Distribution (Always helpful for 800k rows)
  p3 <- ggplot(data.frame(Cluster = as.factor(clusters)), aes(x = Cluster)) +
    geom_bar(fill = "steelblue") +
    theme_minimal() +
    labs(title = "GMM Cluster Size Distribution (Full 800k rows)")
  
  ggsave("./Output/Graphs/Model_Engineering/gmm_cluster_distribution.pdf", plot = p3, width = 8, height = 6)
}

# --- EXECUTION FLOW ---

gmm_path <- "./Output/Models/Model_Engineering/gmm_model.rds"

# --- LOAD ---
inputs <- load_pca_data("./Datasets/Preprocessing/PCA_for_Clustering.csv",
                        "./Output/Stats/Preprocessing/KEY_pca_rotation.csv")

# --- FIT OR LOAD ---
if (file.exists(gmm_path)) {
  message("Existing model found.")
  gmm_model_obj <- readRDS(gmm_path)
} else {
#  best_k <- find_optimal_k(inputs$data, max_k = 10)
  best_k <- 5
  gmm_model_obj <- fit_gmm(inputs$data, clusters = best_k, model_path = gmm_path)
}

# --- PREDICT ---
cluster_labels <- predict_gmm_labels(inputs$data, gmm_model_obj)
gmm_results <- list(model = gmm_model_obj, clusters = cluster_labels)

# --- METRICS & EXPORT ---
metrics_output <- calculate_gmm_metrics(gmm_results, inputs$data, inputs$rotations)
write_json(metrics_output$summary,
           "./Output/Stats/Model_engineering/gmm_comparison_metrics.json", auto_unbox = TRUE, pretty = TRUE)

data.table::fwrite(metrics_output$profiles,
                   "./Output/Stats/Model_engineering/gmm_cluster_profiles.csv")

# Save mapping with original IDs
data.table::fwrite(data.frame(ID = inputs$ids, Cluster = cluster_labels),
                   "./Datasets/Stats/Model_engineering/gmm_assignments.csv")

generate_gmm_plots(gmm_results$clusters, inputs$data, metrics_output)

message("GMM Pipeline Complete.")