# --- Package Management ---
if (!requireNamespace("kernlab", quietly = TRUE)) install.packages("kernlab")
if (!requireNamespace("cluster", quietly = TRUE)) install.packages("cluster")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("fpc", quietly = TRUE)) install.packages("fpc")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("GGally", quietly = TRUE)) install.packages("GGally")
if (!requireNamespace("clusterSim", quietly = TRUE)) install.packages("clusterSim")

library(kernlab)     # For Spectral Clustering (specc)
library(data.table)  # Fast CSV reading
library(jsonlite)    # JSON export
library(cluster)     # Silhouette metrics
library(fpc)         # DB and CH indices
library(ggplot2)     # Visualizations
library(GGally)
library(clusterSim)

# --- Function Definitions ---

load_pca_data <- function(pca_path, rotation_path, num_pcs = 8) {
  # Load PCA scores (800k rows)
  # Assuming 1st column is ID, columns 2-9 are the first 8 PCs
  pca_full <- fread(pca_path)
  ids <- pca_full[[1]]
  pca_matrix <- as.matrix(pca_full[, 2:(num_pcs + 1)])
  
  # Load Rotation matrix (Loadings)
  rotations_raw <- fread(rotation_path)
  
  # If column 1 is variable names, take columns 2 through (num_pcs + 1)
  # and convert only those numeric columns to a matrix
  rotation_matrix <- as.matrix(rotations_raw[, 2:(num_pcs + 1)])
  
  # Optional: Keep the variable names as row names for the profiles later
  rownames(rotation_matrix) <- rotations_raw[[1]]
  
  return(list(data = pca_matrix, ids = ids, rotations = rotation_matrix))
}

find_optimal_k_spectral <- function(pca_matrix, max_k = 20, nystrom_sample = 5000) {
  cat("Calculating eigenvalues via Nystrom approximation for Eigengap Heuristic...\n")
  
  # use the kernel matrix approximation to get eigenvalues
  # In spectral clustering, look for the largest 'gap' between consecutive eigenvalues
  # sorted in descending order (for the similarity matrix)
  set.seed(42)
  # Taking a sample to estimate the global structure
  sample_idx <- sample(1:nrow(pca_matrix), nystrom_sample)
  sample_data <- pca_matrix[sample_idx, ]
  
  # Create a kernel matrix (Linear Kernel as requested)
  # compute the eigenvalues of the approximated Graph Laplacian/Similarity matrix
  K <- kernelMatrix(vanilladot(), sample_data)
  ev <- eigen(K, only.values = TRUE)$values
  
  # Calculate gaps: diff(ev)
  # are looking for the index i where ev[i] - ev[i+1] is maximized
  gaps <- abs(diff(ev[1:max_k]))
  optimal_k <- which.max(gaps)
  
  # Plotting the Eigengap
  df_ev <- data.frame(k = 1:max_k, Eigenvalue = ev[1:max_k])
  
  p_gap <- ggplot(df_ev, aes(x = k, y = Eigenvalue)) +
    geom_line() + geom_point() +
    theme_minimal() +
    labs(title = "Eigengap Heuristic (Spectral Clustering)",
         subtitle = paste("Suggested k based on max gap:", optimal_k))

  ggsave("./Output/Graphs/Model_Engineering/spectral_eigengap_plot.pdf", plot = p_gap, width = 8, height = 6)
  
  return(list(optimal_k = optimal_k, eigenvalues = ev[1:max_k]))
}

run_spectral_clustering <-function(pca_matrix, k, nystrom_sample = 10000) {
  cat(paste0("[", Sys.time(), "] Initiating Turbo Spectral Mode with ", nystrom_sample, " landmarks...\n"))
  
  if (!is.matrix(pca_matrix)) pca_matrix <- as.matrix(pca_matrix)
  n <- nrow(pca_matrix)
  
  # Take landmark sample
  set.seed(42)
  idx <- sample(1:n, nystrom_sample)
  landmark_data <- pca_matrix[idx, ]
  
  cat(paste0("[", Sys.time(), "] Computing Kernel Matrix & Eigen-decomposition...\n"))
  
  # Linear Kernel of landmarks
  K_landmark <- landmark_data %*% t(landmark_data)
  
  # Get Top Eigenvectors
  decomp <- eigen(K_landmark, symmetric = TRUE)
  # Ensure it don't take more eigenvectors than have positive eigenvalues
  U <- decomp$vectors[, 1:k]
  Lambda_inv <- diag(1 / sqrt(pmax(decomp$values[1:k], 1e-10)))
  
  # Pre-compute the projection matrix to save operations inside the loop
  projection_matrix <- U %*% Lambda_inv
  
  cat(paste0("[", Sys.time(), "] Projecting 800k rows via Chunked Execution...\n"))
  
  # Chunked Projection to save RAM
  spectral_space <- matrix(0, nrow = n, ncol = k)
  chunk_size <- 100000 
  num_chunks <- ceiling(n / chunk_size)
  
  for (i in 1:num_chunks) {
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, n)
    
    # Calculate K_cross only for this chunk
    # (chunk_size x nystrom_sample) matrix
    K_chunk <- pca_matrix[start_idx:end_idx, ] %*% t(landmark_data)
    
    # Project chunk into spectral space
    spectral_space[start_idx:end_idx, ] <- K_chunk %*% projection_matrix
    
    if(i %% 2 == 0) cat(paste0("...Processed ", end_idx, " rows\n"))
  }
  
  # Normalize rows (Standard Spectral Step)
  cat("Normalizing spectral space...\n")
  row_norms <- sqrt(rowSums(spectral_space^2))
  # Avoid division by zero
  spectral_space <- spectral_space / pmax(row_norms, 1e-10)
  
  cat(paste0("[", Sys.time(), "] Running final K-means on Spectral Space...\n"))
  
  # Final Clustering
  final_clusters <- kmeans(spectral_space, centers = k, nstart = 5, iter.max = 100)
  
  return(final_clusters$cluster)
}

calculate_spectral_metrics <- function(clusters, data, rotations) {
  cat(paste0("[", Sys.time(), "] Calculating metrics and profiles...\n"))
  
  # --- Sampling for Metrics ---
  set.seed(42)
  # Ensure sample size doesn't exceed data size
  sample_size <- min(10000, nrow(data))
  sample_idx <- sample(1:nrow(data), sample_size)
  
  pca_sample <- data[sample_idx, ]
  cluster_sample <- clusters[sample_idx]
  
  # --- Matrix Preparation ---
  # Ensure everything is in numeric matrix format for math operations
  data_mat <- as.matrix(data)
  rot_mat  <- as.matrix(rotations)
  
  if(!is.numeric(rot_mat)) {
    storage.mode(rot_mat) <- "numeric" 
  }
  
  # --- Centroid & Profile Calculation ---
  # Calculate centroids in PC space
  k_centers <- sort(unique(clusters))
  pc_centroids <- t(sapply(k_centers, function(i) {
    colMeans(data_mat[clusters == i, , drop = FALSE])
  }))
  
  # Project centroids back to original feature space
  # (k x PCs) %*% (PCs x Original_Vars) = (k x Original_Vars)
  profiles_matrix <- pc_centroids %*% t(rot_mat)
  
  # Create DF with Cluster as the FIRST column
  profiles_df <- data.frame(
    Cluster = as.factor(k_centers),
    as.data.frame(profiles_matrix)
  )
  
  # --- Internal Validity Metrics ---
  # These are computationally expensive, hence the sampling
  sil <- silhouette(cluster_sample, dist(pca_sample))
  db_val <- clusterSim::index.DB(pca_sample, cluster_sample)$DB
  ch_val <- clusterSim::index.G1(pca_sample, cluster_sample)
  
  # --- Return Results ---
  return(list(
    summary = list(
      model_type = "Spectral_Clustering_Linear",
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
    profiles = profiles_df  # Cluster is now the first column here
  ))
}

map_back_to_original <- function(pca_matrix, labels, rotation_matrix) {
  # Using data.table for speed given 800k rows
  dt <- data.table(pca_matrix)
  dt[, Cluster := labels]
  
  # Calculate centroids in PC space
  pc_centroids <- as.matrix(dt[, lapply(.SD, mean), by = Cluster, .SDcols = 1:ncol(pca_matrix)][order(Cluster)][, Cluster := NULL])
  
  # Map back to original features
  original_centroids <- pc_centroids %*% t(rotation_matrix)
  return(original_centroids)
}

save_visualizations <- function(clusters, data, metrics_results) {
  sample_idx <- metrics_results$sample_idx
  
  # Prepare data for the first 4 PCs for readability
  plot_df <- as.data.frame(data[sample_idx, 1:4])
  colnames(plot_df) <- paste0("PC", 1:4)
  plot_df$Cluster <- as.factor(clusters[sample_idx])
  
  # Ensure output directory exists
  out_dir <- "./Output/Graphs/Model_Engineering/"
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  
  # 1. PC1 vs PC2 (The standard view)
  p1 <- ggplot(plot_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.3, size = 0.5) +
    theme_minimal() + 
    labs(title = "Spectral Clustering: PC1 vs PC2 (Primary Dimensions)")
  
  ggsave(paste0(out_dir, "spectral_pca_PC1_PC2.pdf"), plot = p1, width = 10, height = 7)
  
  # 2. Pairs Plot (PC1 through PC4)
  p_pairs <- ggpairs(plot_df, 
                     columns = 1:4, 
                     mapping = aes(color = Cluster, alpha = 0.3),
                     upper = list(continuous = wrap("cor", size = 3)),
                     lower = list(continuous = wrap("points", size = 0.1))) +
    theme_minimal() +
    labs(title = "Spectral Cluster Separation across PC1-PC4")
  
  ggsave(paste0(out_dir, "spectral_pca_pairs_plot.pdf"), plot = p_pairs, width = 12, height = 12)
  
  #Cluster Size Distribution (Full 800k rows)
  p3 <- ggplot(data.frame(Cluster = as.factor(clusters)), aes(x = Cluster)) +
    geom_bar(fill = "darkred") +
    theme_minimal() +
    labs(title = "Spectral Cluster Size Distribution (Full 800k rows)")
  
  ggsave(paste0(out_dir, "spectral_cluster_distribution.pdf"), plot = p3, width = 8, height = 6)
}

# --- Execution Flow ---

# Configuration
PCA_FILE <- "./Datasets/Preprocessing/PCA_for_Clustering.csv"
ROTATION_FILE <- "./Output/Stats/Preprocessing/KEY_pca_rotation.csv"
MODEL_RDS <- "./Output/Models/Model_Engineering/spectral_model.rds" 
input_data <- load_pca_data(PCA_FILE, ROTATION_FILE)

# Model Execution
if (file.exists(MODEL_RDS)) {
  cat("Existing model found:", MODEL_RDS, "- Loading from disk...\n")
  spectral_model <- readRDS(MODEL_RDS)
} else {
  # If k is not found with Eigengap Heuristic then uncomment
#  k_selection <- find_optimal_k_spectral(input_data$data, max_k = 15)
#  k_suggested <- k_selection$optimal_k # k=1 is not enough though its the greatest gap.
  k_suggested = 8
  spectral_model <- run_spectral_clustering(input_data$data, k = k_suggested)
  saveRDS(spectral_model, MODEL_RDS)
  cat("Model training complete and saved to RDS.\n")
}

#  Comprehensive Metrics & Profile Generation
labels <- as.integer(spectral_model)
spectral_metrics <- calculate_spectral_metrics(labels, input_data$data, input_data$rotations)
write_json(spectral_metrics$summary, "./Output/Stats/Model_Engineering/spectral_comparison_metrics.json", auto_unbox = TRUE, pretty = TRUE)
write.csv(spectral_metrics$profiles, "./Output/Stats/Model_Engineering/spectral_cluster_profiles.csv", row.names = FALSE)
mapping_df <- data.table(
  Observation_ID = input_data$ids,
  Cluster = labels
)
fwrite(mapping_df, "./Datasets/Stats/Model_Engineering/spectral_row_mappings.csv")

# Visualizations
save_visualizations(labels, input_data$data, spectral_metrics)

cat("Process complete. Model, metrics, and plots saved successfully.\n")