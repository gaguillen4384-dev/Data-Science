# --- SETUP ---
if (!require("data.table")) install.packages("data.table")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("patchwork")) install.packages("patchwork")
if (!require("scales")) install.packages("scales")

library(data.table)
library(ggplot2)
library(patchwork)
library(scales)

# --- FUNCTIONS ---

# Calculate PCA variance metrics
prepare_scree_data <- function(pca_dt) {
  # Exclude ID column and calculate variance for each PC
  pc_vars <- unname(apply(pca_dt[, -1, with=FALSE], 2, var))
  
  data.frame(
    PC = 1:length(pc_vars), 
    Variance = pc_vars,
    PropVar = pc_vars / sum(pc_vars)
  )
}

# Generate Dual-Axis Scree Plot
plot_pca_scree <- function(scree_df) {
  total_var <- sum(scree_df$Variance)
  scale_factor <- total_var # Align proportion to raw variance scale
  
  ggplot(scree_df, aes(x = PC)) +
    geom_bar(aes(y = Variance), stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_line(aes(y = PropVar * scale_factor), color = "firebrick", size = 1) +
    geom_point(aes(y = PropVar * scale_factor), color = "firebrick", size = 2) +
    scale_y_continuous(
      name = "Variance (Eigenvalue)",
      sec.axis = sec_axis(~ . / scale_factor, 
                          name = "Proportion of Variance", 
                          labels = scales::percent)
    ) +
    theme_minimal() +
    theme(
      axis.title.y.left = element_text(color = "steelblue", face = "bold"),
      axis.text.y.left = element_text(color = "steelblue"),
      axis.title.y.right = element_text(color = "firebrick", face = "bold"),
      axis.text.y.right = element_text(color = "firebrick")
    ) +
    labs(
      title = "PCA Scree Plot: Eigenvalues vs. Proportion",
      subtitle = paste0("Evaluation of ", nrow(scree_df), " Principal Components"),
      x = "Principal Component"
    )
}

# Generate UMAP Density Map
plot_umap_density <- function(umap_dt) {
  ggplot(umap_dt, aes(x = UMAP1, y = UMAP2)) +
    geom_bin2d(bins = 100) +
    scale_fill_viridis_c(option = "magma") +
    theme_minimal() +
    labs(
      title = "UMAP Global Density Map",
      subtitle = paste0("Visualization of ", format(nrow(umap_dt), big.mark=","), " observations"),
      x = "UMAP 1", y = "UMAP 2",
      fill = "Point Density"
    )
}

# --- EXECUTION FLOW ---

# Define Paths and Loading PC count
pca_path <- "./Datasets/Preprocessing/PCA_for_Clustering.csv"
umap_path <- "./Datasets/Preprocessing/UMAP_for_HDBSCAN.csv"
output_dir <- "./Output/Graphs/Dimension_Reduction/"

# Load and Prepare Data
pca_cols <- c("ID", paste0("PC", 1:15))
pca_data <- fread(pca_path, select = pca_cols)
umap_data <- fread(umap_path)

scree_stats <- prepare_scree_data(pca_data)

# Generate Plots
combined_scree_plot <- plot_pca_scree(scree_stats)
umap_density_plot <- plot_umap_density(umap_data)

# Save Results
ggsave(paste0(output_dir, "PCA_Combined_Scree_Plot.pdf"), combined_scree_plot, width = 5, height = 5)
ggsave(paste0(output_dir, "UMAP_Density_Map.pdf"), umap_density_plot, width = 5, height = 5)

message("Processing complete. Files saved to: ", output_dir)