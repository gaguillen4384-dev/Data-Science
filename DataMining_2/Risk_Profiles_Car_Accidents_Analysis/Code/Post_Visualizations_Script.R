if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("magrittr", quietly = TRUE)) install.packages("magrittr")
if (!requireNamespace("scales", quietly = TRUE)) install.packages("scales")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")

library(data.table)
library(ggplot2)
library(jsonlite)
library(magrittr)
library(scales)
library(tidyr)

# --- DATA LOADING FUNCTION ---
load_cluster_data <- function(file_paths) {
  data_list <- lapply(names(file_paths), function(nm) {
    dt <- fread(file_paths[[nm]])
    
    dt[, cluster_id := as.factor(Cluster)] 
    dt[, model_name := nm]
    
    return(dt[, .(model_name, cluster_id)])
  })
  
  return(rbindlist(data_list))
}

# --- SUMMARY / AGGREGATION FUNCTION ---
summarize_distributions <- function(combined_dt) {
  summary_dt <- combined_dt[, .N, by = .(model_name, cluster_id)]
  summary_dt[, cluster_id := as.factor(cluster_id)]
  return(summary_dt)
}

# --- VISUALIZATION FUNCTION ---
create_cluster_plot <- function(summary_dt, filename = "cluster_distributions.pdf") {
  p <- ggplot(summary_dt, aes(x = cluster_id, y = N, fill = model_name)) +
    geom_col(color = "white", size = 0.1) + 
    facet_wrap(~model_name, scales = "free_x") +
    theme_minimal(base_size = 14) +
    scale_fill_brewer(palette = "Dark2") +
    labs(
      title = "Observation Distribution across Cluster Models",
      subtitle = "Comparative analysis of cluster assignments (N = 800k+ per model)",
      x = "Cluster Number",
      y = "Total Observations",
    ) +
    theme(
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      strip.background = element_rect(fill = "gray95", color = NA),
      strip.text = element_text(face = "bold")
    )
  
  ggsave(filename, plot = p, device = "pdf", width = 11, height = 8.5)
  message("Success: PDF saved to ", filename)
}

create_independent_radial_plot <- function(json_paths, metadata_path, output_dir = "./Output/Graphs/Post/") {
  meta <- fromJSON(metadata_path)
  centers <- unlist(meta$scaling$center)
  scales_vec <- unlist(meta$scaling$scale)
  
  lapply(names(json_paths), function(nm) {
    
    raw_json <- fromJSON(json_paths[[nm]])
    
    dt <- rbindlist(lapply(1:length(raw_json$Cluster), function(i) {
      d_dt <- as.data.table(lapply(raw_json$Main_Drivers[i, ], unlist))
      # Ensure Cluster is captured correctly
      d_dt[, Cluster := unlist(raw_json$Cluster[[i]])]
      return(d_dt)
    }), fill = TRUE)
    
    dt_long <- melt(dt, id.vars = "Cluster", variable.name = "Feature", value.name = "Z_Score")
    dt_long <- dt_long[!is.na(Z_Score)]
    
    dt_long[, Feature_Str := as.character(Feature)]
    dt_long[, Real_Value := (as.numeric(Z_Score) * scales_vec[Feature_Str]) + centers[Feature_Str]]
    dt_long[is.na(Real_Value), Real_Value := as.numeric(Z_Score)]
    
    dt_long <- dt_long %>%
      group_by(Feature) %>%
      mutate(Value_Norm = scales::rescale(Real_Value, to = c(0.1, 1))) %>%
      ungroup()
    
    p <- ggplot(dt_long, aes(x = Feature, y = Value_Norm, color = as.factor(Cluster))) +
      geom_hline(yintercept = c(0.5, 1.0), color = "gray90", linetype = "dashed") +
      geom_segment(aes(x = Feature, xend = Feature, y = 0, yend = Value_Norm), 
                   linewidth = 0.8, alpha = 0.5) +
      geom_point(size = 3.5) +
      coord_polar(clip = "off") +
      theme_minimal(base_size = 11) +
      scale_color_brewer(palette = "Set1") +
      labs(
        title = paste("Model Profile:", nm),
        subtitle = "Radial Lollipop: Independent Drivers & Scaling",
        color = "Cluster ID",
        x = NULL, y = NULL
      ) +
      theme(
        axis.text.x = element_text(size = 9, face = "bold"),
        axis.text.y = element_blank(),
        panel.grid.major = element_line(color = "gray96"),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 9),
        legend.position = "bottom"
      )
    
    file_out <- paste0(output_dir, "Radial_Profile_", nm, ".pdf")
    ggsave(file_out, plot = p, device = "pdf", width = 8, height = 7)
    
    message(paste("Saved:", file_out))
  })
}

# --- Flow functions ---

run_distribution_flow <- function (){
  files <- list(
    "K-Means" = "./Datasets/Stats/Model_Engineering/cluster_k-means_assignments_2.csv",
    "GMM" = "./Datasets/Stats/Model_Engineering/gmm_assignments.csv",
    "Spectral" = "./Datasets/Stats/Model_Engineering/spectral_row_mappings.csv"
  )
  
  raw_data <- load_cluster_data(files)
  plot_ready_data <- summarize_distributions(raw_data)
  create_cluster_plot(plot_ready_data, 
                      "./Output/Graphs/Post/Model_Comparison_Distributions.pdf")
}

run_profile_flow <- function(){
  json_files <- list(
    "GMM"     = "./Output/Stats/Postprocessing/interpretation_gmm.json",
    "K-Means" = "./Output/Stats/Postprocessing/interpretation_k-means.json",
    "Spectral"= "./Output/Stats/Postprocessing/interpretation_spectral.json"
  )
  
  metadata_path <- "./Output/Stats/Preprocessing/KEY_metadata_report.json"
  
  create_independent_radial_plot(json_files, metadata_path)
}

# --- EXECUTION FLOW ---

# --- Distribution ---
# run_distribution_flow()

# --- Profile ---
run_profile_flow()