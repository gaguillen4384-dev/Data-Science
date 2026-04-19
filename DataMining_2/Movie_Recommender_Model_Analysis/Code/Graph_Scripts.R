if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

library(ggplot2)
library(dplyr)
library(tidyr)
library(jsonlite)

# This function loads the train and test JSON for each model and merges them
load_paired_data <- function(train_paths, test_paths) {
  data_list <- lapply(seq_along(train_paths), function(i) {
    train <- fromJSON(train_paths[i]) %>% as.data.frame()
    test  <- fromJSON(test_paths[i]) %>% as.data.frame()
    
    # Rename columns to distinguish between sets
    colnames(train)[-1] <- paste0("Train_", colnames(train)[-1])
    colnames(test)[-1]  <- paste0("Test_", colnames(test)[-1])
    
    # Merge on Model name
    left_join(train, test, by = "Model")
  })
  bind_rows(data_list)
}

# --- RADAR CHART (Performance Profile) ---
plot_radar_charts <- function(df) {
  plot_data <- df %>%
    mutate(Norm_Accuracy = 1 - Test_RMSE) %>%
    select(Model, Precision = Test_Precision, Recall = Test_Recall, 
           F1 = Test_F1_Score, NDCG = Test_NDCG, Norm_Accuracy) %>%
    pivot_longer(-Model, names_to = "Metric", values_to = "Value")
  
  ggplot(plot_data, aes(x = Metric, y = Value, color = Model, fill = Model)) +
    geom_segment(aes(x = Metric, xend = Metric, y = 0, yend = Value), 
                 linewidth = 0.8, alpha = 0.7) +
    geom_point(size = 2) +
    geom_text(aes(label = sprintf("%.2f", Value)), 
              vjust = -1.5,         
              size = 3.2,           
              color = "black",     
              fontface = "bold",     
              show.legend = FALSE) +
    coord_polar(clip = "off") +    
    facet_wrap(~Model, ncol = 1) +
    ylim(0, 1.1) +                
    theme_minimal() +
    theme(
      plot.margin = margin(10, 20, 10, 20),
      axis.title = element_blank(),
      axis.text.y = element_blank(),
      axis.text.x = element_text(size = 10, face = "bold"),
      panel.grid.major = element_line(color = "grey90"),
      strip.text = element_text(size = 12, face = "bold") 
    ) +
    labs(title = "Model Performance Profiles") +
    facet_wrap(~Model, ncol = 1) 
}

# --- GROUPED BAR CHART (Error Comparison) ---
plot_error_comparison <- function(df) {
  plot_data <- df %>%
    select(Model, Train_RMSE, Test_RMSE, Train_MAE, Test_MAE) %>%
    pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
    separate(Metric, into = c("Set", "Measure"), sep = "_")
  
  ggplot(plot_data, aes(x = Model, y = Value, fill = Set)) +
    # This pairs the Train and Test bars for each model side-by-side
    geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
    # Facets keep MAE and RMSE separated as requested
    facet_wrap(~Measure, scales = "free_y") + 
    theme_bw() +
    # Custom colors to distinguish between Train and Test sets
    scale_fill_manual(values = c("Train" = "#32FF00", "Test" = "#FF5F1F")) +
    labs(
      title = "Error Comparison (RMSE vs MAE)",
      subtitle = "Paired Train vs. Test Metrics",
      x = "Model Name",
      y = "Error Value",
      fill = "Data Set"
    ) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1) # Prevents label overlap
    )
}

# --- PRECISION-RECALL SCATTER PLOT ---
plot_pr_scatter <- function(df) {
  ggplot(df, aes(x = Test_Recall, y = Test_Precision, color = Model)) +
    # Increased point size (e.g., 4)
    geom_point(size = 4) + 
    theme_minimal() +
    # Zoomed into your requested ranges
    coord_cartesian(
      xlim = c(0.25, 0.75), 
      ylim = c(0.5, 1)
    ) +
    # Tight axis padding
    scale_x_continuous(expand = c(0, 0)) + 
    scale_y_continuous(expand = c(0, 0)) +
    labs(
      title = "Precision vs Recall (Test Set)", 
      x = "Recall", 
      y = "Precision"
    )
}

# --- SLOPE CHART (Generalization Analysis) ---
plot_slope_chart <- function(df) {
  plot_data <- df %>%
    select(Model, Train_NDCG, Test_NDCG) %>%
    pivot_longer(-Model, names_to = "Dataset", values_to = "NDCG") %>%
    mutate(Dataset = factor(Dataset, levels = c("Train_NDCG", "Test_NDCG"), 
                            labels = c("Train", "Test")))
  
  ggplot(plot_data, aes(x = Dataset, y = NDCG, group = Model, color = Model)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    theme_classic() +
    labs(title = "NDCG Generalization (Train vs Test)")
}

# --- MAIN FLOW ---

train_files <- c("./Datasets/Output/svd_train_metrics.json",
                 "./Datasets/Output/sgd_train_metrics.json",
                 "./Datasets/Output/sc_train_metrics.json")
test_files  <- c("./Datasets/Output/svd_test_metrics.json",
                 "./Datasets/Output/sgd_test_metrics.json",
                 "./Datasets/Output/sc_test_metrics.json")

output_dir  <- "./Graphs"

# Process and Merge
master_df <- load_paired_data(train_files, test_files)

# Call whichever function you need
#p_radar  <- plot_radar_charts(master_df)
#ggsave(file.path(output_dir, "radar_profile.pdf"), p_radar, width = 3.5, height = 7, device = "pdf")

p_error  <- plot_error_comparison(master_df)
ggsave(file.path(output_dir, "error_comparison.pdf"), p_error, width = 5,  height = 9, device = "pdf")

#p_pr     <- plot_pr_scatter(master_df)
#ggsave(file.path(output_dir, "pr_scatter.pdf"), p_pr, width = 5,  height = 5, device = "pdf")

#p_slope  <- plot_slope_chart(master_df)
#ggsave(file.path(output_dir, "ndcg_slope.pdf"), p_slope, width = 3.5,  height = 5, device = "pdf")

# View the combined data frame structure
#print(head(master_df))