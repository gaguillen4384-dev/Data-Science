pacman::p_load(jsonlite, ggplot2, ggcorrplot, dplyr, tidyr, tidyverse,rpart,randomForest,xgboost)

create_feature_importance_map <- function() {
  dt_json  <- fromJSON("./Dataset/Output/decision_tree_training_metadata.json")
  xgb_json <- fromJSON("./Dataset/Output/xgb_training_metadata.json")
  rf_json  <- fromJSON("./Dataset/Output/rf_training_metadata.json")
  
  normalize <- function(x) {
    if(max(x) - min(x) == 0) return(rep(1, length(x)))
    (x - min(x)) / (max(x) - min(x))
  }

  combined_imp <- dt_json$feature_importance %>%
    transmute(Feature = variable, DT = normalize(score)) %>%
    full_join(transmute(xgb_json$feature_importance, Feature, XGB = normalize(Gain)), by = "Feature") %>%
    full_join(transmute(rf_json$feature_importance, Feature, RF = normalize(MeanDecreaseAccuracy)), by = "Feature") %>%
    mutate(across(where(is.numeric), ~replace_na(., 0))) %>%
    mutate(Consensus_Score = (DT + XGB + RF) / 3) %>%
    arrange(desc(Consensus_Score)) %>%
    head(7)
  
  write_json(combined_imp, "./Dataset/Output/top_7_features.json", pretty = TRUE)
  
  plot_data <- combined_imp %>%
    pivot_longer(cols = c(DT, XGB, RF), names_to = "Model", values_to = "Importance")
  
  heatmap_plot <- ggplot(plot_data, aes(x = Model, y = reorder(Feature, Consensus_Score))) +
    geom_tile(aes(fill = Importance), color = "white", linewidth = 0.8) +
    scale_fill_viridis_c(option = "magma", direction = -1) +
    scale_y_discrete(expand = c(0,0)) + 
    labs(title = "Top 7 Consensus Features",
         subtitle = "Based on Mean Normalized Importance",
         y = "Feature Name",
         fill = "Score") +
    theme_minimal() +
    theme(panel.grid = element_blank())
  
  ggsave("./Graphs/top_7_features_heatmap.png", plot = heatmap_plot, width = 7, height = 5)
  write_json(combined_imp, "./Dataset/Output/consensus_features.json", pretty = TRUE)
  
  message("Success: Heatmap and JSON consensus file generated.")
}

correlation_map_with_predicts <- function() {
  test_data <- read.csv("./Dataset/test_split_10.csv", stringsAsFactors = TRUE)
  
  features <- test_data[, !(names(test_data) %in% c("target"))]
  test_matrix <- model.matrix(target ~ . , data = test_data)[, -1]
  
  dt_model  <- readRDS("./Models/dt_model.rds")
  rf_model  <- readRDS("./Models/rf_model.rds")
  xgb_model <- readRDS("./Models/xgb_model.rds")
  
  # Ensure all are numeric and standardized (e.g., class labels or probabilities)
  dt_preds  <- as.numeric(predict(dt_model, features, type = "class"))
  rf_preds  <- as.numeric(predict(rf_model, features, type = "response"))
  xgb_preds <- as.numeric(predict(xgb_model, test_matrix))
  
  # Standardize XGBoost if it's returning raw probabilities (> 0.5 = 2, else 1)
  # to match the numeric 'class' output of DT/RF
  if(max(xgb_preds) <= 1) {
    xgb_preds <- ifelse(xgb_preds > 0.5, 2, 1)
  }
  
  predictions_df <- data.frame(
    Decision_Tree = dt_preds,
    Random_Forest = rf_preds,
    XGBoost       = xgb_preds
  )
  
  model_corr <- cor(predictions_df, method = "pearson")
  
  corr_plot <- ggcorrplot(model_corr, 
                          hc.order = FALSE, # Keep manual order for clarity
                          type = "full",    # "full" creates a 3x3 grid which looks better for 3 models
                          lab = TRUE, 
                          lab_size = 6, 
                          method = "square",
                          colors = c("#E46726", "white", "#6D9EC1"), 
                          title = "Model Prediction Agreement (3-Way Comparison)",
                          legend.title = "Correlation",
                          outline.color = "lightgrey") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave("./Graphs/model_agreement_correlation.png", plot = corr_plot, width = 8, height = 7)
  write_json(as.data.frame(model_corr), "./Dataset/Output/model_correlation_stats.json", pretty = TRUE)
  
  message("Comparison complete: 3-model correlation matrix saved.")
}

comparisons_model_perfomance <- function(){
  ensemble_data <- fromJSON("./Dataset/Output/ensemble_testing_output.json")
  dt_data <- fromJSON("./Dataset/Output/decision_tree_testing_output.json")
  rf_data <- fromJSON("./Dataset/Output/rf_testing_output.json")
  xgb_data <- fromJSON("./Dataset/Output/xgb_testing_output.json")
  
  metrics_df <- data.frame(
    Model = c("Ensemble", "Decision Tree", "Random Forest", "XGBoost"),
    Accuracy = c(ensemble_data$overall_accuracy[1], 
                 dt_data$overall_accuracy[1], 
                 rf_data$overall_accuracy[1], 
                 xgb_data$overall_accuracy[1]),
    Precision = c(ensemble_data$precision[1], 
                  dt_data$precision[1], 
                  rf_data$precision[1], 
                  xgb_data$precision[1]),
    Recall = c(ensemble_data$recall[1], 
               dt_data$recall[1], 
               rf_data$recall[1], 
               xgb_data$recall[1]),
    F1_Score = c(ensemble_data$f1[1], 
                 dt_data$f1[1], 
                 rf_data$f1[1], 
                 xgb_data$f1[1])
  )
  
  # Convert to long format for ggplot2
  metrics_long <- metrics_df %>%
    pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")
  
  # Graph 1: Grouped Bar Chart for overall metric comparison
  p1 <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "Comparison of Model Performance Metrics",
         subtitle = "Accuracy, Precision, Recall, and F1-Score across four models",
         y = "Score (0 to 1)",
         x = "Tree-Based Model") +
    scale_fill_brewer(palette = "Set2") +
    geom_text(aes(label = round(Value, 3)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 3)
  
  # Graph 2: Faceted Plot for individual metric focus
  p2 <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity") +
    facet_wrap(~Metric, scales = "free_y") +
    theme_light() +
    labs(title = "Detailed Metric Breakdown by Model",
         y = "Value") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(fill = "none")
  
  # Graph 3: Comparison of Precision vs. Recall (Trade-off visualization)
  p3 <- ggplot(metrics_df, aes(x = Recall, y = Precision, color = Model, label = Model)) +
    geom_point(size = 4) +
    geom_text(vjust = -1, check_overlap = TRUE) +
    xlim(0, 1) + ylim(0, 1) +
    theme_bw() +
    labs(title = "Precision vs. Recall Comparison",
         subtitle = "Visualizing the performance trade-off")
  
  ggsave("./Graphs/model_comparison_bar.png", p1, width = 5, height = 3.5, units = "in", dpi = 300)
  ggsave("./Graphs/faceted_plot_individuals.png", p2, width = 5, height = 3.5, units = "in", dpi = 300)
  ggsave("./Graphs/precision_vs_Recall.png", p3, width = 5, height = 3.5, units = "in", dpi = 300)
  
  message("Plots saved to Output folder.")
}

# --- Execution ---
# create_feature_importance_map()
# correlation_map_with_predicts()
# comparisons_model_perfomance()