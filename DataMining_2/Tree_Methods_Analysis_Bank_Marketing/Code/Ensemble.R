# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ipred, randomForest, caret, jsonlite, dplyr)

### --- Ensemble Testing Function (Majority Voting) ---
load_and_test_bagging <- function(test_data_path, target_var, model_dir = "./Models/") {
  test_data <- read.csv(test_data_path, stringsAsFactors = TRUE)
  actuals <- test_data[[target_var]]
  
  # Load Models
  rf_mod <- readRDS(paste0(model_dir, "rf_model.rds"))
  dt_mod <- readRDS(paste0(model_dir, "dt_model.rds"))
  bag_mod <- readRDS(paste0(model_dir, "bagging_model.rds")) # Previously xgb_model.rds
  
  # 1. Random Forest
  rf_preds <- predict(rf_mod, test_data)
  
  # 2. Decision Tree
  dt_preds <- predict(dt_mod, test_data, type = "class")
  
  # 3. Bagging Model
  # ipred::bagging uses a standard predict method for factors
  bag_preds <- predict(bag_mod, test_data)
  
  # Combine for Voting
  voting_df <- data.frame(
    rf = as.character(rf_preds),
    dt = as.character(dt_preds),
    bag = as.character(bag_preds),
    stringsAsFactors = FALSE
  )
  
  # Majority Vote: Find the mode for each row
  ensemble_final <- apply(voting_df, 1, function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  })
  
  # Ensure levels match the original data for metrics
  ensemble_final <- factor(ensemble_final, levels = levels(actuals))
  
  # Metrics
  conf_matrix <- confusionMatrix(ensemble_final, actuals, positive = "yes")
  stats <- conf_matrix$byClass
  clean_cm <- as.data.frame.matrix(conf_matrix$table)
  
  testing_stats <- list(
    overall_accuracy = conf_matrix$overall[["Accuracy"]],
    precision = stats["Precision"],
    recall = stats["Recall"],
    f1 = stats["F1"],
    confusion_matrix = clean_cm,
    model_agreement = as.list(table(rf_preds == dt_preds & dt_preds == bag_preds))
  )
  
  write_json(testing_stats, "./Dataset/Output/ensemble_testing_output.json", pretty = TRUE)
  message("Ensemble testing complete.")
}

# --- Execution ---
load_and_test_bagging("./Dataset/test_split_10.csv", "target")