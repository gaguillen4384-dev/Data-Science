# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
# Swapping xgboost for randomForest
pacman::p_load(randomForest, caret, jsonlite, dplyr)

### --- Ensemble Testing Function (Majority Voting) ---
load_and_test_rf <- function(test_data_path, target_var, model_dir = "./Models/") {
  test_data <- read.csv(test_data_path, stringsAsFactors = TRUE)
  actuals <- test_data[[target_var]]
  
  # Load Models
  rf_mod <- readRDS(paste0(model_dir, "rf_model.rds"))
  dt_mod <- readRDS(paste0(model_dir, "dt_model.rds"))
  xgb_mod <- readRDS(paste0(model_dir, "xgb_model.rds"))
  
  # Generate Predictions
  # Random Forest
  rf_preds <- predict(rf_mod, test_data)
  
  # Decision Tree
  dt_preds <- predict(dt_mod, test_data, type = "class")
  
  # XGBoost (requires matrix and thresholding)
  test_x <- model.matrix(as.formula(paste(target_var, "~ .")), data = test_data)[, -1]
  xgb_prob <- predict(xgb_mod, test_x)
  xgb_preds <- factor(ifelse(xgb_prob > 0.5, levels(actuals)[2], levels(actuals)[1]), levels = levels(actuals))
  
  voting_df <- data.frame(
    rf = as.character(rf_preds),
    dt = as.character(dt_preds),
    xgb = as.character(xgb_preds),
    stringsAsFactors = FALSE
  )
  
  # Majority Vote: Find the mode for each row
  ensemble_final <- apply(voting_df, 1, function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  })
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
    model_agreement = as.list(table(rf_preds == dt_preds & dt_preds == xgb_preds))
  )
  
  write_json(testing_stats, "./Dataset/Output/ensemble_testing_output.json", pretty = TRUE)
  message("Ensemble testing complete.")
}

# --- Execution ---
load_and_test_rf("./Dataset/test_split_10.csv", "target")