# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
# Swapping xgboost for randomForest
pacman::p_load(randomForest, caret, jsonlite, dplyr)

### --- Training Function ---
train_random_forest <- function(csv_path, target_var, model_output_path = "./Models/rf_model.rds") {
  data <- read.csv(csv_path, sep = ",", stringsAsFactors = TRUE)
  
  # Random Forest in R handles formulas directly and doesn't require 
  rf_formula <- as.formula(paste(target_var, "~ ."))
  
  # Train the model
  # ntree = 100 (similar to nrounds), mtry = sqrt(features) is the default for classification
  rf_model <- randomForest(
    rf_formula,
    data = data,
    ntree = 100,
    importance = TRUE
  )
  
  saveRDS(rf_model, file = model_output_path)
  
  imp_matrix <- as.data.frame(importance(rf_model))
  imp_matrix$Feature <- rownames(imp_matrix)
  
  training_json <- list(
    metadata = list(
      target_column = target_var,
      model_type = "Random Forest",
      n_trees = 100
    ),
    feature_importance = imp_matrix
  )
  
  write_json(training_json, "./Dataset/Output/rf_training_metadata.json", pretty = TRUE)
  
  message("Random Forest model saved to: ", model_output_path)
}

### --- Testing Function ---
load_and_test_rf <- function(test_data_path, target_var, model_filename = "./Models/rf_model.rds") {
  test_data <- read.csv(test_data_path, stringsAsFactors = TRUE)
  test_data[[target_var]] <- as.factor(test_data[[target_var]])
  
  loaded_model <- readRDS(model_filename)
  
  test_preds <- predict(loaded_model, test_data)
  
  conf_matrix <- confusionMatrix(test_preds, test_data[[target_var]], positive = "yes")

  stats <- conf_matrix$byClass
  if (is.null(dim(stats))) {
    stats <- t(as.matrix(stats))
  }
  
  testing_stats <- list(
    overall_accuracy = conf_matrix$overall[["Accuracy"]],
    precision = stats[, "Precision"],
    recall = stats[, "Recall"],
    f1 = stats[, "F1"],
    confusion_matrix = as.matrix(as.data.frame.matrix(conf_matrix$table))
  )
  
  write_json(testing_stats, "./Dataset/Output/rf_testing_output.json", pretty = TRUE)
  message("Testing complete. Output saved to ./Dataset/Output/")
}

# --- Execution ---
# train_random_forest("./Dataset/train_split_90.csv", "target")
 # load_and_test_rf("./Dataset/test_split_10.csv", "target")