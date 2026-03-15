# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
# Added 'ipred' for bagging and 'randomForest' for easy importance extraction
pacman::p_load(rpart, caret, jsonlite, dplyr, ipred, randomForest)

train_bagging_model <- function(csv_path, target_var, model_output_path = "./Models/bagging_model.rds") {
  data <- read.csv(csv_path, sep = ",", stringsAsFactors = TRUE)
  
  data[[target_var]] <- as.factor(data[[target_var]])
  
  formula_str <- as.formula(paste(target_var, "~ ."))
  
  # Train Bagging model (nbagg = 25 is a common default, increase for better stability)
  bag_model <- bagging(formula_str, data = data, nbagg = 25, coob = TRUE)
  
  saveRDS(bag_model, file = model_output_path)
  
  # Calculate Variable Importance using the caret wrapper 
  # (Bagging objects don't store importance as simply as rpart)
  vi <- varImp(bag_model)
  var_importance <- data.frame(
    variable = rownames(vi),
    score = vi$Overall
  ) %>% arrange(desc(score))
  
  training_json <- list(
    metadata = list(
      target_column = target_var,
      model_type = "Bagging (Bootstrap Aggregating)",
      number_of_trees = 25,
      out_of_bag_error = bag_model$err
    ),
    feature_importance = var_importance
  )
  
  write_json(training_json, "./Dataset/Output/bagging_training_metadata.json", pretty = TRUE)
  
  message("Metadata saved to: bagging_training_metadata.json")
}

load_and_test_bagging <- function(test_data_path, target_var, model_filename = "./Models/bagging_model.rds") {
  test_data <- read.csv(test_data_path, sep = ",", stringsAsFactors = TRUE)
  test_data[[target_var]] <- as.factor(test_data[[target_var]])
  
  loaded_model <- readRDS(model_filename)
  # For bagging, type = "class" returns the majority vote
  test_preds <- predict(loaded_model, test_data, type = "class")
  
  conf_matrix <- confusionMatrix(test_preds, test_data[[target_var]], positive = "yes")
  
  testing_stats <- list(
    overall_accuracy = conf_matrix$overall[["Accuracy"]],
    precision = conf_matrix$byClass["Precision"],
    recall = conf_matrix$byClass["Recall"],
    f1 = conf_matrix$byClass["F1"],
    confusion_matrix = as.matrix(as.data.frame.matrix(conf_matrix$table))
  )
  
  write_json(testing_stats, "./Dataset/Output/bagging_testing_output.json", pretty = TRUE)
  message("Testing complete. JSON saved.")
}

# --- Execution ---
# train_bagging_model("./Dataset/train_split_90.csv", "target")
# load_and_test_bagging("./Dataset/test_split_10.csv", "target")