# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(rpart, rpart.plot, caret, jsonlite, dplyr)


train_decision_tree <- function(csv_path, target_var, model_output_path = "./Models/dt_model.rds") {
  data <- read.csv(csv_path, sep = ",", stringsAsFactors = TRUE)

  formula_str <- as.formula(paste(target_var, "~ ."))
  dt_model <- rpart(formula_str, data = data, method = "class")


  # This saves the entire R object for later use
  saveRDS(dt_model, file = model_output_path)


  var_importance <- dt_model$variable.importance

  training_json <- list(
    metadata = list(
      target_column = target_var,
      model_type = "Decision Tree"
    ),
    feature_importance = data.frame(
      variable = names(var_importance),
      score = as.numeric(var_importance)
    ),
    tree_structure = list(
      variables_used = unique(as.character(dt_model$frame$var[dt_model$frame$var != "<leaf>"])),
      complexity_table = as.data.frame(dt_model$cptable)
    )
  )

  write_json(training_json, "./Dataset/Output/decision_tree_training_metadata.json", pretty = TRUE)

  message("Model saved to: ", model_output_path)
  message("Metadata saved to: training_metadata.json")
}

load_and_test_tree <- function(test_data_path,  target_var, model_filename = "./Models/dt_model.rds") {
  test_data <- read.csv(test_data_path, sep = ",", stringsAsFactors = TRUE)
  test_data[[target_var]] <- as.factor(test_data[[target_var]])
  
  # Predict
  loaded_model <- readRDS(model_filename)
  test_preds <- predict(loaded_model, test_data, type = "class")
  
  conf_matrix <- confusionMatrix(test_preds, test_data[[target_var]], positive = "yes")
  
  testing_stats <- list(
    overall_accuracy = conf_matrix$overall[["Accuracy"]],
    precision = conf_matrix$byClass["Precision"],
    recall = conf_matrix$byClass["Recall"],
    f1 = conf_matrix$byClass["F1"],
    confusion_matrix = as.matrix(as.data.frame.matrix(conf_matrix$table))
  )
  
  write_json(testing_stats, "./Dataset/Output/decision_tree_testing_output.json", pretty = TRUE)
  message("Testing complete. JSON saved.")
}

# --- Execution ---
# train_decision_tree("./Dataset/train_split_90.csv", "target")
# load_and_test_tree("./Dataset/test_split_10.csv", "target")