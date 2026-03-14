# Load necessary libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(xgboost, caret, jsonlite, dplyr, Matrix)

### --- Training Function ---
train_xgboost <- function(csv_path, target_var, model_output_path = "./Models/xgb_model.rds") {
  data <- read.csv(csv_path, sep = ",", stringsAsFactors = TRUE)
  
  train_y <- as.numeric(data[[target_var]]) - 1

  # model.matrix handles factor conversion to numeric indicators
  train_x <- model.matrix(as.formula(paste(target_var, "~ .")), data = data)[,-1]
  dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)

params <- list(
    objective = "binary:logistic", 
    eval_metric = "logloss",
    max_depth = 6, 
    eta = 0.3
  )
  
  xgb_model <- xgb.train(
    params = params,
    data = dtrain, 
    nrounds = 100,
    verbose = 0
  )

  saveRDS(xgb_model, file = model_output_path)

  imp_matrix <- xgb.importance(feature_names = colnames(train_x), model = xgb_model)

  training_json <- list(
    metadata = list(
      target_column = target_var,
      model_type = "XGBoost",
      n_rounds = 100
    ),
    feature_importance = as.data.frame(imp_matrix)
  )

  write_json(training_json, "./Dataset/Output/xgb_training_metadata.json", pretty = TRUE)

  message("Model saved to: ", model_output_path)
}

### --- Testing Function ---
load_and_test_xgb <- function(test_data_path, target_var, model_filename = "./Models/xgb_model.rds") {
  test_data <- read.csv(test_data_path, stringsAsFactors = TRUE)
  
  # Replicate the matrix transformation used in training
  test_x <- model.matrix(as.formula(paste(target_var, "~ .")), data = test_data)[,-1]
  actuals <- test_data[[target_var]]
  
  loaded_model <- readRDS(model_filename)
  
  # XGBoost outputs probabilities [0, 1] for binary:logistic
  probs <- predict(loaded_model, test_x)
  test_preds <- factor(ifelse(probs > 0.5, levels(actuals)[2], levels(actuals)[1]), 
                       levels = levels(actuals))
  
  conf_matrix <- confusionMatrix(test_preds, actuals)
  
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
  
  write_json(testing_stats, "./Dataset/Output/xgb_testing_output.json", pretty = TRUE)
  message("Testing complete. Output saved to ./Dataset/Output/")
}

# --- Execution --- Uncomment what you want to run first.
# train_xgboost("./Dataset/train_split_90.csv", "target")
# load_and_test_xgb("./Dataset/test_split_10.csv", "target")