if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("Matrix", quietly = TRUE)) install.packages("Matrix")

library(glmnet)
library(jsonlite)
library(Matrix)

# --- Configuration & File Paths ---
MODEL_FILE <- "./Models/sc_model.rds"
TRAIN_FILE <- "./Datasets/movielens_100k.base" 
TEST_FILE  <- "./Datasets/movielens_100k.test"

load_data <- function(file_path) {
  # Loads MovieLens data in 'user item rating timestamp' format
  read.table(file_path, sep="\t", col.names=c("user", "item", "rating", "timestamp"))
}

# --- Core Logic Functions ---

fit_sparse_coding_model <- function(train_df) {
  # Convert to a standard sparse matrix
  train_mat <- as(as(train_df[, 1:3], "realRatingMatrix"), "dgCMatrix")
  
  message("Fitting Sparse Coding model...")
  
  # use a subset of users to find an optimal 'starting' lambda
  sample_idx <- sample(1:nrow(train_mat), min(100, nrow(train_mat)))
  cv_fit <- cv.glmnet(t(train_mat), train_mat[sample_idx[1], ], alpha = 1)
  
  model <- list(
    dictionary = train_mat,
    params = list(lambda = 0.01, alpha = .01)
  )
  
  if (!dir.exists("./Models")) dir.create("./Models")
  saveRDS(model, MODEL_FILE)
  return(model)
}

predict_sparse_coding <- function(model, eval_mat_sparse) {
  D <- model$dictionary
  # Force evaluation data to match the Dictionary's items
  Y <- as(eval_mat_sparse, "dgCMatrix")
  predictions <- matrix(0, nrow = nrow(Y), ncol = ncol(Y))
  colnames(predictions) <- colnames(D)
  rownames(predictions) <- rownames(Y)
  
  Dt <- t(D)
  
  for (i in 1:nrow(Y)) {
    y_i <- as.numeric(Y[i, ])
    
    # Get the ID of the user we are currently predicting
    curr_id <- rownames(Y)[i]
    
    # Check if this user exists in the dictionary (basis)
    if (!is.null(curr_id) && curr_id %in% colnames(Dt)) {
      # Remove ONLY the current user from the dictionary for this specific prediction
      col_idx <- which(colnames(Dt) == curr_id)
      D_subset <- Dt[, -col_idx, drop = FALSE]
    } else {
      D_subset <- Dt
    }
    
    # Solve Sparse Coding
    try({
      if (sum(y_i > 0) > 0) {
        fit <- glmnet(D_subset, y_i, 
                      alpha = model$params$alpha, 
                      lambda = model$params$lambda)
        
        alpha <- as.vector(predict(fit, type = "coefficients")[-1])
        predictions[i, ] <- as.vector(D_subset %*% alpha)
      }
    }, silent = TRUE)
  }
  
  return(predictions)
}

evaluate_model <- function(model, train_df, eval_df, k = 10, threshold = 3) {
  # Setup and Alignment
  all_items <- colnames(model$dictionary)
  eval_mat_raw <- as(as(eval_df[, 1:3], "realRatingMatrix"), "dgCMatrix")
  
  # Align eval_df to the Dictionary dimensions
  aligned_eval_mat <- Matrix(0, nrow = nrow(eval_mat_raw), ncol = length(all_items), 
                             sparse = TRUE, dimnames = list(rownames(eval_mat_raw), all_items))
  
  common_items <- intersect(colnames(eval_mat_raw), all_items)
  aligned_eval_mat[, common_items] <- eval_mat_raw[, common_items]
  
  # Sparse Coding Prediction
  message("Performing Sparse Coding reconstruction...")
  pred_matrix <- predict_sparse_coding(model, aligned_eval_mat)
  
  # Ensure formats are consistent for calculation
  actual_matrix <- as.matrix(aligned_eval_mat)
  
  # RMSE & MAE Calculation
  common_mask <- (actual_matrix != 0)
  num_ratings <- sum(common_mask)
  
  if (num_ratings > 0) {
    diff_sq <- (actual_matrix[common_mask] - pred_matrix[common_mask])^2
    rmse    <- sqrt(sum(diff_sq) / num_ratings)
    mae     <- mean(abs(actual_matrix[common_mask] - pred_matrix[common_mask]))
  } else {
    rmse <- NA; mae <- NA
  }
  
  # NDCG Calculation Logic
  calc_ndcg <- function(actual, predicted, top_k) {
    rank_idx <- order(predicted, decreasing = TRUE)[1:top_k]
    dcg  <- sum(actual[rank_idx] / log2(2:(top_k + 1)))
    idcg <- sum(sort(actual, decreasing = TRUE)[1:top_k] / log2(2:(top_k + 1)))
    return(if (idcg == 0) 0 else dcg / idcg)
  }
  
  user_ndcgs <- sapply(1:nrow(actual_matrix), function(u) {
    if (sum(actual_matrix[u, ] > 0) >= 2) {
      calc_ndcg(actual_matrix[u, ], pred_matrix[u, ], k)
    } else {
      NA
    }
  })
  
  # Precision & Recall at K
  metrics_at_k <- lapply(1:nrow(actual_matrix), function(u) {
    if (sum(actual_matrix[u, ] > 0) == 0) return(NULL)
    actual_positives <- which(actual_matrix[u, ] >= threshold)
    if (length(actual_positives) == 0) return(NULL)
    
    top_k_indices <- order(pred_matrix[u, ], decreasing = TRUE)[1:k]
    tp <- length(intersect(top_k_indices, actual_positives))
    prec <- tp / k
    rec  <- tp / length(actual_positives)
    return(c(prec, rec))
  })
  
  metrics_mat <- do.call(rbind, metrics_at_k)
  avg_prec <- mean(metrics_mat[, 1], na.rm = TRUE)
  avg_rec  <- mean(metrics_mat[, 2], na.rm = TRUE)
  f1       <- 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec)
  
  return(list(
    Model = "Sparce Coding",
    RMSE = as.numeric(rmse),
    MAE = as.numeric(mae),
    Precision = as.numeric(avg_prec),
    Recall = as.numeric(avg_rec),
    F1_Score = as.numeric(f1),
    NDCG = mean(user_ndcgs, na.rm = TRUE)
  ))
}

# --- Main Execution Section ---

if (!dir.exists("./Datasets/Output")) dir.create("./Datasets/Output", recursive = TRUE)

if (!file.exists(TRAIN_FILE) || !file.exists(TEST_FILE)) {
  stop("Dataset files not found in ./Datasets/")
}

train_df <- load_data(TRAIN_FILE)
test_df  <- load_data(TEST_FILE)

# Load or Fit the Sparse Coding Model
if (file.exists(MODEL_FILE)) {
  message("Loading existing Sparse Coding model...")
  model <- readRDS(MODEL_FILE)
} else {
  # Use the function that saves Dictionary and Params
  model <- fit_sparse_coding_model(train_df)
}

# EVALUATE ON TRAINING DATA
message("Evaluating Sparse Coding on training data (Internal Consistency)...")
train_metrics <- evaluate_model(model, train_df, train_df, k = 10)
write_json(train_metrics, "./Datasets/Output/sc_train_metrics.json", auto_unbox = TRUE, pretty = TRUE)

# EVALUATE ON TEST DATA
message("Evaluating Sparse Coding on test data (Generalization)...")
test_metrics <- evaluate_model(model, train_df, test_df, k = 10)
write_json(test_metrics, "./Datasets/Output/sc_test_metrics.json", auto_unbox = TRUE, pretty = TRUE)

message("Execution complete. JSON metrics saved to ./Datasets/Output/")