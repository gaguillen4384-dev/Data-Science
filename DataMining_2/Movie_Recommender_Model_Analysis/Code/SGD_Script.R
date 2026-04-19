if (!requireNamespace("recosystem", quietly = TRUE)) install.packages("recosystem")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(recosystem)
library(jsonlite)
library(dplyr)

# Assuming files are in the working directory
MODEL_FILE <- "./Models/sgd_model.rds"
TRAIN_FILE <- "./Datasets/movielens_100k.base" 
TEST_FILE  <- "./Datasets/movielens_100k.test"

load_data <- function(file_path) {
  read.table(file_path, sep="\t", col.names=c("user", "item", "rating", "timestamp"))
}

# Updated Function: Calculate Metrics including Ranking
calculate_metrics <- function(user_ids, actual, predicted, k = 10, threshold = 3) {
  
  # Global Error Metrics (RMSE and MAE)
  # RMSE formula: sqrt(sum((R_ui - R_ui')^2) / N) 
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  
  # Per-User Ranking Metrics
  results_df <- data.frame(user = user_ids, actual = actual, pred = predicted)
  
  user_stats <- results_df %>%
    group_by(user) %>%
    summarise(
      # Identify top K items based on predicted ratings
      top_k_indices = list(order(pred, decreasing = TRUE)[1:min(n(), k)]),
      
      # Precision at K: (Hits in top K) / K
      # Recall at K: (Hits in top K) / (Total Relevant items for user)
      prec = {
        top_actusgd <- actual[unlist(top_k_indices)]
        sum(top_actusgd >= threshold) / k
      },
      rec = {
        top_actusgd <- actual[unlist(top_k_indices)]
        relevant_count <- sum(actual >= threshold)
        if(relevant_count > 0) sum(top_actusgd >= threshold) / relevant_count else NA
      },
      
      # NDCG Calculation
      ndcg = {
        top_actusgd <- actual[unlist(top_k_indices)]
        # DCG = sum((2^rel - 1) / log2(rank + 1))
        dcg <- sum((2^top_actusgd - 1) / log2(seq_along(top_actusgd) + 1))
        
        # IDCG = Ideal DCG (if the user's actual ratings were perfectly sorted)
        ideal_actusgd <- sort(actual, decreasing = TRUE)[1:min(n(), k)]
        idcg <- sum((2^ideal_actusgd - 1) / log2(seq_along(ideal_actusgd) + 1))
        
        if(idcg > 0) dcg / idcg else 0
      }
    )
  
  # 3. Aggregate results
  avg_prec <- mean(user_stats$prec, na.rm = TRUE)
  avg_rec <- mean(user_stats$rec, na.rm = TRUE)
  
  # Calculate F1 Score using the averages
  f1 <- if((avg_prec + avg_rec) > 0) 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) else 0
  
  return(list(
    Model = "LFM with SGD",
    RMSE = rmse,
    MAE = mae,
    Precision = avg_prec,
    Recall = avg_rec,
    F1_Score = f1,
    NDCG = mean(user_stats$ndcg, na.rm = TRUE)
  ))
}

# --- Usage in the main workflow ---

train_raw <- load_data(TRAIN_FILE)
test_raw  <- load_data(TEST_FILE)

r <- recosystem::Reco()

train_set <- data_memory(user_index = train_raw$user, 
                         item_index = train_raw$item, 
                         rating     = train_raw$rating)

test_set  <- data_memory(user_index = test_raw$user, 
                         item_index = test_raw$item, 
                         rating     = test_raw$rating)

if (file.exists(MODEL_FILE)) {
  cat("Loading existing model from", MODEL_FILE, "\n")
  r <- readRDS(MODEL_FILE)
} else {
  cat("Training new sgd model...\n")
  # sgd optimization
  r$train(train_set, opts = list(dim = 20, niter = 20, nthread = 4))
  saveRDS(r, MODEL_FILE)
}

train_pred <- r$predict(train_set, out_memory())
test_pred  <- r$predict(test_set, out_memory())

# Evaluation & Export
train_metrics <- calculate_metrics(train_raw$user, train_raw$rating, train_pred)
test_metrics  <- calculate_metrics(test_raw$user, test_raw$rating, test_pred)

write_json(train_metrics, "./Datasets/Output/sgd_train_metrics.json", auto_unbox = TRUE, pretty = TRUE)
write_json(test_metrics, "./Datasets/Output/sgd_test_metrics.json", auto_unbox = TRUE, pretty = TRUE)

cat("Metrics written to JSON. Process complete.\n")