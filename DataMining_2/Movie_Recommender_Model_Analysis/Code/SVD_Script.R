if (!require("jsonlite")) install.packages("jsonlite")
library(jsonlite)
if (!require("dplyr")) install.packages("dplyr")
library(dplyr)

# --- Configuration ---
MODEL_FILE <- "./Models/svd_model.rds"
TRAIN_FILE <- "./Datasets/movielens_100k.base" 
TEST_FILE  <- "./Datasets/movielens_100k.test"

load_data <- function(file_path) {
  read.table(file_path, sep="\t", col.names=c("user", "item", "rating", "timestamp"))
}

# Matrix Factorization (SVD) Training Method
fit_mf <- function(train_data, factors = 50, lr = 0.005, reg = 0.1, epochs = 50) {
  users <- max(train_data$user)
  items <- max(train_data$item)
  
  global_mu <- mean(train_data$rating)
  bu <- numeric(users) 
  bi <- numeric(items)
  
  P <- matrix(rnorm(users * factors, 0, 0.01), nrow = users, ncol = factors)
  Q <- matrix(rnorm(items * factors, 0, 0.01), nrow = items, ncol = factors)
  
  cat("Training with Biases & Regularization...\n")
  for (e in 1:epochs) {
    train_data <- train_data[sample(nrow(train_data)), ]
    total_error <- 0
    for (row in 1:nrow(train_data)) {
      u <- train_data$user[row]
      i <- train_data$item[row]
      r_ui <- train_data$rating[row]
      
      pred <- global_mu + bu[u] + bi[i] + sum(P[u, ] * Q[i, ])
      err <- r_ui - pred
      total_error <- total_error + err^2
      
      bu[u] <- bu[u] + lr * (err - reg * bu[u])
      bi[i] <- bi[i] + lr * (err - reg * bi[i])
      
      p_u_temp <- P[u, ]
      P[u, ] <- P[u, ] + lr * (err * Q[i, ] - reg * P[u, ])
      Q[i, ] <- Q[i, ] + lr * (err * p_u_temp - reg * Q[i, ])
    }
    if(e %% 10 == 0) cat(sprintf("Epoch %d complete\n", e))
  }
  
  return(list(P = P, Q = Q, bu = bu, bi = bi, mu = global_mu))
}

predict_mf <- function(model, data) {
  preds <- numeric(nrow(data))

  num_known_users <- length(model$bu)
  num_known_items <- length(model$bi)
  
  for (row in 1:nrow(data)) {
    u <- data$user[row]
    i <- data$item[row]
    
    # Check if User and Item exist in the trained model matrices
    if (u > 0 && u <= num_known_users && i > 0 && i <= num_known_items) {
      # Normal Prediction: Mu + bu + bi + (P dot Q)
      preds[row] <- model$mu + model$bu[u] + model$bi[i] + sum(model$P[u, ] * model$Q[i, ])
    } else {
      # Fallback to the global average (mu)
      preds[row] <- model$mu 
    }
  }
  preds[is.na(preds)] <- model$mu
  return(pmax(1, pmin(5, preds)))
}

get_metrics <- function(results_df, k = 10, threshold = 3) {
  rmse <- sqrt(mean((results_df$actual - results_df$predicted)^2))
  mae  <- mean(abs(results_df$actual - results_df$predicted))
  
  user_stats <- results_df %>%
    group_by(user) %>%
    arrange(desc(predicted)) %>%
    summarise(
      hits = sum(actual[1:min(n(), k)] >= threshold),
      total_rel = sum(actual >= threshold),
      dcg = sum(actual[1:min(n(), k)] / log2(2:(min(n(), k) + 1))),
      idcg = sum(sort(actual, decreasing = TRUE)[1:min(n(), k)] / 
                   log2(2:(min(n(), k) + 1)))
    ) %>%
    mutate(
      precision = hits / k,
      recall = ifelse(total_rel > 0, hits / total_rel, 0),
      ndcg = ifelse(idcg > 0, dcg / idcg, 0)
    )
  
  avg_prec <- mean(user_stats$precision, na.rm = TRUE)
  avg_rec  <- mean(user_stats$recall, na.rm = TRUE)
  f1 <- ifelse((avg_prec + avg_rec) > 0, 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec), 0)
  
  return(list(
    Model = "Funk SVD",
    RMSE = rmse,
    MAE = mae,
    Precision = avg_prec,
    Recall = avg_rec,
    F1_Score = f1,
    NDCG = mean(user_stats$ndcg, na.rm = TRUE)
  ))
}

# --- Main Execution Flow ---
train_set <- load_data(TRAIN_FILE)
test_set  <- load_data(TEST_FILE)

if (file.exists(MODEL_FILE)) {
  cat("Loading existing model...\n")
  model <- readRDS(MODEL_FILE)
} else {
  # FIXED: Calling the 'improved' function name
  model <- fit_mf(train_set)
  saveRDS(model, MODEL_FILE)
  cat("Model trained and saved.\n")
}

# Training Metrics
train_results <- data.frame(
  user = train_set$user,
  actual = train_set$rating,
  predicted = predict_mf(model, train_set)
)
train_metrics <- get_metrics(train_results, k = 10)

# Test Metrics
test_results <- data.frame(
  user = test_set$user,
  actual = test_set$rating,
  predicted = predict_mf(model, test_set)
)
test_metrics <- get_metrics(test_results, k = 10)

# Export
write(toJSON(train_metrics, pretty=TRUE, auto_unbox=TRUE), "./Datasets/Output/svd_train_metrics.json")
write(toJSON(test_metrics, pretty=TRUE, auto_unbox=TRUE), "./Datasets/Output/svd_test_metrics.json")

cat("Process complete. Metrics exported to JSON.\n")