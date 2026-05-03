library(data.table)

df <- fread("./Datasets/State_Roads_TDA.csv")

# a seed for reproducibility 
set.seed(42)

# Create a random shuffling index
n <- nrow(df)
shuffled_indices <- sample(n)

# Calculate split points
train_end <- floor(0.75 * n)
val_end   <- floor((0.75 + 0.15) * n)

train_data <- df[shuffled_indices[1:train_end]]
val_data   <- df[shuffled_indices[(train_end + 1):val_end]]
test_data  <- df[shuffled_indices[(val_end + 1):n]]

fwrite(train_data, "./Datasets/FL_train.csv")
fwrite(val_data,   "./Datasets/FL_val.csv")
fwrite(test_data,  "./Datasets/FL_test.csv")

cat("Total rows: ", n, "\n",
    "Train (75%):", nrow(train_data), "\n",
    "Val   (15%):", nrow(val_data), "\n",
    "Test  (10%):", nrow(test_data), "\n")