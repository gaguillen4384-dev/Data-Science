# Load necessary library
if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret)


data_full <- read.csv("./Dataset/bank/bank-full.csv", stringsAsFactors = TRUE)


target_col <- "target" 


set.seed(42) # Sets a seed so your results are reproducible, randomized by createDataPartition
train_index <- createDataPartition(data_full[[target_col]], p = 0.9, list = FALSE)


train_data <- data_full[train_index, ]
test_data  <- data_full[-train_index, ]


write.csv(train_data, "./Dataset/train_split_90.csv", row.names = FALSE)
write.csv(test_data, "./Dataset/test_split_10.csv", row.names = FALSE)


cat("Split Complete!\n")
cat("Training rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")

