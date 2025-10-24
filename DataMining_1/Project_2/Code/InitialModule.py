# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import pandas as panda_object
from sklearn.model_selection import train_test_split

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the full path to the file
file_path = os.path.join(script_dir, 'DataSet', 'maize.sas7bdat')

# Load dataframe for file manipulation
dataframe = panda_object.read_sas(file_path)
columns_to_drop = ['Entry', 'Geno_Code','DtoA']
X = dataframe.drop(columns_to_drop, axis=1) # Features (columns to retain)
y = dataframe['DtoA']             # Target variable

X_data = X.values
y_data = y.values

# 1. Split: 90% (Train/Val) and 10% (Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_data, y_data, 
    test_size=0.10,            # 10% for the final test set
    random_state=42
)

# 2. Split: The remaining 90% is split into 75% (Train) and 15% (Validation)
# Target ratio: 15% / 90% = 0.16666...
val_size_ratio = 0.15 / 0.90 

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=val_size_ratio,  # ~16.67% of the remaining data (90% * 0.1667 = 15%)
    random_state=42
)

# Create DataFrames for each split, preserving original column names
train_dataframe = panda_object.DataFrame(X_train, columns=X.columns)
train_dataframe['target'] = y_train

val_dataframe = panda_object.DataFrame(X_val, columns=X.columns)
val_dataframe['target'] = y_val

test_dataframe = panda_object.DataFrame(X_test, columns=X.columns)
test_dataframe['target'] = y_test

# Save the files (using CSV is a common interchange format)
file_path = os.path.join(script_dir, 'DataSet','train_data.csv')
train_dataframe.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet','validation_data.csv')
val_dataframe.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet','test_data.csv')
test_dataframe.to_csv(file_path, index=False)