# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import pandas as panda_object

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the full path to the file
file_path = os.path.join(script_dir, 'DataSet', 'train.csv')

# Load dataframe for file manipulation
dataframe = panda_object.read_csv(file_path)
total_rows = len(dataframe)
split_75_size = int(total_rows * 0.75)
split_15_size = int(total_rows * 0.15)
split_10_size = total_rows - split_75_size - split_15_size

# Split into three DataFrames
dataframe1 = dataframe.iloc[:split_75_size]
dataframe2 = dataframe.iloc[split_75_size:split_75_size + split_15_size]
dataframe3 = dataframe.iloc[split_75_size + split_15_size:]

# Save each DataFrame to a new CSV file
file_path = os.path.join(script_dir, 'DataSet', 'TrainingSet.csv')
dataframe1.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet', 'ValidationSet.csv')
dataframe2.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet', 'TestSet.csv')
dataframe3.to_csv(file_path, index=False)

