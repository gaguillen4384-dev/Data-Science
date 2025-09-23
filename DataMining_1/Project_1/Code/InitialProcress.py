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

dataframe = panda_object.read_csv(file_path)
total_rows = len(dataframe)
rows_per_file = total_rows // 3

# Split into three DataFrames
dataframe1 = dataframe.iloc[:rows_per_file]
dataframe2 = dataframe.iloc[rows_per_file:2*rows_per_file]
dataframe3 = dataframe.iloc[2*rows_per_file:]

# Save each DataFrame to a new CSV file
file_path = os.path.join(script_dir, 'DataSet', 'TrainingSet.csv')
dataframe1.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet', 'ValidationSet.csv')
dataframe2.to_csv(file_path, index=False)

file_path = os.path.join(script_dir, 'DataSet', 'TestSet.csv')
dataframe3.to_csv(file_path, index=False)

