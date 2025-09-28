# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import os
import numpy as numpy_object
import pandas as panda_object
import matplotlib.pyplot as plot_object
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_data(file_path):
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    return panda_object.read_csv(file_path)

def initialize_script():
    """
    Sets up script
    Returns:
        A dataframe from the passed in data set.
        A folder path
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Construct the full path to the file
    file_path = os.path.join(script_dir, 'DataSet','TrainingSet.csv')
    
    # Prepare files structure to be used and load data to memory
    data = load_data(file_path)
    folder_path = os.path.join(script_dir, 'DataSet', 'Training_Results')
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, data

if __name__ == "__main__":
    ''' 
    GETTO: Normalize
    GETTO: Calculate VIF (to see correlation on variables chosen)
    GETTO: Create MLR model (maybe different versions)
    GETTO: Validate model against validation set
    '''
    try:
        folder_path, training_data = initialize_script()
        

    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")