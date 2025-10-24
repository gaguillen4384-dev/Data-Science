# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import pandas as panda_object
import numpy as numpy_object
import matplotlib.pyplot as plot_object
import seaborn as graphics_object
from scipy import stats

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
    file_path = os.path.join(script_dir, 'DataSet','train.csv')
    
    # Set the style for the plots
    graphics_object.set_style("whitegrid")    
    
    # Prepare files structure to be used and load data to memory
    data = load_data(file_path)
    folder_path = os.path.join(script_dir, 'DataSet', 'EDA_Results')
    os.makedirs(folder_path, exist_ok=True)
    return data, folder_path



# Main Function of Script
if __name__ == "__main__":
    try:
        data, folder_path = initialize_script()
     

    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")