# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import pandas as panda_object

def load_data(file_path):
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    return panda_object.read_csv(file_path)

def split_dataframe_by_wtd_columns(dataframe):
    """
    Splits a CSV file into multiple files based on columns with 'wtd_' in their name.
    
    Returns:
      tuple: A tuple containing two DataFrames:
             - dataframe_wtd_cols: DataFrame with columns containing 'wtd_'.
             - dataframe_other_cols: DataFrame with all other columns.
    """

    # Identify the columns with 'wtd_' in their name
    wtd_columns = [col for col in dataframe.columns if 'wtd_' in col]
    added_standard_columns = ['number_of_elements'] + wtd_columns + ['critical_temp']

    
    # Identify the other columns
    other_columns = [col for col in dataframe.columns if 'wtd_' not in col]
      
    # Create the two new DataFrames
    dataframe_wtd_cols = dataframe[added_standard_columns].copy()
    dataframe_other_cols = dataframe[other_columns].copy()
      
    return dataframe_wtd_cols, dataframe_other_cols


# Main Function of Script
if __name__ == "__main__":
    # The main workflow using the defined functions
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)

        # Construct the full path to the file
        file_path = os.path.join(script_dir, 'DataSet', 'TrainingSet.csv')
        
        data = load_data(file_path)
        
        dataframe_wtd_cols, dataframe_other_cols = split_dataframe_by_wtd_columns(data)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")