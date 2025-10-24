# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import os
import numpy as numpy_object
import pandas as panda_object
import matplotlib.pyplot as plot_object
import statsmodels.api as stats_model_api
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def load_data(file_path):
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    return panda_object.read_csv(file_path)


def get_data_folder():
    '''
    Returns the main data folder
    '''
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    folder_path = os.path.join(script_dir, 'DataSet')
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def get_dataset_as_dataframe(filename):
    '''
    Retruns a dataframe form given dataset
    '''
    folder_path = get_data_folder()
    file_path = os.path.join(folder_path, filename)
    data = load_data(file_path)
    return data


def initialize_script():
    """
    Sets up script
    Returns:
        A dataframe from the passed in data set.
        A folder path
    """
    # Prepare files structure to be used and load data to memory
    data_path = get_data_folder()
    folder_path = os.path.join(data_path,'Training_Results')
    os.makedirs(folder_path, exist_ok=True)
    data = get_dataset_as_dataframe('TrainingSet.csv')
    return folder_path, data

def save_to_csv(folder_path, file_name, dataframe):
    # Save the coefficients to a CSV file
    file_path = os.path.join(folder_path, file_name)
    dataframe.to_csv(file_path)


def get_normalizer(dataframe):
    '''
    Sets up a normalizer loaded with a dataframe fit
    '''
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    return scaler


def normalize_data_set(dataframe, scaler):
    '''
    Normalize the dataset
    '''
    normalized_data = scaler.transform(dataframe)
    normalized_data_enhanced = panda_object.DataFrame(normalized_data, columns=dataframe.columns)
    return  normalized_data_enhanced


    

if __name__ == "__main__":
    try:
        folder_path, training_data = initialize_script()
        data_folder = get_data_folder()
        
        #normalizing training
        scaler = get_normalizer(training_data)
        normalized_training_data = normalize_data_set(training_data, scaler)
       
        
        #using same scaler with the training_data normalize scale
        validation_data = get_dataset_as_dataframe('validation_data.csv')
        normalized_validation_data = normalize_data_set(validation_data, scaler)
      
        
        
    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")