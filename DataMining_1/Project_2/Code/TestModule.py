# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import os
import ModelBuildingModule as workhorse 

def initialize_script():
    """
    Sets up script
    Returns:
        A dataframe from the passed in data set.
        A folder path
    """
    # Prepare files structure to be used and load data to memory
    data_path = workhorse.get_data_folder()
    folder_path = os.path.join(data_path,'Test_Results')
    os.makedirs(folder_path, exist_ok=True)
    data = workhorse.get_dataset_as_dataframe('TrainingSet.csv')
    return folder_path, data


if __name__ == "__main__":
    try:
        folder_path, training_data = initialize_script()
        data_folder = workhorse.get_data_folder()
        
        #normalizing training
        scaler = workhorse.get_normalizer(training_data)
        normalized_training_data = workhorse.normalize_data_set(training_data, scaler)
        
        #using same scaler with the training_data normalize scale
        test_data = workhorse.get_dataset_as_dataframe('TestSet.csv')
        normalized_test_data = workhorse.normalize_data_set(test_data, scaler)
        
        
        
    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")