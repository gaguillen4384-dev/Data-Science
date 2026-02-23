# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import os
import numpy as numpy_object
import pandas as panda_object
import ModelBuildingModule as workhorse 
import statsmodels.api as stats_model_api
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    models_folder = os.path.join(data_path,'Training_Results')
    data = workhorse.get_dataset_as_dataframe('train_data.csv')
    return folder_path, models_folder,data


def get_loaded_model(models_folder, filename):
    model_path = os.path.join(models_folder, filename)
    return load(model_path)

def predict_stepwise_test(models_folder,scaled_X_test, y_test,folder_path):
    loaded_model = get_loaded_model(models_folder,'stepwise_BIC_regression.joblib')
    file_path = os.path.join(models_folder, 'stepwise_BIC_regression.json')
    dataframe_results = panda_object.read_json(file_path)
    selected_features = dataframe_results['Features'].iloc[0]
    X_val_predictors = scaled_X_test[selected_features]
    X_val_final = stats_model_api.add_constant(X_val_predictors, has_constant='add')
    y_predicted = loaded_model.predict(X_val_final)
    mse_to_json(folder_path,'stepwise_BIC_regression',y_test, y_predicted)

    
def predict_penalized_test(models_folder, model_name ,scaled_X_test, y_test,folder_path):
    loaded_model = get_loaded_model(models_folder,f'{model_name}.joblib')
    y_predicted = loaded_model.predict(scaled_X_test)
    mse_to_json(folder_path, f'{model_name}', y_test, y_predicted)
    

def predict_dimension_reduction_test(models_folder,model_name,scaled_X_test,y_test,folder_path):
    loaded_model = get_loaded_model(models_folder,f'{model_name}.joblib')
    y_predicted = loaded_model.predict(scaled_X_test)
    mse_to_json(folder_path, f'{model_name}', y_test, y_predicted)
    

def mse_to_json(folder_path, name_of_model, y_true, y_predicted):
    mse_value = mean_squared_error(y_true, y_predicted)
    mae_value = mean_absolute_error(y_true, y_predicted)
    q_2 = workhorse.calculate_Q_2(y_true, y_predicted)
    results_df = panda_object.DataFrame({
    'MSEValue': [mse_value],
    "RMSEValue": [numpy_object.sqrt(mse_value)],
    "MAEValue": [mae_value],
    'Q_2Value':[q_2]
    }, index=[0])
    file_path = os.path.join(folder_path, f"{name_of_model}.json")
    results_df.to_json(file_path, orient='records', indent=2)
    
    
def predict_with_model(folder_path, models_folder, training_data,data_folder):
    #standardizing training
    X_training, y_training = workhorse.split_into_x_y(training_data)
    scaler = workhorse.fit_scaler_with_training_set(X_training)    
    
    #using same scaler with the training_data standardizing scale
    test_data = workhorse.get_dataset_as_dataframe('test_data.csv')
    X_test, y_test = workhorse.split_into_x_y(test_data)
    scaled_X_test = workhorse.scale_data(scaler, X_test)
    '''
    # Test stepwise
    predict_stepwise_test(models_folder, scaled_X_test, y_test,folder_path)
    '''
    
    '''
    # Test Elastic Net
    predict_penalized_test(models_folder,'elastic_net_regression',scaled_X_test,y_test,folder_path)
    '''
    
    '''
    # Test 
    predict_penalized_test(models_folder,'lasso_regression',scaled_X_test,y_test,folder_path)
    '''
    
    '''
    # Test Ridge
    predict_penalized_test(models_folder,'ridge_regression',scaled_X_test,y_test,folder_path)
    '''
    
    '''
    # Test PLS
    predict_dimension_reduction_test(models_folder,'pls_regression',scaled_X_test,y_test,folder_path)
    '''

    '''
    # Test PCR
    predict_dimension_reduction_test(models_folder,'pcr_regression',scaled_X_test,y_test,folder_path)
    '''
    
    

if __name__ == "__main__":
    try:
        folder_path, models_folder, training_data = initialize_script()
        data_folder = workhorse.get_data_folder()
        
        '''
        # Predict by going to function and uncommenting, could be enhance to use a registry pattern
        predict_with_model(folder_path, models_folder, training_data,data_folder)
        ''' 
        
        
    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")