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


def split_into_features_and_target(dataframe):
    '''
    Split features from target given a dataset
    '''
    X = dataframe[['number_of_elements', 'wtd_entropy_atomic_mass',
                   'range_fie','wtd_entropy_atomic_radius',
                   'range_atomic_radius','wtd_std_atomic_radius',
                   'range_ThermalConductivity','std_ThermalConductivity',
                   'wtd_std_ThermalConductivity','entropy_Valence',
                   'wtd_std_fie','entropy_fie','wtd_entropy_FusionHeat',
                   'std_atomic_radius','entropy_atomic_radius','entropy_FusionHeat',
                   'entropy_atomic_mass','std_fie']]
    y = dataframe['critical_temp']
    return X, y


def calculate_models_stats(y_actual, y_pred, model, folder_path, model_type):
    '''
    Calculates the following: RMSE, the model summary, and MAE
    '''
    mean_squared_diff = mean_squared_error(y_actual, y_pred)
    rmse = numpy_object.sqrt(mean_squared_diff)
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    results = panda_object.DataFrame({
        'Metric': ['MSE','RMSE', 'MAE', 'MAPE'],
        'Value': [mean_squared_diff, rmse, mae, f'{mape * 100:.2f}']
    })    
    
    save_to_csv(folder_path, f'{model_type}_metrics.csv', results)
    
    # Extract key summary information
    if('glm' not in model_type):
        summary_data = {
            'Metric': ['R-squared','Null Deviance', 'Deviance', 'AIC', 
                       'BIC', 'log_likelihoods'],
            'Value': [r2_score(y_actual, y_pred), 
                      model.null_deviance, model.deviance,
                      model.aic, model.bic, model.llf]
        }
    else:
        summary_data = {
            'Metric': ['Null Deviance', 'Deviance', 'AIC', 
                       'BIC','log_likelihoods'],
            'Value': [model.null_deviance, model.deviance, 
                      model.aic, model.bic, model.llf]
        }
    summary_dataframe = panda_object.DataFrame(summary_data)

    save_to_csv(folder_path, f'{model_type}_summary_metrics.csv', summary_dataframe)
        
    
def plot_scatter(dataframe, y_validation, r2 ,folder_path, file_name):
    """
    Creates a scatter plot to visualize the relationship between two numerical variables.
    """
    plot_object.figure(figsize=(8, 6))
    plot_object.scatter(dataframe['Actual_Values'],
                        dataframe['Predicted_Values'],
                        alpha=0.7)
   
    # Add a perfect prediction line (y=x)
    plot_object.plot([y_validation.min(), y_validation.max()],
                     [y_validation.min(), y_validation.max()],
                     'r--', lw=2)
   
    # Add labels and a title
    plot_object.title('Actual vs. Predicted Values Plot')
    plot_object.xlabel('Actual Values')
    plot_object.ylabel('Predicted Values')
   
    # Add the R-squared score as text on the plot
    plot_object.text(0.05, 0.95, f'adjusted-R-squared = {r2:.4f}',
                     transform=plot_object.gca().transAxes,
                     fontsize=12, verticalalignment='top')
    plot_object.grid(True)
    
    # Save the plot
    file_path = os.path.join(folder_path,file_name)
    plot_object.tight_layout()
    plot_object.savefig(file_path)
    plot_object.close() 
    
    
def plot_residual_plots(y_actual, y_pred, folder_path, file_name):
    residuals = y_actual - y_pred

    plot_object.figure(figsize=(8, 6))
    plot_object.scatter(y_pred, residuals, alpha=0.7)
    plot_object.axhline(y=0, color='r', linestyle='--', lw=2)
    
    model_name = file_name.replace('.png', '')
    plot_object.title(f'{model_name}')
    plot_object.xlabel('Predicted Values')
    plot_object.ylabel('Residuals (Actual - Predicted)')
    plot_object.grid(True)
    # Save the plot
    file_path = os.path.join(folder_path,file_name)
    plot_object.tight_layout()
    plot_object.savefig(file_path)
    plot_object.close()


def fit_mlr_model(X, y, folder_path):
    '''
    Fits a split dataset into dataframes with a MLR model
    Return:
        A fitted model
    '''
    # Create and fit the OLS model or Gaussian distribution
    X_train_const = stats_model_api.add_constant(X)
    mlr = stats_model_api.GLM(y, X_train_const, 
                                    family=stats_model_api.families.Gaussian())
    mlr_model = mlr.fit()

    save_to_csv(folder_path,'mlr_coefficients.csv', mlr_model.summary2().tables[1])

    return mlr_model

    
def predict_with_mlr_model(mlr_model, X_validation, y_validation, folder_path):
    '''
    Predicts and outputs a y_pred vs y_actual
    '''
    X_validation_const = stats_model_api.add_constant(X_validation)
    y_pred = mlr_model.predict(X_validation_const)
    calculate_models_stats(y_validation, y_pred, mlr_model, folder_path, 'mlr')
    plot_residual_plots(y_validation, y_pred, folder_path,'mlr_residual_plot.png')


def fit_glm_poisson_model(X_train, y_train, folder_path):    
    '''
    Fits a split dataset into dataframes with a GLM poisson model
    Return:
        A fitted model
    '''
    X_train_const = stats_model_api.add_constant(X_train)
    glm_poisson = stats_model_api.GLM(y_train, X_train_const, 
                                    family=stats_model_api.families.Poisson())
    glm_poisson_model = glm_poisson.fit()

    save_to_csv(folder_path,'glm_poisson_coefficients.csv', glm_poisson_model.summary2().tables[1])
    return glm_poisson_model


def predict_with_glm_poisson_model(X_validation, y_validation, glm_poisson_model, folder_path):
    '''
    Predicts and outputs a set of metrics
    '''
    X_validation_const = stats_model_api.add_constant(X_validation)
    y_pred = glm_poisson_model.predict(X_validation_const)
    calculate_models_stats(y_validation, y_pred, glm_poisson_model,
                           folder_path, 'glm_poisson')
    plot_residual_plots(y_validation, y_pred, folder_path, 'glm_poisson_residual_plot.png')


def fit_glm_tweedie_model(X_train, y_train, folder_path):    
    '''
    Fits a split dataset into dataframes with a GLM tweedie model
    Return:
        A fitted model
    '''
    X_train_const = stats_model_api.add_constant(X_train)
    glm_tweedie = stats_model_api.GLM(y_train, X_train_const, 
                                    family=stats_model_api.families.Tweedie(var_power=1.5))
    glm_tweedie_model = glm_tweedie.fit()

    save_to_csv(folder_path,'glm_tweedie_coefficients.csv', glm_tweedie_model.summary2().tables[1])
    return glm_tweedie_model


def predict_with_glm_tweedie_model(X_validation, y_validation, glm_tweedie_model, folder_path):
    '''
    Predicts and out a set of metrics
    '''
    X_validation_const = stats_model_api.add_constant(X_validation)
    y_pred = glm_tweedie_model.predict(X_validation_const)
    calculate_models_stats(y_validation, y_pred, glm_tweedie_model,
                           folder_path, 'glm_tweedie')
    plot_residual_plots(y_validation, y_pred, folder_path, 'glm_tweedie_residual_plot.png')

        
def get_vif_from_given_features_set(dataframe, folder_path):
    '''
    Gives a VIF result from a given dataframe.
    '''
    X = stats_model_api.add_constant(dataframe)
    vif_data = panda_object.DataFrame()
    vif_data['features'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    file_path = os.path.join(folder_path, 'vif_with_model_features.csv')
    vif_data.to_csv(file_path, index=False)  
    

if __name__ == "__main__":
    try:
        folder_path, training_data = initialize_script()
        data_folder = get_data_folder()
        
        #normalizing training
        scaler = get_normalizer(training_data)
        normalized_training_data = normalize_data_set(training_data, scaler)
        X_train, y_train = split_into_features_and_target(normalized_training_data)
        
        #using same scaler with the training_data normalize scale
        validation_data = get_dataset_as_dataframe('ValidationSet.csv')
        normalized_validation_data = normalize_data_set(validation_data, scaler)
        X_validation, y_validation = split_into_features_and_target(normalized_validation_data)
        
        '''
        mlr_model = fit_mlr_model(X_train, y_train, folder_path)
        predict_with_mlr_model(mlr_model, X_validation, y_validation, folder_path)
        '''
        '''
        glm_poisson_model = fit_glm_poisson_model(X_train, y_train, folder_path)
        predict_with_glm_poisson_model(X_validation, y_validation,
                                     glm_poisson_model, folder_path)
        '''
        '''
        glm_tweedie_model= fit_glm_tweedie_model(X_train, y_train, folder_path)
        predict_with_glm_tweedie_model(X_validation, y_validation, 
                                        glm_tweedie_model, folder_path)
        '''
        
        
    except FileNotFoundError:
        print(f"Error: The file '{folder_path}' was not found.")