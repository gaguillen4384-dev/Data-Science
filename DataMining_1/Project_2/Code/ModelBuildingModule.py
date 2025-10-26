# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import os
import time
import numpy as numpy_object
import pandas as panda_object
import matplotlib.pyplot as plot_object
import statsmodels.api as stats_model_api
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from joblib import dump


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
    data = get_dataset_as_dataframe('train_data.csv')
    return folder_path, data


def save_to_csv(folder_path, file_name, dataframe):
    # Save the coefficients to a CSV file
    file_path = os.path.join(folder_path, file_name)
    dataframe.to_csv(file_path)


def fit_scaler_with_training_set(dataframe):
    # Initialize the Scaler
    scaler = StandardScaler()
    scaler.fit(dataframe)
    return scaler

def split_into_x_y(dataframe):
    columns_to_drop = ['target']
    X = dataframe.drop(columns_to_drop, axis=1) # Features (columns to retain)
    y = dataframe['target'] 
    return X , y


def scale_data(scaler, X):
    # Transform the Training and Test data
    X_train_scaled = scaler.transform(X)
    dataframe_result = panda_object.DataFrame(X_train_scaled, columns=X.columns)
    return dataframe_result


def perform_stepwise_forward_with_bic_fit(X, y):
    """
     Performs forward stepwise regression using BIC as the selection criterion.

     Returns:
         stats_model_api.regression.linear_model.RegressionResultsWrapper: The final OLS model object.
     """
    # --- Start Timer ---
    start_time = time.time()
    
    selected_features = []
    candidate_features = list(X.columns)
    
    # 1. Calculate BIC for the Null Model (only intercept)
    null_model = stats_model_api.OLS(y, stats_model_api.add_constant(panda_object.DataFrame({'const': numpy_object.ones(len(y))}))).fit()
    current_bic = null_model.bic
    
    # 2. Main Forward Selection Loop
    while candidate_features:
        best_feature = None
        # Initialize improvement to 0 (BIC must decrease to improve)
        best_bic_decrease = 0 
        
        # Test each remaining candidate feature
        for feature in candidate_features:
            features_to_test = selected_features + [feature]
            
            # Fit OLS model and get BIC
            X_model = stats_model_api.add_constant(X[features_to_test])
            model = stats_model_api.OLS(y, X_model).fit()
            new_bic = model.bic
            
            # Check for improvement (decrease in BIC)
            if new_bic < current_bic:
                bic_decrease = current_bic - new_bic
                if bic_decrease > best_bic_decrease:
                    best_bic_decrease = bic_decrease
                    best_feature = feature
           
        # If the best feature improved BIC, add it
        if best_feature is not None:
            selected_features.append(best_feature)
            candidate_features.remove(best_feature)
            current_bic = current_bic - best_bic_decrease
        else:
            # Stop condition: No remaining feature improves the BIC
            break

    # 3. Final Model Fit and Timer Stop
    X_final = stats_model_api.add_constant(X[selected_features])
    final_model = stats_model_api.OLS(y, X_final).fit()
    
    end_time = time.time()
    total_time = end_time - start_time
    dataframe_results= panda_object.DataFrame({
       'training_time': [total_time],
       'features': [selected_features]
       })
    return final_model , dataframe_results


def predict_with_stepwise(model, dataframe_results, X_validation):
    selected_features = dataframe_results['features'].iloc[0]
    X_val_predictors = X_validation[selected_features]
    X_val_final = stats_model_api.add_constant(X_val_predictors, has_constant='add')

    predictions = model.predict(X_val_final)
    
    return predictions


def perform_ridge_fit(X,y):
    # The tuning parameter (lambda or alpha) controls regularization strength.
    # Searching on a logarithmic scale is critical.
    # --- Start Timer ---
    start_time = time.time()
    
    # Hyperparameter Grid ($\Lambda$):
    param_grid = {
        # 'ridge__alpha' links to the 'ridge' step in the Pipeline
        'ridge__alpha': numpy_object.logspace(-4, 4, 9) # 9 values from 10^-4 to 10^4
    }
    
    # K-fold cross-validation is performed only on the TRAINING data.
    K_FOLDS_USE = 5
    
    # Define the model pipeline: Standardize features -> Apply Ridge
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Standardize features
        ('ridge', Ridge(random_state=42))            # Step 2: Apply the Ridge model
    ])
    
    # Define the K-Fold object for cross-validation on the training data
    kf = KFold(n_splits=K_FOLDS_USE, shuffle=True, random_state=42)
    
    # We use the combined pipeline to search for the best alpha.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_mean_squared_error', 
        n_jobs=-1 
    )
    
    grid_search.fit(X, y)
    end_time = time.time()
    total_time = end_time - start_time
    results = grid_search.cv_results_
    tested_alphas = results['param_ridge__alpha']
    optimal_alpha = grid_search.best_params_['ridge__alpha']
    dataframe_results = panda_object.DataFrame({
        'OptimalAlpha': [optimal_alpha],
        'Hyperparameters': [tested_alphas],
        'training_time': [total_time],
    })
    final_model = grid_search.best_estimator_
    return final_model, dataframe_results

def predict_with_model(model,  X_validation):
    predictions = model.predict(X_validation)
    return predictions


def perform_lasso_fit(X,y):
    # The tuning parameter (lambda or alpha) controls regularization strength.
    # Searching on a logarithmic scale is critical.
    # --- Start Timer ---
    start_time = time.time()
    
    # Hyperparameter Grid ($\Lambda$):
    param_grid = {
        'alpha': numpy_object.logspace(-3, 1, 9) # Search range from 0.001 to 10.0
    }
    
    # K-fold cross-validation is performed only on the TRAINING data.
    K_FOLDS_USE = 5

    # Define the K-Fold object for cross-validation on the training data
    kf = KFold(n_splits=K_FOLDS_USE, shuffle=True, random_state=42)
    
    # Initialize the Lasso Model
    # Set max_iter higher for stability with many features
    lasso = Lasso(max_iter=50000, tol=0.001, random_state=42)
    
    # Initialize RandomizedSearchCV
    # 'neg_mean_squared_error' is standard for regression tuning (maximizing negative MSE minimizes MSE)
    random_search = GridSearchCV(
        estimator=lasso,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=0,
        n_jobs=-1, 
    )
    
    random_search.fit(X, y)
    end_time = time.time()
    final_model = random_search.best_estimator_
    total_time = end_time - start_time
    results = random_search.cv_results_
    tested_alphas = results['param_alpha']
    optimal_alpha = random_search.best_params_['alpha']
    zero_coefs = numpy_object.sum(final_model.coef_ == 0)
    non_zero_coefs = X.shape[1] - zero_coefs
    dataframe_results = panda_object.DataFrame({
        'OptimalAlpha': [optimal_alpha],
        'Hyperparameters': [tested_alphas],
        'NumberOfZeroCoefs': zero_coefs,
        'NumberOfNonZeroCoefs': non_zero_coefs,
        'training_time': [total_time],
    })
    return final_model, dataframe_results


def perform_elastic_net_fit(X,y):
    # The tuning parameter (lambda or alpha) controls regularization strength.
    # Searching on a logarithmic scale is critical.
    # --- Start Timer ---
    start_time = time.time()
    
    # Hyperparameter Grid ($\Lambda$):
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 
        'l1_ratio': [0.1, 0.5, 1.0]
    }
    
    # K-fold cross-validation is performed only on the TRAINING data.
    K_FOLDS_USE = 5

    # Define the K-Fold object for cross-validation on the training data
    kf = KFold(n_splits=K_FOLDS_USE, shuffle=True, random_state=42)
    
    # Initialize the Model
    base_model = ElasticNet(max_iter=5000, random_state=42)
    
    # Initialize RandomizedSearchCV
    # 'neg_mean_squared_error' is standard for regression tuning (maximizing negative MSE minimizes MSE)
    random_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=0,
        n_jobs=-1, 
    )
    
    random_search.fit(X, y)
    end_time = time.time()
    final_model = random_search.best_estimator_
    total_time = end_time - start_time
    results = random_search.cv_results_
    tested_alphas = results['param_alpha']
    optimal_alpha = random_search.best_params_['alpha']
    optimal_l1_ratio = random_search.best_params_['l1_ratio']
    zero_coefs = numpy_object.sum(final_model.coef_ == 0)
    non_zero_coefs = X.shape[1] - zero_coefs
    dataframe_results = panda_object.DataFrame({
        'OptimalAlpha': [optimal_alpha],
        'OptimalL1Ratio': [optimal_l1_ratio],
        'Hyperparameters': [tested_alphas],
        'NumberOfZeroCoefs': zero_coefs,
        'NumberOfNonZeroCoefs': non_zero_coefs,
        'training_time': [total_time],
    })
    return final_model, dataframe_results

def perform_pcr_fit(X,y):
    # The tuning parameter (lambda or alpha) controls regularization strength.
    # Searching on a logarithmic scale is critical.
    # --- Start Timer ---
    start_time = time.time()
    
    # Hyperparameter Grid ($\Lambda$):
    max_components = min(X.shape) - 1
    
    # Define the grid of k_comps to test.
    k_comps_grid = [1, 5, 10, 50, 100, 200, 300, 400, 500] 
    
    # Ensure grid points don't exceed the max allowed components
    k_comps_grid = [k for k in k_comps_grid if k <= max_components]
    
    # Create the hyperparameter grid dictionary for GridSearchCV
    hyperparameter_grid = {
        'pca__n_components': k_comps_grid
    }
    # K-fold cross-validation is performed only on the TRAINING data.
    K_FOLDS_USE = 5

    # Define the K-Fold object for cross-validation on the training data
    kf = KFold(n_splits=K_FOLDS_USE, shuffle=True, random_state=42)
    
    pcr_pipeline = Pipeline(steps=[
    ('pca', PCA()),                    # Perform PCA
    ('regressor', LinearRegression())  # Linear Regression on components
    ])
    
    random_search = GridSearchCV(
        estimator=pcr_pipeline,
        param_grid=hyperparameter_grid,
        cv=kf,
        scoring='neg_mean_squared_error', # or 'r2'
        n_jobs=-1,
        verbose=0
    )
    
    random_search.fit(X, y)
    end_time = time.time()
    final_model = random_search.best_estimator_
    total_time = end_time - start_time
    optimal_k_comps = random_search.best_params_['pca__n_components']
    dataframe_results = panda_object.DataFrame({
        'OptiomalKcomps': [optimal_k_comps],
        'Hyperparameters': [hyperparameter_grid],
        'training_time': [total_time],
    })
    return final_model, dataframe_results

def perform_pls_fit(X,y):
    # The tuning parameter (lambda or alpha) controls regularization strength.
    # Searching on a logarithmic scale is critical.
    # --- Start Timer ---
    start_time = time.time()
    
    # Hyperparameter Grid ($\Lambda$):
    max_components = min(X.shape) - 1
    
    # Define the grid of k_comps to test.
    k_comps_grid =[1, 5, 10, 50, 100, 200, 300, 400, 500] 
    
    # Ensure grid points don't exceed the max allowed components
    k_comps_grid = [k for k in k_comps_grid if k <= max_components]
    
    # Create the hyperparameter grid dictionary for GridSearchCV
    hyperparameter_grid = {
        'pls__n_components': k_comps_grid
    }
    # K-fold cross-validation is performed only on the TRAINING data.
    K_FOLDS_USE = 5

    # Define the K-Fold object for cross-validation on the training data
    kf = KFold(n_splits=K_FOLDS_USE, shuffle=True, random_state=42)
    
    pls_pipeline = Pipeline(steps=[
    ('pls', PLSRegression(scale=False))
    ])
    
    random_search = GridSearchCV(
        estimator= pls_pipeline,
        param_grid=hyperparameter_grid,
        cv=kf,
        scoring='neg_mean_squared_error', # or 'r2'
        n_jobs=-1,
        verbose=0
    )
    
    random_search.fit(X, y)
    end_time = time.time()
    final_model = random_search.best_estimator_
    total_time = end_time - start_time
    optimal_k_comps = random_search.best_params_['pls__n_components']
    dataframe_results = panda_object.DataFrame({
        'OptiomalKcomps': [optimal_k_comps],
        'Hyperparameters': [hyperparameter_grid],
        'training_time': [total_time],
    })
    return final_model, dataframe_results


def calculate_Q_2(y_true,y_predicted):
    predicted_residuals = y_true - y_predicted
    PRESS = numpy_object.sum(predicted_residuals**2)

    # 2. Calculate SSY (Total Sum of Squares)
    # This is the total variance in the true values.
    mean_y_true = numpy_object.mean(y_true)
    SSY = numpy_object.sum((y_true - mean_y_true)**2)

    # Handle the case where SSY is zero (all y_true values are the same)
    if SSY == 0:
        return 1.0 # If there's no variance, a perfect prediction yields 1.0

    # 3. Calculate Q2
    Q2 = 1 - (PRESS / SSY)

    return Q2

def mse_to_json_for_stepwise_training(folder_path, name_of_model, y_true, y_predicted, dataframe_results, model):
    mse_value = mean_squared_error(y_true, y_predicted)
    mae_value = mean_absolute_error(y_true, y_predicted)
    q_2 = calculate_Q_2(y_true, y_predicted)
    number_of_features = len(dataframe_results['features'].iloc[0])
    results_df = panda_object.DataFrame({
    'TrainingTimeSeconds': dataframe_results['training_time'],
    'NumberOfFeatures': number_of_features,
    'MSEValue': [mse_value],
    "RMSEValue": [numpy_object.sqrt(mse_value)],
    "MAEValue": [mae_value],
    'Q_2Value':[q_2],
    'Features': dataframe_results['features']
    }, index=[0])
    file_path = os.path.join(folder_path, f"{name_of_model}.json")
    results_df.to_json(file_path, orient='records', indent=2)
    file_path = os.path.join(folder_path, f"{name_of_model}.joblib")
    dump(model, file_path)   
    
def mse_to_json_for_ridge_training(folder_path, name_of_model, y_true, y_predicted, dataframe_results, model):
    mse_value = mean_squared_error(y_true, y_predicted)
    mae_value = mean_absolute_error(y_true, y_predicted)
    q_2 = calculate_Q_2(y_true, y_predicted)
    results_df = panda_object.DataFrame({
    'TrainingTimeSeconds': dataframe_results['training_time'],
    'MSEValue': [mse_value],
    "RMSEValue": [numpy_object.sqrt(mse_value)],
    "MAEValue": [mae_value],
    'Q_2Value':[q_2],
    'OptimalAlpha': dataframe_results['OptimalAlpha'],
    'Hyperparameters': [dataframe_results['Hyperparameters']]
    }, index=[0])
    file_path = os.path.join(folder_path, f"{name_of_model}.json")
    results_df.to_json(file_path, orient='records', indent=2)
    file_path = os.path.join(folder_path, f"{name_of_model}.joblib")
    dump(model, file_path)   
    
def mse_to_json_for_penalized_training(folder_path, name_of_model, y_true, y_predicted, dataframe_results, model):
    mse_value = mean_squared_error(y_true, y_predicted)
    mae_value = mean_absolute_error(y_true, y_predicted)
    q_2 = calculate_Q_2(y_true, y_predicted)
    results_df = panda_object.DataFrame({
    'TrainingTimeSeconds': dataframe_results['training_time'],
    'MSEValue': [mse_value],
    "RMSEValue": [numpy_object.sqrt(mse_value)],
    "MAEValue": [mae_value],
    'Q_2Value':[q_2],
    'NumberOfZeroCoefs': dataframe_results['NumberOfZeroCoefs'],
    'NumberOfNonZeroCoefs': dataframe_results['NumberOfNonZeroCoefs'],
    'OptimalAlpha': dataframe_results['OptimalAlpha'],
    'Hyperparameters': [dataframe_results['Hyperparameters']]
    }, index=[0])
    file_path = os.path.join(folder_path, f"{name_of_model}.json")
    results_df.to_json(file_path, orient='records', indent=2)
    file_path = os.path.join(folder_path, f"{name_of_model}.joblib")
    dump(model, file_path)   
        
    
def mse_to_json_for_pca_training(folder_path, name_of_model, y_true, y_predicted, dataframe_results, model):
    mse_value = mean_squared_error(y_true, y_predicted)
    mae_value = mean_absolute_error(y_true, y_predicted)
    q_2 = calculate_Q_2(y_true, y_predicted)
    results_df = panda_object.DataFrame({
    'TrainingTimeSeconds': dataframe_results['training_time'],
    'MSEValue': [mse_value],
    "RMSEValue": [numpy_object.sqrt(mse_value)],
    "MAEValue": [mae_value],
    'Q_2Value':[q_2],
    'OptiomalKcomps': dataframe_results['OptiomalKcomps'],
    'Hyperparameters': [dataframe_results['Hyperparameters']]
    }, index=[0])
    file_path = os.path.join(folder_path, f"{name_of_model}.json")
    results_df.to_json(file_path, orient='records', indent=2)
    file_path = os.path.join(folder_path, f"{name_of_model}.joblib")
    dump(model, file_path)    



if __name__ == "__main__":
    try:
        
        folder_path, training_data = initialize_script()
        
        data_folder = get_data_folder()
        
        # Prep scaling and training
        X_training, y_training = split_into_x_y(training_data)
        scaler = fit_scaler_with_training_set(X_training)        
        scaled_X_training = scale_data(scaler, X_training)
        
        # Prep validation
        validation_data = get_dataset_as_dataframe('validation_data.csv')
        X_validation, y_validation = split_into_x_y(validation_data)
        scaled_X_validation = scale_data(scaler, X_validation)

        
        '''
        # Predict with a Principal Component Regression
        model, dataframe_results = perform_pcr_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_model(model, scaled_X_validation)
        mse_to_json_for_pca_training(folder_path,"pcr_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
        
        '''
        # Predict with a Ridge Regression
        model, dataframe_results = perform_ridge_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_model(model, scaled_X_validation)
        mse_to_json_for_ridge_training(folder_path,"ridge_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
        
        '''
        # Predict with a Partial Least Square Regression
        model, dataframe_results = perform_pls_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_model(model, scaled_X_validation)
        mse_to_json_for_pca_training(folder_path,"pls_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
        
        '''
        # Predict with a Elastic Net Regression
        model, dataframe_results = perform_elastic_net_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_model(model, scaled_X_validation)
        mse_to_json_for_penalized_training(folder_path,"elastic_net_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
               
        '''
        # Predict with a Lasso Regression
        model, dataframe_results = perform_lasso_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_model(model, scaled_X_validation)
        mse_to_json_for_penalized_training(folder_path,"lasso_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
        
        '''
        # Predict with a forward stepwise BIC 
        model, dataframe_results = perform_stepwise_forward_with_bic_fit(scaled_X_training, y_training)
        predicted_y_validation = predict_with_stepwise(model, dataframe_results, scaled_X_validation)
        mse_to_json_for_stepwise_training(folder_path,"stepwise_BIC_regression",y_validation, predicted_y_validation, dataframe_results,model)
        '''
        
        
        
        
        
        
    except FileNotFoundError:
        print(f"Error: The file {folder_path} was not found.")