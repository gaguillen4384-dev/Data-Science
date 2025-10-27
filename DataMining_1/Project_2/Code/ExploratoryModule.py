# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import json
import pandas as panda_object
import numpy as numpy_object
import ModelBuildingModule as workhorse 
import matplotlib.pyplot as plot_object
import seaborn as sns

def load_data(file_path):
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    return panda_object.read_csv(file_path)


def initialize_script():
    data_path = workhorse.get_data_folder()
    test_folder = os.path.join(data_path,'Test_Results')
    validation_folder = os.path.join(data_path,'Training_Results')
    plots_folder = os.path.join(data_path,'Plots_Visuals')
    os.makedirs(plots_folder, exist_ok=True)

    # Set a consistent plotting style
    plot_object.style.use('ggplot')
    return test_folder, validation_folder,plots_folder

def get_dataframe_from_json_results(data_path):
    files_list = [
        'elastic_net_regression.json',
        'lasso_regression.json',
        'ridge_regression.json',
        'stepwise_BIC_regression.json',
        'pls_regression.json',
        'pcr_regression.json'
        ]
    
    all_data = []

    # Loop through files, load JSON, and append to a list
    for filename in files_list:
        file_path = os.path.join(data_path,filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                row = {}
                delimiter = '_regression'
                model_name = filename.split(delimiter)[0]
                row["Model"] = model_name
                if "TrainingTimeSeconds" in data[0]:
                    row["TrainingTimeSeconds"] = data[0]["TrainingTimeSeconds"]
                if "MSEValue" in data[0]:
                    row["MSEValue"] = data[0]["MSEValue"]
                if "RMSEValue" in data[0]:
                    row["RMSEValue"] = data[0]["RMSEValue"]
                if "MAEValue" in data[0]:
                    row["MAEValue"] = data[0]["MAEValue"]
                if "Q_2Value" in data[0]:
                    row["Q_2Value"] = data[0]["Q_2Value"]
                all_data.append(row)
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {filename}")
            continue
    
    result = panda_object.DataFrame(all_data)
    result['RMSE_MAE_Ratio'] = result['RMSEValue'] / result['MAEValue']
    return result

    
def plot_rmse_mae_ratio(plots_folder, dataframe, is_test):
    if(is_test):
        filename = os.path.join(plots_folder,'test_rmse_mae_ratio_barchart.png')
    else:
        filename = os.path.join(plots_folder,'validation_rmse_mae_ratio_barchart.png')
    # Sort by the ratio for consistent x-axis ordering
    df_sorted = dataframe.sort_values(by='RMSE_MAE_Ratio', ascending=False)
    
    models = df_sorted['Model']
    rmse = df_sorted['RMSEValue']
    mae = df_sorted['MAEValue']
    ratio = df_sorted['RMSE_MAE_Ratio']
    
    x = numpy_object.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    
    # Create the figure and primary axis (for RMSE and MAE bars)
    fig, ax1 = plot_object.subplots(figsize=(8, 6))
    
    # --- Plot 1: RMSE and MAE (Grouped Bar Chart on left axis) ---
    rects1 = ax1.bar(x - width/2, rmse, width, label='RMSE', color='#bcbcf2')
    rects2 = ax1.bar(x + width/2, mae, width, label='MAE', color='#EDBC64')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Error Value (RMSE and MAE)', color='#222222')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.tick_params(axis='y', labelcolor='#222222')
    
    # Adjust the y-limit for the error metrics
    max_error = df_sorted[['RMSEValue', 'MAEValue']].max().max()
    ax1.set_ylim(0, max_error * 1.1)
    
    # --- Plot 2: RMSE/MAE Ratio (Line Chart on right axis) ---
    ax2 = ax1.twinx() 
    
    line, = ax2.plot(x, ratio, color='black', marker='o', linestyle='--', linewidth=2, label='RMSE/MAE Ratio')
    ax2.set_ylabel('RMSE / MAE Ratio', color='#36454F')
    ax2.tick_params(axis='y', labelcolor='#36454F')
    
    # Set tighter y-limit for the ratio
    min_ratio = df_sorted['RMSE_MAE_Ratio'].min()
    max_ratio = df_sorted['RMSE_MAE_Ratio'].max()
    ax2.set_ylim(min_ratio * 0.999, max_ratio * 1.01)
    
    # --- Final Touches ---
    if(is_test):
        plot_object.title('Test Model Performance: Error Metrics and RMSE/MAE Ratio', fontsize=14)
    else:
        plot_object.title('Validation Model Performance: Error Metrics and RMSE/MAE Ratio', fontsize=14)

    
    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center')
    
    fig.tight_layout()
    plot_object.savefig(filename)
    plot_object.close()


def plot_q2_vs_rmse(plots_folder, dataframe, is_test):
    if(is_test):
        filename = os.path.join(plots_folder,'test_q2_vs_rmse_barchart.png')
    else:
        filename = os.path.join(plots_folder,'validation_q2_vs_rmse_barchart.png')
    fig, ax1 = plot_object.subplots(figsize=(8, 6))
    
    # Sort by RMSE for the bar chart base
    df_sorted_rmse = dataframe.sort_values(by='RMSEValue', ascending=True)
    models = df_sorted_rmse['Model']
    rmse_values = df_sorted_rmse['RMSEValue']
    q2_values = df_sorted_rmse['Q_2Value']
    
    # Bar chart for RMSE on ax1 (left y-axis)
    bar_colors = ['#6495ED']
    ax1.bar(models, rmse_values, color=bar_colors, alpha=0.6, label='RMSE Value',width=0.4)
    ax1.set_ylabel('RMSE Value (Bar)', color='black', fontsize=12)
    max_error = df_sorted_rmse['RMSEValue'].max()
    ax1.set_ylim(3, max_error * 1.1)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create a second y-axis for Q_2Value (dot line plot)
    ax2 = ax1.twinx()
    # Note the use of LaTeX for $Q^2$
    ax2.plot(models, q2_values, color='red', marker='o', linestyle='--', linewidth=2, label='$Q^2$ Value')
    ax2.set_ylabel('$Q^2$ Value (Dot Line)', color='red', fontsize=12)
    max_q2 = q2_values.max()
    min_q2 = q2_values.min()
    q2_range = max_q2 - min_q2
    padding = q2_range * 0.1 
    ax2.set_ylim(min_q2 - padding, max_q2 + padding)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Title and legend
    if(is_test):
        fig.suptitle('Test RMSE vs $Q^2$ Comparison Across Models', fontsize=14)
    else:
        fig.suptitle('Validation RMSE vs $Q^2$ Comparison Across Models', fontsize=14)
    
    ax1.set_xlabel('Model', fontsize=12)
    
    plot_object.grid(True, which='both', axis='y', linestyle='--')
    plot_object.tight_layout(rect=[0, 0, 0.9, 1])
    
    fig.tight_layout()
    plot_object.savefig(filename)
    plot_object.close()

def plot_time_rmse_per_model(plots_folder, dataframe, ):
    filename = os.path.join(plots_folder,'rmse_and_time_scatterplot.png')
        
    data = dataframe.sort_values(by='TrainingTimeSeconds')

    # --- Plotting ---
    plot_object.figure(figsize=(8, 6))

    # Scatter plot: X=Training Time, Y=RMSE, colored by Type (Training/Test)
    sns.scatterplot(
        data=data,
        x='TrainingTimeSeconds', 
        y='RMSE', 
        hue='Type',            # Differentiate points by Training/Test
        style='Type',          # Use different markers
        size='RMSE',           # Size points by RMSE
        sizes=(100, 300),      # Range of point sizes
        palette={'Training': 'dodgerblue', 'Test': 'red'}, # Custom colors
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Annotate each point with the model name
    for i, row in data.iterrows():
        # Offset text based on Type (Test points are above the model name, Training points are below)
        v_offset = 0.03 if row['Type'] == 'Test' else -0.03
        
        plot_object.annotate(
            row['Model'], 
            (row['TrainingTimeSeconds'] + 0.1, row['RMSE'] + v_offset), 
            fontsize=9,
            alpha=0.7,
            ha='left'
        )

    # --- Labels and Title ---
    plot_object.title('Model Efficiency-Accuracy Trade-Off (RMSE vs. Training Time)', fontsize=16)
    
    # X-axis: Training Time (Efficiency: lower is better)
    plot_object.xlabel('Training Time (seconds) - Efficiency $\\longrightarrow$ Worse', fontsize=12)
    
    # Y-axis: RMSE (Accuracy: lower is better)
    plot_object.ylabel('Root Mean Square Error (RMSE) - Accuracy $\\longrightarrow$ Better', fontsize=12)

    # --- Customizations ---
    plot_object.grid(True, linestyle='--', alpha=0.6)
    plot_object.gca().invert_yaxis()
    
    # --- Final Touches ---
    plot_object.title('Model Efficiency-Accuracy Trade-Off (RMSE vs. Training Time)', fontsize=16)

    plot_object.tight_layout()
    plot_object.savefig(filename)
    plot_object.close()

def plot_time_scatter(test_results,validation_results):
    merged = panda_object.merge(
    test_results, 
    validation_results, 
    on='Model', # The column to match on
    how='inner', # Only keep models that appear in both tables
    suffixes=('_Training', '_Test')
    )
    # Step 2: Melt the DataFrames (Long Format for Plotting)
    df_plot = merged.melt(
        id_vars=['Model', 'TrainingTimeSeconds'],
        value_vars=['RMSEValue_Training', 'RMSEValue_Test'],
        var_name='Type',          
        value_name='RMSE'         
    )
    
    # Clean up the 'Type' column values to be 'Training' and 'Test' for plotting
    df_plot['Type'] = df_plot['Type'].replace({'RMSEValue_Training': 'Training', 'RMSEValue_Test': 'Test'})
    plot_time_rmse_per_model(plots_folder, df_plot )

# Main Function of Script
if __name__ == "__main__":
    try:
        test_folder, validation_folder, plots_folder = initialize_script()
        
        test_results = get_dataframe_from_json_results(test_folder)  
        validation_results = get_dataframe_from_json_results(validation_folder)
        
        '''
        plot_rmse_mae_ratio(plots_folder, test_results, True)
        plot_rmse_mae_ratio(plots_folder, validation_results, False)
        '''
        
        '''
        plot_q2_vs_rmse(plots_folder, test_results, True)
        plot_q2_vs_rmse(plots_folder, validation_results, False) 
        '''
        
        
        plot_time_scatter(test_results,validation_results)
        

        



    except FileNotFoundError:
        print("Error: The file was not found.")