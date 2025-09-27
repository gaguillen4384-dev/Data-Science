# -*- coding: utf-8 -*-
"""
@author: gagui
"""

import os
import pandas as panda_object
import numpy as numpy_object
import matplotlib.pyplot as plot_object
import seaborn as graphics_object

def load_data(file_path):
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    return panda_object.read_csv(file_path)


def describe_with_interpretation(dataframe):
    """
    Performs a dataframe.describe() and prints an interpretation of the key statistics.
    """
    stats_dataframe = panda_object.DataFrame()

    for column in dataframe.columns:
        if panda_object.api.types.is_numeric_dtype(dataframe[column]):
            desc = dataframe[column].describe()
            column_stats = {
                'Count': int(desc['count']),
                'Mean': desc['mean'],
                'Std Dev': desc['std'],
                'Min': desc['min'],
                'Max': desc['max'],
                '25% Quartile': desc['25%'],
                'Median (50%)': desc['50%'],
                '75% Quartile': desc['75%']
            }
            stats_dataframe = panda_object.concat([stats_dataframe, panda_object.DataFrame(column_stats, index=[column])])
        else:
            # Handle non-numeric columns
            non_numeric_stats = {
                'Count': dataframe[column].count(),
                'Mean': 'N/A',
                'Std Dev': 'N/A',
                'Min': 'N/A',
                'Max': 'N/A',
                '25% Quartile': 'N/A',
                'Median (50%)': 'N/A',
                '75% Quartile': 'N/A'
            }
            stats_dataframe = panda_object.concat([stats_dataframe, panda_object.DataFrame(non_numeric_stats, index=[column])])

    return stats_dataframe


def plot_numerical_distributions(dataframe, file_path):
    """
    Identifies numerical columns in a DataFrame and generates a histogram for each
    to visualize its distribution, skewness, and multimodality.
    """

    # Get a list of only the numerical columns
    numerical_cols = dataframe.columns.tolist()

    # Determine the number of rows and columns for the subplots
    num_plots = len(numerical_cols)
    num_cols = 5  
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create the subplots
    fig, axes = plot_object.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axes = axes.flatten()
    
    first_col = numerical_cols.pop(0)
    last_col = numerical_cols.pop(-1)
    
    # Re-add them to the end of the list
    reordered_cols = numerical_cols + [first_col, last_col]

    # Iterate through the reordered columns and create a histogram for each
    for i, col in enumerate(reordered_cols):
        ax = axes[i]
        graphics_object.histplot(data=dataframe, x=col, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}', fontsize=16)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    # Save the figure as a PNG file
    plot_object.tight_layout()
    plot_object.savefig(file_path)
    plot_object.close() 
    
    
def plot_scatter(dataframe, x_col, y_col, file_path):
    """
    Creates a scatter plot to visualize the relationship between two numerical variables.
    """
    plot_object.figure(figsize=(8, 6))
    graphics_object.scatterplot(data=dataframe, x=x_col, y=y_col)
    plot_object.title(f'Scatter Plot of {y_col} vs {x_col}', fontsize=16)
    plot_object.xlabel(x_col, fontsize=12)
    plot_object.ylabel(y_col, fontsize=12)
    plot_object.savefig(file_path)
    plot_object.close() 


def aggregate_columns_names(list_of_names):
    '''
    Creates a array of unique simplified names for presentation
    Returns:
        an array of names
    '''
    prefixes_to_remove = ['wtd_', 'mean_', 'gmean_', 'entropy_', 'range_', 'std_']

    # Create an empty array
    name_mapping = []
    unique_properties = set() 
    
    # Loop through each column name
    for col in list_of_names:
        cleaned_name = col
        
        for prefix in prefixes_to_remove:
            if cleaned_name.startswith(prefix):
                cleaned_name = cleaned_name[len(prefix):]
        
        # Add the mapping to the array only if the cleaned name is unique
        if cleaned_name not in unique_properties:
            name_mapping.append(cleaned_name)
            unique_properties.add(cleaned_name)

        
    return name_mapping


def plot_box(dataframe, file_path):
    """
    Creates a box plot to compare the distribution of a numerical variable
    """
    chunk_size = 10
    num_chunks = (len(dataframe.columns) + chunk_size - 1) // chunk_size
    columns_names = aggregate_columns_names(dataframe.columns.tolist())
    og_file_path = file_path
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, len(dataframe.columns))
        chunk_cols = dataframe.columns[start_index:end_index]

        plot_object.figure(figsize=(15, 8))
        graphics_object.boxplot(data=dataframe[chunk_cols], orient='v')
        
        plot_object.title(f'Box Plots for {columns_names[i]}', fontsize=16)
        plot_object.xlabel('Column Names', fontsize=12)
        plot_object.ylabel('Value', fontsize=12)
        plot_object.xticks(rotation=45, ha='right')
        plot_object.tight_layout()
        
        file_path = os.path.join(og_file_path, f'{columns_names[i]}_plotbox.png')
        plot_object.savefig(file_path)
        plot_object.close() 
        
        
def group_by_columns(dataframe):
    '''
    Gives a simplified property name and the value in a list of all related columns.
    Returns:
        an dictionary of names
    '''
    prefixes_to_remove = ['wtd_', 'mean_', 'gmean_', 'entropy_', 'range_', 'std_']
    prop_mapping = {col: col for col in dataframe.columns}

    for col in dataframe.columns:
        cleaned_name = col
        for prefix in prefixes_to_remove:
            if cleaned_name.startswith(prefix):
                cleaned_name = cleaned_name[len(prefix):]
        prop_mapping[col] = cleaned_name

    groups = {prop: [] for prop in set(prop_mapping.values())}
    for col, prop in prop_mapping.items():
        groups[prop].append(col)       
        
    return groups


def plot_correlation_by_feature_heatmap(dataframe, file_path):
    """
    Generates a correlation matrix and visualizes it as a heatmap.
    """
    first_col = dataframe.columns[0]
    target_col = dataframe.columns[-1]
    groups = group_by_columns(dataframe)
    
    og_file_path = file_path
    # Plot a heatmap for each group
    for prop, cols in groups.items():
        if len(cols) > 1: # Only plot groups with more than one feature
            subset_cols = [first_col] + cols + [target_col]
            subset_dataframe = dataframe[subset_cols]
            
            plot_object.figure(figsize=(8, 7))
            graphics_object.heatmap(subset_dataframe.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
            plot_object.title(f'Correlation Heatmap for {prop.capitalize()} Properties', fontsize=16)
            plot_object.xticks(rotation=45, ha='right')
            plot_object.yticks(rotation=0)
            plot_object.tight_layout()
        file_path = os.path.join(og_file_path, f'{prop}_heatmap.png')
        plot_object.savefig(file_path)
        plot_object.close()




# Main Function of Script
if __name__ == "__main__":
    '''
    GETTO: clean up call and make this more modular
    GETTO: Initialize the script with a function
    GETTO: Only used what its needed for a run, running all at once is slow.
    '''
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)

        # Construct the full path to the file
        file_path = os.path.join(script_dir, 'DataSet','train.csv')
        
        # Set the style for the plots
        graphics_object.set_style("whitegrid")    
        
        
        data = load_data(file_path)
        folder_path = os.path.join(script_dir, 'DataSet', 'EDA_Results')
        os.makedirs(folder_path, exist_ok=True)
        '''
        # Start EDA process
        file_path = os.path.join(folder_path,'data_describe_results.csv')
        descriptive_summary = describe_with_interpretation(data)
        descriptive_summary.to_csv(file_path, index=False)
        '''

        # Distributions
        file_path = os.path.join(folder_path, 'numerical_distributions.png')
        plot_numerical_distributions(data,file_path)
        '''
        
        # Boxplots
        folder_path = os.path.join(script_dir, 'DataSet', 'EDA_Results', 'Boxplots')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path)
        
        first_cols = data.iloc[:, 0:1]
        plot_box(first_cols, file_path)
       
        target_cols = data.iloc[ :, [-1]]
        plot_box(target_cols, file_path)
        
        dataframe_subset = data.iloc[:, 1:-1]
        plot_box(dataframe_subset, file_path)
        
        # Heatmaps
        folder_path = os.path.join(script_dir, 'DataSet', 'EDA_Results', 'Heatmaps')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path)
        plot_correlation_by_feature_heatmap(data, file_path)
        '''

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")