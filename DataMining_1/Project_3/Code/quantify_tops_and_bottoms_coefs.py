# -*- coding: utf-8 -*-
"""
@author: gagui
"""
import json
import pandas as panda_object
import numpy as numpy_object

try:
    with open('Results/log_ridge_classification_report.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: The file 'Results/log_ridge_classification_report.json' was not found.")
    # Exit or handle the error appropriately
    # For demonstration, we'll use an empty list if the file is not found
    data = {"ridge_selected_features_list": []}
    
features_list = data["ridge_selected_features_list"]


df = panda_object.DataFrame(features_list)

# Check if DataFrame is empty before proceeding
if df.empty:
    print("DataFrame is empty. No features to analyze.")
    results = {'positive_results': [], 'negative_results': []}
else:
    id_vars = ["Feature"]
    # Ensure 'Coeff_Class_' columns exist before slicing
    value_vars = [col for col in df.columns if col.startswith("Coeff_Class_")]
    
    if not value_vars:
        print("Error: No 'Coeff_Class_' columns found in the DataFrame.")
        results = {'positive_results': [], 'negative_results': []}
    else:

        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                            var_name="Coefficient Class", value_name="Coefficient Value")


        df_positive = df_melted.nlargest(10, "Coefficient Value")
        df_positive["Direction"] = "Most Positive"


        df_negative = df_melted.nsmallest(10, "Coefficient Value")
        df_negative["Direction"] = "Most Negative"

        df_results = panda_object.concat([df_positive, df_negative])
        df_results = df_results[["Direction", "Feature", "Coefficient Class", "Coefficient Value"]]
        df_results["Coefficient Value"] = df_results["Coefficient Value"].round(4)

        # Function to format the results for output
        def format_results(df_final, direction):
            # No change needed here, the original logic is sound
            subset = df_final[df_final["Direction"] == direction].drop(columns=["Direction"])
            result_list = subset.to_dict("records")
            
            # Convert list of dicts to a list of formatted strings
            formatted_list = [
                f"Feature: **{item['Feature']}** | Class: {item['Coefficient Class'].replace('Coeff_Class_', '')} | Value: **{item['Coefficient Value']}**"
                for item in result_list
            ]
            return formatted_list

        results = {
            'positive_results': format_results(df_results, "Most Positive"),
            'negative_results': format_results(df_results, "Most Negative")
        }

try:
    with open('Results/log_ridge_tops.json', 'w') as f:
        json.dump(results, f, indent=4)
except Exception as e:
    print(f"An error occurred while writing the output file: {e}")