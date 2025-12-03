"""
@author: gagui
"""

import zipfile
import json
import time
import pandas as panda_object
import numpy as numpy_object
import matplotlib.pyplot as plot_object
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay

def load_data():
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    zip_path = "Data/data_processed.zip"

    with zipfile.ZipFile(zip_path, 'r') as z:
        X_train_std = panda_object.read_csv(z.open('X_train_std.csv'))
        Y_train = panda_object.read_csv(z.open('Y_train.csv'))
        X_test_std  = panda_object.read_csv(z.open('X_test_std.csv'))
        Y_test  = panda_object.read_csv(z.open('Y_test.csv'))
    return X_train_std, Y_train, X_test_std, Y_test

def ex_load_data():
    """
    Load dataframe for analysis
    Returns:
        A dataframe from the passed in data set.
    """
    zip_path = "Data/ex_data_processed.zip"

    with zipfile.ZipFile(zip_path, 'r') as z:
        X_train_std = panda_object.read_csv(z.open('ex_X_train_std.csv'))
        Y_train = panda_object.read_csv(z.open('ex_Y_train.csv'))
        X_test_std  = panda_object.read_csv(z.open('ex_x_test_std.csv'))
        Y_test  = panda_object.read_csv(z.open('ex_Y_test.csv'))
    return X_train_std, Y_train, X_test_std, Y_test


def save_confusion_matrix(y_true, y_pred, class_labels, filename):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plot_object.subplots(figsize=(8, 8))
    disp.plot(cmap=plot_object.cm.Blues, ax=ax, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix (Raw Counts)")
    plot_object.tight_layout()
    plot_object.savefig(filename, dpi=300)
    plot_object.close()

def save_per_class_roc_curve(y_test, y_prob, class_labels, filename):
    n_classes = len(class_labels)
    y_test_bin = label_binarize(y_test, classes=class_labels)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plot_object.figure(figsize=(10, 8))
    colors = plot_object.cm.get_cmap('viridis', n_classes)

    for i, label in enumerate(class_labels):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plot_object.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'ROC curve of class {label} (AUC = {roc_auc[i]:0.2f})')

    plot_object.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance') # Diagonal line
    plot_object.xlim([0.0, 1.0])
    plot_object.ylim([0.0, 1.05])
    plot_object.xlabel('False Positive Rate (FPR)')
    plot_object.ylabel('True Positive Rate (TPR)')
    plot_object.title('Per-Label Receiver Operating Characteristic (ROC) Curve')
    plot_object.legend(loc="lower right")
    plot_object.grid(True)
    plot_object.savefig(filename, dpi=300)
    plot_object.close()

def pre_tunning_log_lasso(X_train):
    '''
    Sets up the preprocessor and the Lasso-enabled Logistic Regression model.
    Lasso performs feature selection implicitly through L1 regularization.
    '''
    
    lr = LogisticRegression(
        solver='liblinear', 
        penalty='l1', # Use L1 penalty for Lasso feature selection
        C=0.1,
        random_state=42, 
        max_iter=1000
    )
    ovr_classifier = OneVsRestClassifier(lr)
    
    X_train_cat = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    X_train_num = X_train.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), X_train_cat), 
            ('num', 'passthrough', X_train_num) 
        ],
        remainder='passthrough'
    )
    
    result_object = {
        "preprocessor": preprocessor,
        "ovr_classifier": ovr_classifier,  
    }
    
    return result_object

def tunning_log_lasso(pre_tunning_result, X_train,y_train,X_test):
    '''
    Tunes the sfs model and returns processed x_train and x_test to use.
    '''

    preprocessor_pipeline = Pipeline(steps=[
        ('preprocessor', pre_tunning_result['preprocessor'])
    ])


    X_train_processed = preprocessor_pipeline.fit_transform(X_train, y_train)
    X_test_processed = preprocessor_pipeline.transform(X_test)
    
    result_object = {
        "feature_pipeline": preprocessor_pipeline,
        "X_train_processed": X_train_processed,  
        "X_test_processed": X_test_processed 
    }
    
    return result_object

def fit_final_log_model(pre_tunning_result,tunning_results, X_train, y_train):
    '''
    Tunes the Lasso regularization parameter C using RandomizedSearchCV.
    '''
    pipeline = Pipeline([
        ('preprocessor', tunning_results['feature_pipeline']['preprocessor']),
        ('classifier', pre_tunning_result['ovr_classifier'])
    ])

    # The regularization parameter for Logistic Regression is 'C'.
    # For L1 (Lasso), a smaller C means stronger regularization and more feature coefficients set to 0.
    param_dist = {
        'classifier__estimator__C': numpy_object.logspace(-4, 4, 10) # Tune C from 0.0001 to 10000
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10, 
        scoring='f1_weighted', # Metric for optimization
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1 
    )

    random_search.fit(X_train, y_train)
    
    return random_search

def store_non_zero_coefficients_to_json(coefficients_df, tolerance=1e-9):
    '''
    Filters the coefficients DataFrame to keep only features that have at least one 
    non-zero coefficient (across all classes) and saves the result to a JSON file.
    '''
    coeff_columns = [col for col in coefficients_df.columns if col.startswith('Coeff_')]

    is_non_zero = (coefficients_df[coeff_columns].abs() > tolerance).any(axis=1)

    non_zero_coefficients_df = coefficients_df[is_non_zero].copy()
    
    return non_zero_coefficients_df.to_dict(orient='records') 

def get_elasticnet_coefficients(random_search_result, X_train):
    '''
    Extracts and organizes the coefficients from the best Elastic Net Logistic Regression model.
    '''

    best_pipeline = random_search_result.best_estimator_

    preprocessor = best_pipeline.named_steps['preprocessor']

    X_train_cat = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(X_train_cat)
    
    num_features = X_train.select_dtypes(include=['number']).columns.tolist()

    feature_names = numpy_object.concatenate([ohe_features, num_features])

    ovr_classifier = best_pipeline.named_steps['classifier']

    class_labels = ovr_classifier.classes_

    coef_df = panda_object.DataFrame({'Feature': feature_names})
    
    for i, label in enumerate(class_labels):
        # The actual LogisticRegression model is the `estimator` of the OVR at index i
        log_reg_model = ovr_classifier.estimators_[i]
        
        # Get the coefficients (coef_ is a 2D array, but LogisticRegression has only one row here)
        coefficients = log_reg_model.coef_.flatten()
        
        # Add a column for this class's coefficients
        coef_df[f'Coeff_Class_{label}'] = coefficients

    return coef_df
    
def get_cv_metrics(random_search, tunning_results, X_train):
    '''
    Formats training metrics for report
    '''
    preprocessor = tunning_results['feature_pipeline'].named_steps['preprocessor']
    feature_names_processed = preprocessor.get_feature_names_out(X_train.columns.tolist())

    best_model = random_search.best_estimator_
    ovr_classifier_best = best_model.named_steps['classifier'] 
    
    selected_features = get_elasticnet_coefficients(random_search,X_train)
    selected_features_jsonfy = store_non_zero_coefficients_to_json(selected_features)

    cv_metrics ={
        'best_cv_f1_weighted': random_search.best_score_,
        'feature_names_processed': feature_names_processed,
        'selected_features': selected_features_jsonfy
    }
    
    return cv_metrics

def run_log_multiclass_classification(X_train,y_train,X_test,y_test):
    """
    Performs SFS, RandomizedSearchCV tuning, and evaluation for OVR Logistic Regression.
    """
    # --- tunning and fitting ---
    
    start_time = time.time()
    pre_tunning_result = pre_tunning_log_lasso(X_train)
    
    tunning_results = tunning_log_lasso(pre_tunning_result, X_train,y_train,X_test)
    
    random_search = fit_final_log_model(pre_tunning_result, tunning_results, X_train, y_train)
    end_time = time.time()
    
    # --- Predict on test data ---
    best_estimator = random_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    y_prob = best_estimator.predict_proba(X_test)

    # --- metric and plotting ---
    
    cv_metrics = get_cv_metrics(random_search, tunning_results,X_train)

    test_accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    total_time = end_time - start_time
    results = {
        "best_cv_f1_weighted": cv_metrics['best_cv_f1_weighted'],
        "f1_weighted_score": f1_weighted,
        "overall_test_accuracy": test_accuracy,
        "overall_error_rate": 1-test_accuracy,
        "total_training_tuning_time_seconds": total_time,
        "lasso_selected_features_count": len(cv_metrics['selected_features']),
        "lasso_selected_features_list": cv_metrics['selected_features']
    }

    with open('Results/log_lasso_classification_report.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    class_labels = numpy_object.unique(y_train)
    save_confusion_matrix(y_test, y_pred, class_labels,"Results/log_lasso_confusion_matrix.png")
    save_per_class_roc_curve(y_test, y_prob, class_labels, "Results/log_lasso_roc_curve.png")
    
    
if __name__ == "__main__":
    '''
    Every result file will be dumped into the directory its in.
    '''
    
    try:          
        X_train,y_train,X_test,y_test = load_data()
        
        run_log_multiclass_classification(X_train,y_train,X_test,y_test)   
        
        
    except FileNotFoundError:
        print(f"Error: The file was not found.")
