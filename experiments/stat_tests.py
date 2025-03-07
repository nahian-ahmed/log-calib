"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTENANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

import os
import pandas as pd
from scipy.stats import wilcoxon

def run_stat_sig_tests(RESULTS_DIR):
    # Load the dataset
    file_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    df = pd.read_csv(file_path)

    # Selecting performance metrics columns
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ECE", "BCE"]

    # Filtering out "No Calibration" and grouping by classifier
    calibration_methods = df["Calibration Method"].unique()
    no_calibration_df = df[df["Calibration Method"] == "No Calibration"]

    # Wilcoxon test results
    wilcoxon_results = {}

    for method in calibration_methods:
        if method == "No Calibration":
            continue  # Skip the baseline

        method_df = df[df["Calibration Method"] == method]
        
        # Ensuring matched pairs by aligning indices
        merged_df = no_calibration_df.merge(method_df, on="Classifier", suffixes=("_no_calib", f"_{method}"))

        wilcoxon_results[method] = {}
        
        for metric in metrics:
            stat, p_value = wilcoxon(merged_df[f"{metric}_{method}"], merged_df[f"{metric}_no_calib"])
            wilcoxon_results[method][metric] = {"W-statistic": stat, "p-value": p_value}

    # Convert results to a DataFrame and save
    stat_test_df = pd.DataFrame.from_dict({(i, j): wilcoxon_results[i][j] 
                                        for i in wilcoxon_results.keys() 
                                        for j in wilcoxon_results[i].keys()}, 
                                        orient='index')

    # Save results to CSV
    stat_test_df.to_csv(os.path.join(RESULTS_DIR, "stat_tests.csv"))

    # Display results
    print(stat_test_df)