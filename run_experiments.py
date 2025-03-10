"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTENANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""


import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from experiments.preprocess import preprocess_data
from experiments.ML_model_experiments import run_ML_models
from experiments.DL_model_experiments import run_DL_models
from experiments.stat_tests import run_stat_sig_tests

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "data/HDFS_100k.log_structured.csv")
LABEL_FILE = os.path.join(BASE_DIR, "data/anomaly_label.csv")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

########
# 1. Preprocess Data
########
data = preprocess_data(BASE_DIR, LOG_FILE, LABEL_FILE, PREPROCESSED_DIR, RESULTS_DIR)

x_train_transformed = data["x_train_transformed"]
y_train = data["y_train"]
x_calibration_transformed = data["x_calibration_transformed"]
y_calibration = data["y_calibration"]
x_test_transformed = data["x_test_transformed"]
y_test = data["y_test"]


########
# 2. Run ML models
########

RANDOM_STATE = 1  # Fixed random state for reproducibility

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE)
}


# Define calibration methods
calibration_methods = {
    "No Calibration": lambda model: model,
    "Platt Scaling": lambda model: CalibratedClassifierCV(estimator=model, method="sigmoid"),
    "Logistic Calibration": lambda model: CalibratedClassifierCV(model, method="sigmoid", cv=5),
    "Isotonic Calibration": lambda model: CalibratedClassifierCV(model, method="isotonic", cv=5)
}

ML_results = run_ML_models(data, classifiers, calibration_methods)

########
# 3. Run DL Models
########

# Define classifiers
DL_classifiers = {
    "Convolutional Neural Network",
    "Long Short-Term Memory"
}

DL_results = run_DL_models(data, DL_classifiers)


# Print and save results
results_df = pd.concat([pd.DataFrame(ML_results), pd.DataFrame(DL_results)], ignore_index=True) 
print(results_df)
results_df.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"), index=False)


########
# 4. Run statistical tests
########

run_stat_sig_tests(RESULTS_DIR)