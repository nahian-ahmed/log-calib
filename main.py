"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

import os
import numpy as np
import pandas as pd
from loglizer import dataloader, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "data/HDFS_100k.log_structured.csv")
LABEL_FILE = os.path.join(BASE_DIR, "data/anomaly_label.csv")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set train-test ratio
TRAIN_RATIO = 0.8  # 80% training, 20% testing
RANDOM_STATE = 1  # Fixed random state for reproducibility

# Load the data using dataloader.py
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=TRAIN_RATIO)

# Print dataset information
print("DATASET:")
print(f"Number of training instances: {len(y_train)}")
print(f"Number of test instances: {len(y_test)}")

# Class distribution
train_class_distribution = pd.Series(y_train).value_counts().rename("Train Count")
test_class_distribution = pd.Series(y_test).value_counts().rename("Test Count")

num_train_anomalies = train_class_distribution.get(1, 0)
num_train_normal = train_class_distribution.get(0, 0)
num_test_anomalies = test_class_distribution.get(1, 0)
num_test_normal = test_class_distribution.get(0, 0)

print(f"Training set - Anomalous instances: {num_train_anomalies}, Normal instances: {num_train_normal}")
print(f"Test set - Anomalous instances: {num_test_anomalies}, Normal instances: {num_test_normal}")

train_anomaly_percentage = (num_train_anomalies / len(y_train)) * 100
test_anomaly_percentage = (num_test_anomalies / len(y_test)) * 100

print(f"Training set anomaly percentage: {train_anomaly_percentage:.2f}%")
print(f"Test set anomaly percentage: {test_anomaly_percentage:.2f}%")
print("\n")

class_distribution_df = pd.concat([train_class_distribution, test_class_distribution], axis=1).fillna(0)
class_distribution_df.to_csv(os.path.join(PREPROCESSED_DIR, "train_test_stats.csv"))

# Convert event sequences into numerical features using FeatureExtractor
feature_extractor = preprocessing.FeatureExtractor()
x_train_transformed = feature_extractor.fit_transform(x_train, term_weighting="tf-idf", normalization="zero-mean")
x_test_transformed = feature_extractor.transform(x_test)

# Save transformed dataset as CSV
np.savetxt(os.path.join(PREPROCESSED_DIR, "x_train_transformed.csv"), x_train_transformed, delimiter=",")
np.savetxt(os.path.join(PREPROCESSED_DIR, "x_test_transformed.csv"), x_test_transformed, delimiter=",")

# Define classifiers in a dictionary
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(random_state=RANDOM_STATE)
}

# Initialize results list
results = []

# Train and evaluate each model
print("RESULTS:")
for name, model in classifiers.items():
    model.fit(x_train_transformed, y_train)
    y_pred = model.predict(x_test_transformed)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    
    results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print("\n")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)

# Save labels as CSV for further ML processing
train_df = pd.DataFrame({"EventSequence": x_train, "Label": y_train})
test_df = pd.DataFrame({"EventSequence": x_test, "Label": y_test})

train_df.to_csv(os.path.join(PREPROCESSED_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(PREPROCESSED_DIR, "test.csv"), index=False)

print("Supervised ML dataset created and models trained successfully!")