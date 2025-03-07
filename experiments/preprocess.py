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
from loglizer import dataloader, preprocessing


# Set data split ratios
TRAIN_RATIO = 0.5
CALIBRATION_RATIO = 0.3
TEST_RATIO = 0.2

def preprocess_data (BASE_DIR, LOG_FILE, LABEL_FILE, PREPROCESSED_DIR, RESULTS_DIR):
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    (x_train, y_train), (x_remaining, y_remaining) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=TRAIN_RATIO)
    (x_calibration, y_calibration), (x_test, y_test) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=CALIBRATION_RATIO / (CALIBRATION_RATIO + TEST_RATIO))


    # Print dataset information
    print("DATASET:")
    print(f"Number of training instances: {len(y_train)}")
    print(f"Number of calibration instances: {len(y_calibration)}")
    print(f"Number of test instances: {len(y_test)}")

    # Class distribution
    train_class_distribution = pd.Series(y_train).value_counts().rename("Train Count")
    calibration_class_distribution = pd.Series(y_calibration).value_counts().rename("Calibration Count")
    test_class_distribution = pd.Series(y_test).value_counts().rename("Test Count")

    num_train_anomalies = train_class_distribution.get(1, 0)
    num_train_normal = train_class_distribution.get(0, 0)
    num_calibration_anomalies = calibration_class_distribution.get(1, 0)
    num_calibration_normal = calibration_class_distribution.get(0, 0)
    num_test_anomalies = test_class_distribution.get(1, 0)
    num_test_normal = test_class_distribution.get(0, 0)

    print(f"Training set - Anomalous instances: {num_train_anomalies}, Normal instances: {num_train_normal}")
    print(f"Calibration set - Anomalous instances: {num_calibration_anomalies}, Normal instances: {num_calibration_normal}")
    print(f"Test set - Anomalous instances: {num_test_anomalies}, Normal instances: {num_test_normal}")

    train_anomaly_percentage = (num_train_anomalies / len(y_train)) * 100
    calibration_anomaly_percentage = (num_calibration_anomalies / len(y_calibration)) * 100
    test_anomaly_percentage = (num_test_anomalies / len(y_test)) * 100

    print(f"Training set anomaly percentage: {train_anomaly_percentage:.2f}%")
    print(f"Calibration set anomaly percentage: {calibration_anomaly_percentage:.2f}%")
    print(f"Test set anomaly percentage: {test_anomaly_percentage:.2f}%")
    print("\n")

    class_distribution_df = pd.concat([train_class_distribution, calibration_class_distribution, test_class_distribution], axis=1).fillna(0)
    class_distribution_df.to_csv(os.path.join(PREPROCESSED_DIR, "dataset_stats.csv"))


    # Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train_transformed = feature_extractor.fit_transform(x_train, term_weighting="tf-idf", normalization="zero-mean")
    x_calibration_transformed = feature_extractor.transform(x_calibration)
    x_test_transformed = feature_extractor.transform(x_test)


    np.savetxt(os.path.join(PREPROCESSED_DIR, "x_train_transformed.csv"), x_train_transformed, delimiter=",")
    np.savetxt(os.path.join(PREPROCESSED_DIR, "x_calibration_transformed.csv"), x_calibration_transformed, delimiter=",")
    np.savetxt(os.path.join(PREPROCESSED_DIR, "x_test_transformed.csv"), x_test_transformed, delimiter=",")


    train_df = pd.DataFrame({"EventSequence": x_train, "Label": y_train})
    calibration_df = pd.DataFrame({"EventSequence": x_calibration, "Label": y_calibration})
    test_df = pd.DataFrame({"EventSequence": x_test, "Label": y_test})

    train_df.to_csv(os.path.join(PREPROCESSED_DIR, "train.csv"), index=False)
    calibration_df.to_csv(os.path.join(PREPROCESSED_DIR, "calibration.csv"), index=False)
    test_df.to_csv(os.path.join(PREPROCESSED_DIR, "test.csv"), index=False)

    # Save all datasets as a compressed .npz file
    np.savez(os.path.join(PREPROCESSED_DIR, "preprocessed_data.npz"),
            x_train_transformed=x_train_transformed, y_train=y_train,
            x_calibration_transformed=x_calibration_transformed, y_calibration=y_calibration,
            x_test_transformed=x_test_transformed, y_test=y_test)

    data = np.load(os.path.join(PREPROCESSED_DIR, "preprocessed_data.npz"), allow_pickle=True)

    return data