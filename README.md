<!-- 
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
-->

# Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

## Overview
This project extracts features from HDFS logs and trains five models for anomaly detection:
- Logistic Regression
- Decision Tree
- Support Vector Machine
- Convolutional Neural Network
- Long Short-Term Memory

Calibration methods:
- Platt scaling
- Logistic calibration
- Isotonic calibration

## Dataset
The script expects **HDFS_100K.log_structured.csv** and **anomaly_label.csv** inside the `data/` directory.

## Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch loglizer
   ```
2. Run the script:
   ```bash
   python run_experiments.py
   ```

## Output
- Classification reports for all models.
