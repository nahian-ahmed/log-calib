# Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

## Overview
This project extracts an event occurrence matrix from HDFS logs and trains four models for anomaly detection:
- Logistic Regression
- Decision Tree
- Support Vector Machine
- Convolutional Neural Network

## Dataset
The script expects **HDFS_100K.log_structured.csv** and **anomaly_label.csv** inside the `data/` directory.

## Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch
   ```
2. Run the script:
   ```bash
   python main.py
   ```

## Output
- Classification reports for all models.
