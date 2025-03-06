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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# Load data
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=TRAIN_RATIO)

# Feature extraction
feature_extractor = preprocessing.FeatureExtractor()
x_train_transformed = feature_extractor.fit_transform(x_train, term_weighting="tf-idf", normalization="zero-mean")
x_test_transformed = feature_extractor.transform(x_test)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_transformed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test_transformed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(random_state=RANDOM_STATE)
}

# Define Deep Learning Models
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((input_dim // 2) * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

# Train ML models
results = []
for name, model in classifiers.items():
    model.fit(x_train_transformed, y_train)
    y_pred = model.predict(x_test_transformed)
    results.append({
        "Classifier": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred, zero_division=1),
        "F1-score": f1_score(y_test, y_pred, zero_division=1)
    })

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_model(model, train_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader, device, model_name):
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
    
    results.append({
        "Classifier": model_name,
        "Accuracy": accuracy_score(y_true_list, y_pred_list),
        "Precision": precision_score(y_true_list, y_pred_list, zero_division=1),
        "Recall": recall_score(y_true_list, y_pred_list, zero_division=1),
        "F1-score": f1_score(y_true_list, y_pred_list, zero_division=1)
    })

def run_deep_learning_models():
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        "CNN": CNNModel(input_dim=x_train_transformed.shape[1]),
        "LSTM": LSTMModel(input_dim=x_train_transformed.shape[1])
    }
    
    for name, model in models.items():
        train_model(model, train_loader, device)
        evaluate_model(model, test_loader, device, name)

run_deep_learning_models()

# Save results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"), index=False)