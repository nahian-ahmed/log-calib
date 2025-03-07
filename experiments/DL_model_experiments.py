"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTENANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from utils.metrics import compute_ece

# Define Deep Learning Models
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((input_dim // 2) * 16, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

# Train function
def train_dl_model(model, train_loader, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_dl_model(model, test_loader, device, model_name, calibration_method, results, n_bins=10):
    model.eval() if hasattr(model, "eval") else None  # Ensure compatibility with callable calibration functions
    y_pred_list, y_true_list, y_prob_list = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).long().squeeze()
            
            # Handle both standard PyTorch models and callable calibration functions
            if callable(model):  
                predicted_probs = model(inputs)  # Use calibrated function directly
            else:
                outputs = model(inputs)
                predicted_probs = torch.sigmoid(outputs).squeeze()
            
            predicted = (predicted_probs > 0.5).long()
            
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
            y_prob_list.extend(predicted_probs.cpu().numpy())
    
    # Convert to 1D arrays
    y_true_list = np.array(y_true_list).ravel().astype(int)
    y_pred_list = np.array(y_pred_list).ravel().astype(int)
    y_prob_list = np.array(y_prob_list).ravel()
    
    # Ensure probability values are within [0,1]
    y_prob_list = np.clip(y_prob_list, 0, 1)
    
    ece = compute_ece(y_true_list, y_prob_list, n_bins)
    bce = brier_score_loss(y_true_list, y_prob_list)
    
    results.append({
        "Classifier": model_name,
        "Calibration Method": calibration_method,
        "Accuracy": accuracy_score(y_true_list, y_pred_list),
        "Precision": precision_score(y_true_list, y_pred_list, average='binary', zero_division=1),
        "Recall": recall_score(y_true_list, y_pred_list, average='binary', zero_division=1),
        "F1-score": f1_score(y_true_list, y_pred_list, average='binary', zero_division=1),
        "ECE": ece,
        "BCE": bce
    })
    
    return results


def collect_logits_labels(model, loader, device):
    """Collects logits and true labels for calibration."""
    model.eval()
    logits_list, labels_list = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            logits_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    return np.array(logits_list), np.array(labels_list)

def platt_scaling(model, loader, device):
    """Applies Platt Scaling (Logistic Regression) to the model's outputs."""
    logits, labels = collect_logits_labels(model, loader, device)
    
    lr = LogisticRegression()
    lr.fit(logits.reshape(-1, 1), labels)
    
    def calibrated_model(x):
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
        return torch.tensor(lr.predict_proba(probs)[:, 1], dtype=torch.float32).to(device)
    
    return calibrated_model

def logistic_calibration(model, loader, device):
    """Applies Logistic Calibration using CalibratedClassifierCV."""
    logits, labels = collect_logits_labels(model, loader, device)
    
    base_lr = LogisticRegression()
    calibrated_clf = CalibratedClassifierCV(base_lr, method='sigmoid')
    calibrated_clf.fit(logits.reshape(-1, 1), labels)
    
    def calibrated_model(x):
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
        return torch.tensor(calibrated_clf.predict_proba(probs)[:, 1], dtype=torch.float32).to(device)
    
    return calibrated_model

def isotonic_regression(logits, labels):
    """Applies Isotonic Regression for calibration."""
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(logits, labels)
    return iso_reg

def run_DL_models(data, classifiers):
    x_train, y_train = data["x_train_transformed"], data["y_train"]
    x_calibration, y_calibration = data["x_calibration_transformed"], data["y_calibration"]
    x_test, y_test = data["x_test_transformed"], data["y_test"]
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
    calibration_loader = DataLoader(TensorDataset(torch.tensor(x_calibration, dtype=torch.float32), torch.tensor(y_calibration, dtype=torch.long)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=32, shuffle=False)
    
    models = {"Convolutional Neural Network": CNNModel(input_dim=x_train.shape[1]), "Long Short-Term Memory": LSTMModel(input_dim=x_train.shape[1])}
    
    for clf_name, model in models.items():
        if clf_name not in classifiers:
            continue
        
        model.to(device)
        train_dl_model(model, train_loader, device)
        
        # Collect logits and labels from the calibration set
        logits, labels = collect_logits_labels(model, calibration_loader, device)
        
        # Define calibration methods
        calibration_methods = {
            "No Calibration": model,
            "Platt Scaling": platt_scaling(model, calibration_loader, device),
            "Logistic Calibration": logistic_calibration(model, calibration_loader, device),
            "Isotonic Calibration": lambda x: torch.tensor(
                isotonic_regression(logits, labels).predict(torch.sigmoid(model(x)).cpu().numpy()),
                dtype=torch.float32
            ).to(device)
        }
        
        for cal_name, calibrated_model in calibration_methods.items():
            if callable(calibrated_model):  # If it's a function, apply it to the model
                final_model = calibrated_model
            else:
                final_model = calibrated_model  # No calibration, use original model
            
            results = evaluate_dl_model(final_model, test_loader, device, clf_name, cal_name, results)
    
    return results
