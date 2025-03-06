"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define Deep Learning Models
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear((input_dim // 2) * 16, 64)
        self.fc2 = nn.Linear(64, 2)

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
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

# Train and evaluate deep learning models with calibration
def train_dl_model(model, train_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_dl_model(model, test_loader, device, model_name, results):
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
        "Calibration Method": "N/A",
        "Accuracy": accuracy_score(y_true_list, y_pred_list),
        "Precision": precision_score(y_true_list, y_pred_list, zero_division=1),
        "Recall": recall_score(y_true_list, y_pred_list, zero_division=1),
        "F1-score": f1_score(y_true_list, y_pred_list, zero_division=1)
    })
    return results

def run_DL_models(data):

    x_train_transformed = data["x_train_transformed"]
    y_train = data["y_train"]
    x_calibration_transformed = data["x_calibration_transformed"]
    y_calibration = data["y_calibration"]
    x_test_transformed = data["x_test_transformed"]
    y_test = data["y_test"]

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train_transformed, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test_transformed, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=32, shuffle=False)
    models = {"CNN": CNNModel(input_dim=x_train_transformed.shape[1]), "LSTM": LSTMModel(input_dim=x_train_transformed.shape[1])}
    for name, model in models.items():
        train_dl_model(model, train_loader, device)
        results = evaluate_dl_model(model, test_loader, device, name, results)

    return results