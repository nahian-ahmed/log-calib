"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from utils.metrics import compute_ece



def run_ML_models(data, classifiers, calibration_methods, n_bins=10):
    x_train_transformed = data["x_train_transformed"]
    y_train = data["y_train"]
    x_calibration_transformed = data["x_calibration_transformed"]
    y_calibration = data["y_calibration"]
    x_test_transformed = data["x_test_transformed"]
    y_test = data["y_test"]

    results = []
    for clf_name, model in classifiers.items():
        model.fit(x_train_transformed, y_train)
        for cal_name, cal_func in calibration_methods.items():
            cal_model = cal_func(model)
            if cal_name not in ["No Calibration"]:
                cal_model.fit(x_calibration_transformed, y_calibration)
            
            y_prob = cal_model.predict_proba(x_test_transformed)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            ece = compute_ece(y_test, y_prob, n_bins)
            bce = brier_score_loss(y_test, y_prob)

            results.append({
                "Classifier": clf_name,
                "Calibration Method": cal_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=1),
                "Recall": recall_score(y_test, y_pred, zero_division=1),
                "F1-score": f1_score(y_test, y_pred, zero_division=1),
                "ECE": ece,
                "BCE": bce
            })
    return results

