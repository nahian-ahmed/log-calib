"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_ML_models (data, classifiers, calibration_methods):

    x_train_transformed = data["x_train_transformed"]
    y_train = data["y_train"]
    x_calibration_transformed = data["x_calibration_transformed"]
    y_calibration = data["y_calibration"]
    x_test_transformed = data["x_test_transformed"]
    y_test = data["y_test"]

    # Train and evaluate models
    results = []
    for clf_name, model in classifiers.items():
        model.fit(x_train_transformed, y_train)
        for cal_name, cal_func in calibration_methods.items():
            cal_model = cal_func(model)
            if cal_name not in ["No Calibration"]:
                cal_model.fit(x_calibration_transformed, y_calibration)
            y_pred = cal_model.predict(x_test_transformed)
            results.append({
                "Classifier": clf_name,
                "Calibration Method": cal_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=1),
                "Recall": recall_score(y_test, y_pred, zero_division=1),
                "F1-score": f1_score(y_test, y_pred, zero_division=1)
            })
    return results
