"""
Model Calibration of Learning-Based Classifiers: A Case Study on Anomalous System Log Detection

CS 563 : SOFTWARE MAINTAINANCE AND EVOLUTION
Winter 2025
Oregon State University

Nahian Ahmed
"""

import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if np.any(in_bin):
            bin_acc = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
            bin_conf = np.mean(y_prob[in_bin])
            ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin) / len(y_true)
    return ece
