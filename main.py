import os
import numpy as np
import pandas as pd
from loglizer import dataloader, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "data/HDFS_100k.log_structured.csv")
LABEL_FILE = os.path.join(BASE_DIR, "data/anomaly_label.csv")

# Set train-test ratio
TRAIN_RATIO = 0.8  # 80% training, 20% testing

# Load the data using dataloader.py
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=TRAIN_RATIO)

# Convert event sequences into numerical features using FeatureExtractor
feature_extractor = preprocessing.FeatureExtractor()
x_train_transformed = feature_extractor.fit_transform(x_train, term_weighting="tf-idf", normalization="zero-mean")
x_test_transformed = feature_extractor.transform(x_test)

# Save transformed dataset as CSV
np.savetxt(os.path.join(BASE_DIR, "preprocessed", "x_train_transformed.csv"), x_train_transformed, delimiter=",")
np.savetxt(os.path.join(BASE_DIR, "preprocessed", "x_test_transformed.csv"), x_test_transformed, delimiter=",")

# Train a Logistic Regression model
model = LogisticRegression(random_state=1)
model.fit(x_train_transformed, y_train)

# Predict on the test set
y_pred = model.predict(x_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save labels as CSV for further ML processing
train_df = pd.DataFrame({"EventSequence": x_train, "Label": y_train})
test_df = pd.DataFrame({"EventSequence": x_test, "Label": y_test})

train_df.to_csv(os.path.join(BASE_DIR, "preprocessed", "train.csv"), index=False)
test_df.to_csv(os.path.join(BASE_DIR, "preprocessed", "test.csv"), index=False)

print("Supervised ML dataset created and model trained successfully!")