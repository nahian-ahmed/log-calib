import os
from loglizer import dataloader
import pandas as pd

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "data/HDFS_100k.log_structured.csv")
LABEL_FILE = os.path.join(BASE_DIR, "data/anomaly_label.csv")

# Load the data using dataloader.py
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(LOG_FILE, LABEL_FILE, train_ratio=0.8)

# Convert data into a pandas DataFrame for easier ML use
train_df = pd.DataFrame({'EventSequence': x_train, 'Label': y_train})
test_df = pd.DataFrame({'EventSequence': x_test, 'Label': y_test})

# Save as CSV for further ML processing
train_df.to_csv(os.path.join(BASE_DIR, 'HDFS_train.csv'), index=False)
test_df.to_csv(os.path.join(BASE_DIR, 'HDFS_test.csv'), index=False)

print("Supervised ML dataset created successfully!")
print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")