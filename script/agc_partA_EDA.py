import pandas as pd
import numpy as np

# Load data
data_path = "/home/yangze2065/Documents/autonomous_greenhouse_challenge_2024/dataset/4th_dwarf_tomato/image/train/ground_truth_train.json"  # Update path accordingly
data = pd.read_json(data_path)

# Display rows with NaN or Inf values
print(data[data.isin([np.nan, np.inf, -np.inf]).any(axis=1)])

# Option 1: Remove rows with NaN or Inf values
# cleaned_data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Option 2: Replace NaN or Inf values with a specified value, e.g., the mean of the column
for column in data.columns:
    if data[column].dtype == "float64":  # Check if the data type of the column is float
        data[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        data[column].fillna(data[column].mean(), inplace=True)

# Convert back to JSON if necessary, or directly use DataFrame in your training loop
data.to_json("cleaned_ground_truth_data.json")
