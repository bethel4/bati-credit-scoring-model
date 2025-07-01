import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data_processing import process_data

input_path = 'data/raw/data.csv'
output_path = 'data/processed/processed_data.csv'

# Run the processing pipeline
processed = process_data(input_path)

# If processed is a numpy array, convert to DataFrame
if not isinstance(processed, pd.DataFrame):
    processed = pd.DataFrame(processed)

processed.to_csv(output_path, index=False)
print(f'Processed data saved to {output_path}') 