#!/bin/bash

# Get command line arguments
raw_data_file="$1"
processed_data_file="$2"
model_file="$3"
test_data_file="$4"
predictions_file="$5"

# Run data_processing.py
echo "Starting data processing..."
python src/data_processing.py --input_file="$raw_data_file" --output_file="$processed_data_file"

# Run model_training.py (XGBoost)
echo "Starting model training..."
python src/model_training.py --input_file="$processed_data_file" --model_file="$model_file" --model_type="xgboost"

# Run model_prediction.py (XGBoost)
echo "Starting prediction..."
python src/model_prediction.py --input_file="$test_data_file" --model_file="$model_file" --output_file="$predictions_file" --model_type="xgboost"

echo "Pipeline completed."