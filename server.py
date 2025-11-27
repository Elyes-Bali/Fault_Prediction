# app_regression.py

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
API_URL = "http://127.0.0.1:4000/api"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app) 

# Global variable to store the model
GRU_MODEL = None

# List containing ALL columns (34 items)
FEATURE_COLUMNS_WITH_DATETIME = ['datetime', 'FIC2003_valeur', 'LI2001_valeur', 'MX2001_valeur',
   'MX2002A_valeur', 'MX2002B_valeur', 'MX2002C_valeur', 'MX2002D_valeur',
   'MX2003A_valeur', 'MX2003B_valeur', 'MX2003C_valeur', 'temperature_ext',
   'TI2001__2_valeur', 'panne_MX2002A', 'hour_sin', 'hour_cos',
   'minute_sin', 'minute_cos', 'day_sin', 'day_cos', 'month_sin',
   'month_cos', 'FIC2003_valeur_diff', 'MX2001_valeur_diff',
   'MX2002A_valeur_diff', 'MX2002B_valeur_diff', 'MX2002C_valeur_diff',
   'MX2002D_valeur_diff', 'MX2003A_valeur_diff', 'MX2003B_valeur_diff',
   'MX2003C_valeur_diff', 'temperature_ext_diff', 'LI2001_valeur_diff',
   'TI2001__2_valeur_diff']

# 1. List of ALL 33 Numerical Features (used for scaling)
ALL_NUMERICAL_FEATURES = [col for col in FEATURE_COLUMNS_WITH_DATETIME if col != 'datetime']

# 2. Define the 32 Input Features for the Model (excluding the target)
TARGET_FEATURE = ALL_NUMERICAL_FEATURES[-1] # 'TI2001__2_valeur_diff'
INPUT_FEATURES_FOR_MODEL = ALL_NUMERICAL_FEATURES[:-1] # This list now has 32 columns

WINDOW_SIZE = 60 # Placeholder window size
SCALER = MinMaxScaler() # Mock scaler, replace with your loaded scaler

# --- Utility Functions ---

def allowed_file(filename):
  return '.' in filename and \
     filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_gru_model():
  """Loads the GRU model from the .keras file."""
  global GRU_MODEL
  model_path = 'best_gru_model.keras'
  if GRU_MODEL is None and os.path.exists(model_path):
    app.logger.info(f"Loading model from {model_path}...")
    try:
      GRU_MODEL = tf.keras.models.load_model(model_path)
      app.logger.info("GRU Model loaded successfully.")
    except Exception as e:
      app.logger.error(f"Error loading GRU model: {e}")
      GRU_MODEL = None
  return GRU_MODEL

def preprocess_and_window_data(df):
  """
  Scales and windows the data for GRU prediction.
  """
  if df.empty:
    return np.array([]), np.array([])
  
  # STEP 1: Select ALL 33 Numerical Features for fitting the scaler 
  scaler_fit_data = df[ALL_NUMERICAL_FEATURES].values
  
  # Fit/Transform the full 33-feature dataset. 
  scaled_data_full = SCALER.fit_transform(scaler_fit_data) 

    # STEP 2: Slice the scaled data to get ONLY the 32 input features for the model (excluding the target)
  scaled_data_input = scaled_data_full[:, :-1]
  
  X = []
  
  # Get the number of windows M (e.g., 57990)
  M = len(scaled_data_input) - WINDOW_SIZE
    
  # Create time series windows
  for i in range(M):
    X.append(scaled_data_input[i:i + WINDOW_SIZE, :])
    
  # 💡 FIX: Explicitly slice the datetime array to match the M windows created
  # This guarantees that len(datetimes) == M == len(X)
  datetimes = df['datetime'].iloc[WINDOW_SIZE : WINDOW_SIZE + M].values
  
  return np.array(X), datetimes

# --- API Endpoints ---

@app.route('/api/upload-preview', methods=['POST'])
def upload_preview():
  if 'file' not in request.files:
    return jsonify({'status': 'error', 'error': 'No file part'}), 400
  file = request.files['file']
  if file.filename == '':
    return jsonify({'status': 'error', 'error': 'No selected file'}), 400
  
  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    
    try:
      df = pd.read_csv(filepath)
      df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Use up to 1000 rows for initial preview (as per DataChart logic)
      preview_df = df.head(1000).copy() 
      preview_df['datetime'] = preview_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

      preview_data = {
        'columns': preview_df.columns.tolist(),
        'data': preview_df.values.tolist(),
        'total_rows': len(df)
      }

      return jsonify({'status': 'preview_ready', 'preview_data': json.dumps(preview_data)})

    except Exception as e:
      return jsonify({'status': 'error', 'error': f'Error processing file: {e}'}), 500
  
  return jsonify({'status': 'error', 'error': 'Invalid file type'}), 400


@app.route('/api/run-model', methods=['POST'])
def run_model():
  # 1. Load Model
  model = load_gru_model()
  if model is None:
    return jsonify({'status': 'error', 'error_message': 'GRU model not loaded. Check model file path.'}), 500
  
  if 'file' not in request.files:
    return jsonify({'status': 'error', 'error_message': 'No file part in request.'}), 400
  
  file = request.files['file']
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
  
  try:
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # FIX (Kept from previous): Reset the index to ensure it is contiguous (0, 1, 2, ...)
    df = df.reset_index(drop=True) 

    # 2. Preprocess Data
    X_test, datetimes = preprocess_and_window_data(df)
    
    M = X_test.shape[0] # The number of data points to predict (e.g., 57990)
    
    if M == 0:
      return jsonify({'status': 'error', 'error_message': 'Insufficient data for windowing.'}), 500

    # 3. Predict
    app.logger.info(f"Starting prediction for {M} windows...")
    predictions_scaled = model.predict(X_test)
    
    # 4. Inverse Transform 
    target_index = len(ALL_NUMERICAL_FEATURES) - 1 # Index 32 (the 33rd column)
    
    # Create a dummy array matching the scaler dimensionality (33 columns)
    dummy_inversed = np.zeros((predictions_scaled.shape[0], len(ALL_NUMERICAL_FEATURES)))
    
    # Place the predictions in the target column
    dummy_inversed[:, target_index] = predictions_scaled.flatten()
    
    predictions = SCALER.inverse_transform(dummy_inversed)[:, target_index]
    app.logger.info("Prediction complete.")

    # --- Generate Results ---
    
    # A. Detailed Data (for Charting)
    target_column_name = TARGET_FEATURE
    
    # Extract actuals using explicit M length slice.
    actuals = df[target_column_name].iloc[WINDOW_SIZE : WINDOW_SIZE + M].values
        
        # New robust check to confirm array lengths match before DataFrame creation
    if not (len(datetimes) == M and len(actuals) == M and len(predictions) == M):
            # This will raise a descriptive error if the issue persists
            raise ValueError(f"Final array length mismatch: Predicted={len(predictions)}, Datetimes={len(datetimes)}, Actuals={len(actuals)}. Check data integrity/NaN values.")
    
    detailed_df = pd.DataFrame({
      'datetime': datetimes,
      'Actual_Value': actuals,
      'Predicted_Value': predictions
    })
    
    detailed_df['datetime'] = detailed_df['datetime'].astype(str)

    detailed_data = {
      'columns': detailed_df.columns.tolist(),
      'data': detailed_df.values.tolist(),
    }

    # B. Summary Data (for Table)
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    mae = np.mean(np.abs(actuals - predictions))
    
    summary_df = pd.DataFrame({
      'Metric': ['RMSE', 'MAE', 'Data_Points', 'Window_Size'],
      'Value': [f'{rmse:.4f}', f'{mae:.4f}', len(predictions), WINDOW_SIZE],
    })

    summary_data = {
      'columns': summary_df.columns.tolist(),
      'data': summary_df.values.tolist(),
    }

    return jsonify({
      'status': 'success',
      'summary_data': json.dumps(summary_data),
      'detailed_data': json.dumps(detailed_data),
      'logs': f"Model run finished.\nPredicted {len(predictions)} points.\nRMSE: {rmse:.4f}"
    })

  except Exception as e:
    app.logger.error(f"Error during model run: {e}", exc_info=True)
    return jsonify({'status': 'error', 'error_message': f'Model run failed: {e}'}), 500


if __name__ == '__main__':
  load_gru_model() 
  app.run(debug=True, port=4000)